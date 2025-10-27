"""
EvaluationRunner class for sequential evaluation of multiple model checkpoints.

This module provides a clean interface for loading different model checkpoints
from a training run and evaluating each one, allowing for performance tracking
across training steps.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import training_logger as logs
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from cares_reinforcement_learning.util.training_context import ActionContext
from environments.environment_factory import EnvironmentFactory
from natsort import natsorted
from util.configurations import GymEnvironmentConfig
from util.overlay import overlay_info
from util.record import Record


class EvaluationRunner:
    """
    Handles sequential evaluation of multiple model checkpoints.

    This class loads different model checkpoints from a training run and evaluates
    each one, providing performance tracking across different training steps.
    """

    def __init__(
        self,
        eval_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        former_base_path: str,
        num_episodes: int | None = None,
        save_configurations: bool = False,
    ):
        """
        Initialize EvaluationRunner for sequential checkpoint evaluation.

        Args:
            seed: Random seed for this evaluation run
            configurations: Dictionary containing all parsed configurations
            model_base_path: Base path to the trained model directory (contains seed subdirs)
            base_log_dir: Base directory for logging evaluation results
            num_episodes: Number of episodes to run per checkpoint (if None, uses config default)
        """
        # Extract configurations
        env_config: GymEnvironmentConfig = configurations["env_config"]
        training_config: TrainingConfig = configurations["train_config"]
        alg_config: AlgorithmConfig = configurations["alg_config"]

        self.eval_seed = eval_seed
        self.former_model_base_path = Path(former_base_path)
        self.former_model_seed_path = self.former_model_base_path / str(eval_seed)

        # Set up logging
        self.eval_logger = logs.get_seed_logger()
        self.eval_logger.info(f"[SEED {self.eval_seed}] Starting checkpoint evaluation")

        # Create factory instances
        env_factory = EnvironmentFactory()
        network_factory = NetworkFactory()

        # Create record for evaluation results
        self.record = Record(
            base_directory=base_log_dir,
            algorithm=alg_config.algorithm,
            task=env_config.task,
            agent=None,
            record_video=training_config.record_eval_video,
            record_checkpoints=False,  # No checkpoints needed for evaluation
            checkpoint_interval=0,
            logger=self.eval_logger,
        )

        # Set up record with subdirectory for this seed
        self.record.set_sub_directory(f"{self.eval_seed}")

        # Save configurations if requested
        if save_configurations:
            self.record.save_configurations(configurations)

        # Create the environment (only need eval environment)
        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Loading Environment: {env_config.gym}"
        )
        _, self.env_eval = env_factory.create_environment(
            env_config, alg_config.image_observation
        )

        # Set the seed for everything
        hlp.set_seed(self.eval_seed)
        self.env_eval.set_seed(self.eval_seed)

        # Create the algorithm
        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Algorithm: {alg_config.algorithm}"
        )
        self.agent: Algorithm = network_factory.create_network(
            self.env_eval.observation_space, self.env_eval.action_num, alg_config
        )

        # Validate agent creation
        if self.agent is None:
            raise ValueError(
                f"Unknown agent for default algorithms {alg_config.algorithm}"
            )

        # Set up record with agent
        self.record.set_agent(self.agent)

        # Store configurations
        self.alg_config = alg_config
        self.env_config = env_config
        self.training_config = training_config

        # Runtime behavior
        self.apply_action_normalisation = self.agent.policy_type in ["policy", "usd"]

        # Evaluation parameters
        self.number_eval_episodes = (
            num_episodes
            if num_episodes is not None
            else training_config.number_eval_episodes
        )

        self.eval_logger.info(
            f"[SEED {self.eval_seed}] will run {self.number_eval_episodes} episodes per checkpoint"
        )

    def discover_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Discover all available model checkpoints for this seed.

        Returns:
            List of checkpoint info dictionaries with 'path', 'step', 'type' keys
        """
        if not self.former_model_seed_path.exists():
            raise FileNotFoundError(
                f"No model directory found for seed {self.eval_seed} at {self.former_model_seed_path}"
            )

        models_path = self.former_model_seed_path / "models"
        if not models_path.exists():
            raise FileNotFoundError(f"No models directory found at {models_path}")

        checkpoints: List[Dict[str, Any]] = []

        folders = list(models_path.glob("*"))

        # # Sort folders and remove the final and best model folders
        folders = natsorted(folders)[:-2]

        for folder in folders:
            step = int(folder.name)
            if step is not None:
                checkpoints.append(
                    {
                        "path": folder,
                        "step": step,
                    }
                )

        return checkpoints

    def load_checkpoint(self, checkpoint_info: Dict[str, Any]) -> bool:
        """
        Load a specific model checkpoint into the agent.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            True if loading succeeded, False otherwise
        """
        checkpoint_path = checkpoint_info["path"]
        step = checkpoint_info["step"]

        self.eval_logger.info(f"[SEED {self.eval_seed}] (step {step})")

        try:
            self.agent.load_models(checkpoint_path, self.alg_config.algorithm)
            self.eval_logger.info(
                f"[SEED {self.eval_seed}] Successfully loaded checkpoint: {step}"
            )
            return True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.eval_logger.warning(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {step}: {e}"
            )
            return False

    def evaluate_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            Dictionary with evaluation results
        """
        step = checkpoint_info["step"]

        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Evaluating checkpoint: (step {step})"
        )

        start_time = time.time()

        if self.agent.policy_type == "usd":
            results = self._evaluate_usd_checkpoint(checkpoint_info["step"])
        else:
            results = self._evaluate_agent_checkpoint(checkpoint_info["step"])

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Completed evaluation of {step}: "
            f"Avg reward: {results.get('avg_reward', 'N/A'):.2f}, "
            f"Time: {evaluation_time:.1f}s"
        )

    def _evaluate_agent_checkpoint(self, total_step: int) -> Dict[str, Any]:
        """
        Evaluate a standard RL agent checkpoint.

        Args:
            total_step: The total step of the checkpoint being evaluated

        Returns:
            Dictionary with evaluation results
        """

        state = self.env_eval.reset(training=False)

        if self.record is not None:
            frame = self.env_eval.grab_frame()
            self.record.start_video(f"{total_step}", frame)

            log_path = self.record.current_sub_directory
            self.env_eval.set_log_path(log_path, total_step)

        episode_rewards = []
        total_reward = 0.0

        for eval_episode_counter in range(self.number_eval_episodes):
            episode_timesteps = 0
            episode_reward = 0.0
            done = False
            truncated = False

            episode_states = []
            episode_actions = []
            episode_rewards_detailed: list[float] = []

            while not done and not truncated:
                episode_timesteps += 1

                available_actions = self.env_eval.get_available_actions()
                action_context = ActionContext(
                    state=state, evaluation=True, available_actions=available_actions
                )
                normalised_action = self.agent.select_action_from_policy(action_context)

                denormalised_action = (
                    hlp.denormalize(
                        normalised_action,
                        self.env_eval.max_action_value,
                        self.env_eval.min_action_value,
                    )
                    if self.apply_action_normalisation
                    else normalised_action
                )

                state, reward, done, truncated, env_info = self.env_eval.step(
                    denormalised_action
                )
                episode_reward += reward

                # For Bias Calculation
                episode_states.append(state)
                episode_actions.append(normalised_action)
                episode_rewards_detailed.append(reward)

                # Record video for first episode only
                if eval_episode_counter == 0 and self.record is not None:
                    frame = self.env_eval.grab_frame()
                    overlay = overlay_info(
                        frame,
                        reward=f"{episode_reward:.1f}",
                        **self.env_eval.get_overlay_info(),
                    )
                    self.record.log_video(overlay)

                if done or truncated:
                    # Calculate bias
                    bias_data = self.agent.calculate_bias(
                        episode_states,
                        episode_actions,
                        episode_rewards_detailed,
                    )

                    # Log evaluation information
                    if self.record is not None:
                        self.record.log_eval(
                            total_steps=total_step,
                            episode=eval_episode_counter + 1,
                            episode_reward=episode_reward,
                            display=True,
                            **env_info,
                            **bias_data,
                        )

                    episode_rewards.append(episode_reward)
                    total_reward += episode_reward

                    # Reset environment for next episode
                    state = self.env_eval.reset()
                    episode_reward = 0.0
                    episode_timesteps = 0

                    self.agent.episode_done()

        if self.record is not None:
            self.record.stop_video()

        # Calculate statistics
        if episode_rewards:
            avg_reward = total_reward / len(episode_rewards)
            max_reward = max(episode_rewards)
            min_reward = min(episode_rewards)
            std_reward = (
                sum((r - avg_reward) ** 2 for r in episode_rewards)
                / len(episode_rewards)
            ) ** 0.5
        else:
            avg_reward = max_reward = min_reward = std_reward = 0.0

        return {
            "episode_rewards": episode_rewards,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "std_reward": std_reward,
            "total_episodes": len(episode_rewards),
        }

    def _evaluate_usd_checkpoint(self, total_step: int) -> Dict[str, Any]:
        """
        Evaluate a USD (Unsupervised Skill Discovery) agent checkpoint.

        Args:
            total_step: The total step of the checkpoint being evaluated

        Returns:
            Dictionary with evaluation results
        """

        self.env_eval.reset(training=False)
        skill_results = []
        total_reward = 0.0

        for skill_counter, skill in enumerate(range(self.agent.num_skills)):
            self.agent.set_skill(skill, evaluation=True)

            self.eval_logger.info(
                f"[SEED {self.eval_seed}] Evaluating skill {skill + 1}/{self.agent.num_skills} "
                f"for checkpoint {total_step}"
            )

            if self.record is not None:
                frame = self.env_eval.grab_frame()
                video_label = f"{total_step+1}-{skill}"
                self.record.start_video(video_label, frame)

                log_path = self.record.current_sub_directory
                self.env_eval.set_log_path(log_path, total_step)

            # Run one episode per skill
            episode_reward = self._run_single_skill_episode(skill, total_step)

            skill_results.append({"skill": skill, "reward": episode_reward})
            total_reward += episode_reward

            if self.record is not None:
                self.record.stop_video()

            # Reset environment for next skill
            self.env_eval.reset()

        # Calculate statistics
        if skill_results:
            avg_skill_reward = total_reward / len(skill_results)
            max_skill_reward = max(r["reward"] for r in skill_results)
            min_skill_reward = min(r["reward"] for r in skill_results)
        else:
            avg_skill_reward = max_skill_reward = min_skill_reward = 0.0

        return {
            "skill_results": skill_results,
            "avg_skill_reward": avg_skill_reward,
            "max_skill_reward": max_skill_reward,
            "min_skill_reward": min_skill_reward,
            "total_skills": len(skill_results),
        }

    def _run_single_skill_episode(self, skill: int, log_step: int) -> float:
        """
        Run a single episode for a USD skill.

        Args:
            skill: Skill index
            log_step: Step number for logging

        Returns:
            Episode reward
        """
        episode_timesteps = 0
        episode_reward = 0.0
        done = False
        truncated = False

        state = self.env_eval.reset()

        while not done and not truncated:
            episode_timesteps += 1

            available_actions = self.env_eval.get_available_actions()
            action_context = ActionContext(
                state=state, evaluation=True, available_actions=available_actions
            )
            normalised_action = self.agent.select_action_from_policy(action_context)

            denormalised_action = (
                hlp.denormalize(
                    normalised_action,
                    self.env_eval.max_action_value,
                    self.env_eval.min_action_value,
                )
                if self.apply_action_normalisation
                else normalised_action
            )

            state, reward, done, truncated, env_info = self.env_eval.step(
                denormalised_action
            )
            episode_reward += reward

            if self.record is not None:
                frame = self.env_eval.grab_frame()
                overlay = overlay_info(
                    frame,
                    reward=f"{episode_reward:.1f}",
                    **self.env_eval.get_overlay_info(),
                )
                self.record.log_video(overlay)

        # Log skill evaluation
        if done or truncated and self.record is not None:
            self.record.log_eval(
                total_steps=log_step,
                episode=skill + 1,
                episode_reward=episode_reward,
                display=True,
                **env_info,
            )

        self.agent.episode_done()
        return episode_reward

    def run_evaluation(self) -> None:
        """
        Execute sequential evaluation of all discovered checkpoints.

        Returns:
            List of evaluation results for each checkpoint
        """
        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Starting sequential checkpoint evaluation"
        )

        # Discover all checkpoints
        checkpoints = self.discover_checkpoints()

        if not checkpoints:
            self.eval_logger.warning(
                f"[SEED {self.eval_seed}] No checkpoints found to evaluate"
            )
            return

        successful_evaluations = 0
        for i, checkpoint_info in enumerate(checkpoints):
            self.eval_logger.info(
                f"[SEED {self.eval_seed}] Processing checkpoint {i + 1}/{len(checkpoints)}: "
            )

            # Load the checkpoint
            if not self.load_checkpoint(checkpoint_info):
                self.eval_logger.error(
                    f"[SEED {self.eval_seed}] Failed to load checkpoint {checkpoint_info['step']}, skipping"
                )
                continue

            # Evaluate the checkpoint
            try:
                self.evaluate_checkpoint(checkpoint_info)
                successful_evaluations += 1
            except Exception as e:
                self.eval_logger.error(
                    f"[SEED {self.eval_seed}] Failed to evaluate checkpoint {checkpoint_info['step']}: {e}"
                )
                continue

        # Save all results
        self.record.save()

        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Sequential evaluation completed. "
            f"Successfully evaluated {successful_evaluations}/{len(checkpoints)} checkpoints"
        )

    def run_test(self) -> None:
        """
        Execute testing (alias for evaluation for now).

        This is kept separate in case future differentiation is needed
        between evaluation and testing protocols.
        """
        self.eval_logger.info(
            f"[SEED {self.eval_seed}] Starting testing with {self.number_eval_episodes} episodes per checkpoint"
        )
        self.run_evaluation()
