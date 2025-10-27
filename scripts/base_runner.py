"""
BaseRunner class containing shared logic for TrainingRunner and EvaluationRunner.

This module provides the common initialization and evaluation functionality that both
training and evaluation runners can inherit from, reducing code duplication.
"""

from abc import ABC
from typing import Any, Dict, List

import training_logger as logs
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.network_factory import NetworkFactory
from cares_reinforcement_learning.util.training_context import ActionContext
from environments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig
from util.overlay import overlay_info
from util.record import Record


class BaseRunner(ABC):
    """
    Abstract base class containing shared initialization and evaluation logic.

    This class provides common initialization and evaluation methods for both TrainingRunner
    and EvaluationRunner, allowing them to share core setup and evaluation logic while
    maintaining their specific purposes.
    """

    def __init__(
        self,
        train_seed: int,
        eval_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        save_configurations: bool = False,
        num_eval_episodes: int | None = None,
    ):
        """
        Initialize BaseRunner with common setup logic.

        Args:
            train_seed: Random seed for this training run
            eval_seed: Random seed for evaluation (if None, uses train_seed)
            configurations: Dictionary containing all parsed configurations
            base_log_dir: Base directory for logging
            save_configurations: Whether to save configurations to disk
            num_episodes: Number of episodes for evaluation (if None, uses config default)
        """
        # Extract configurations
        self.env_config: GymEnvironmentConfig = configurations["env_config"]
        self.training_config: TrainingConfig = configurations["train_config"]
        self.alg_config: AlgorithmConfig = configurations["alg_config"]

        self.train_seed = train_seed
        self.eval_seed = eval_seed

        # Set up logging
        self.logger = logs.get_seed_logger()
        self.logger.info(f"[SEED {self.train_seed}] Setting up Runner")

        # Create factory instances (each process needs its own)
        self.env_factory = EnvironmentFactory()
        self.network_factory = NetworkFactory()
        self.memory_factory = MemoryFactory()

        # Create record for this seed
        self.record = Record(
            base_directory=base_log_dir,
            algorithm=self.alg_config.algorithm,
            task=self.env_config.task,
            agent=None,
            record_video=self.training_config.record_eval_video,
            record_checkpoints=bool(self.env_config.save_train_checkpoints),
            checkpoint_interval=self.training_config.checkpoint_interval,
            logger=self.logger,
        )

        # Set up record with subdirectory
        self.record.set_sub_directory(f"{self.train_seed}")

        # Save configurations if requested
        if save_configurations:
            self.record.save_configurations(configurations)

        # Create environments
        self.logger.info(
            f"[SEED {self.train_seed}] Loading Environment: {self.env_config.gym}"
        )
        self.env, self.env_eval = self.env_factory.create_environment(
            self.env_config, self.alg_config.image_observation
        )

        # Set the seed for everything
        hlp.set_seed(self.train_seed)
        self.env.set_seed(self.train_seed)
        self.env_eval.set_seed(self.eval_seed)

        # Create the algorithm
        self.logger.info(
            f"[SEED {self.train_seed}] Algorithm: {self.alg_config.algorithm}"
        )
        self.agent: Algorithm = self.network_factory.create_network(
            self.env_eval.observation_space, self.env_eval.action_num, self.alg_config
        )

        # Validate agent creation
        if self.agent is None:
            raise ValueError(
                f"Unknown agent for default algorithms {self.alg_config.algorithm}"
            )

        # Set up record with agent
        self.record.set_agent(self.agent)

        # Runtime behavior - action normalisation
        self.apply_action_normalisation = self.agent.policy_type in ["policy", "usd"]

        # Evaluation parameters
        self.number_eval_episodes = (
            num_eval_episodes
            if num_eval_episodes is not None
            else self.training_config.number_eval_episodes
        )

    def _run_single_episode_evaluation(
        self,
        episode_counter: int,
        log_step: int,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode and return detailed results.

        Args:
            episode_counter: Episode number for logging
            log_step: Training/evaluation step for logging context
            record_video: Whether to record video for this episode

        Returns:
            Dictionary with episode results including reward, states, actions, etc.
        """
        episode_timesteps = 0
        episode_reward = 0.0
        done = False
        truncated = False

        episode_states = []
        episode_actions = []
        episode_rewards: List[float] = []

        # Reset environment
        state = self.env_eval.reset()

        while not done and not truncated:
            episode_timesteps += 1

            # Action selection
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

            # Step environment
            state, reward, done, truncated, env_info = self.env_eval.step(
                denormalised_action
            )
            episode_reward += reward

            # Collect data for bias calculation
            episode_states.append(state)
            episode_actions.append(normalised_action)
            episode_rewards.append(reward)

            # Record video if requested
            if record_video and self.record is not None:
                frame = self.env_eval.grab_frame()
                overlay = overlay_info(
                    frame,
                    reward=f"{episode_reward:.1f}",
                    **self.env_eval.get_overlay_info(),
                )
                self.record.log_video(overlay)

        # Calculate bias and log results
        episode_results = {
            "episode_reward": episode_reward,
            "episode_timesteps": episode_timesteps,
            "episode_states": episode_states,
            "episode_actions": episode_actions,
            "episode_rewards": episode_rewards,
            "env_info": env_info if done or truncated else {},
        }

        if done or truncated:
            # Calculate bias
            bias_data = self.agent.calculate_bias(
                episode_states, episode_actions, episode_rewards
            )
            episode_results["bias_data"] = bias_data

            # Log evaluation information
            if self.record is not None:
                self.record.log_eval(
                    total_steps=log_step,
                    episode=episode_counter + 1,
                    episode_reward=episode_reward,
                    display=True,
                    **env_info,
                    **bias_data,
                )

            self.agent.episode_done()

        return episode_results

    def _evaluate_agent_episodes(
        self,
        log_step: int,
        video_label: str,
    ) -> Dict[str, Any]:
        """
        Evaluate standard RL agent over multiple episodes.

        Args:
            log_step: Training/evaluation step for logging context
            video_label: Label for video recording
            num_episodes: Number of episodes to run (uses self.number_eval_episodes if None)

        Returns:
            Dictionary with aggregated evaluation results
        """
        self.env_eval.reset(training=False)

        if self.record is not None:
            frame = self.env_eval.grab_frame()
            self.record.start_video(video_label, frame)

            log_path = self.record.current_sub_directory
            self.env_eval.set_log_path(log_path, log_step)

        episode_rewards = []
        total_reward = 0.0
        all_bias_data = []

        for eval_episode_counter in range(self.number_eval_episodes):
            episode_results = self._run_single_episode_evaluation(
                episode_counter=eval_episode_counter,
                log_step=log_step,
                record_video=(eval_episode_counter == 0),  # Only record first episode
            )

            episode_reward = episode_results["episode_reward"]
            episode_rewards.append(episode_reward)
            total_reward += episode_reward

            if "bias_data" in episode_results:
                all_bias_data.append(episode_results["bias_data"])

            # Reset environment for next episode
            self.env_eval.reset()

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
            "bias_data": all_bias_data,
        }

    def _evaluate_usd_skills(
        self,
        log_step: int,
        video_label: str,
    ) -> Dict[str, Any]:
        """
        Evaluate USD (Unsupervised Skill Discovery) agent skills.

        Args:
            log_step: Training/evaluation step for logging context
            video_label: Base label for video recording

        Returns:
            Dictionary with skill evaluation results
        """
        self.env_eval.reset(training=False)
        skill_results = []
        total_reward = 0.0

        for skill_counter, skill in enumerate(range(self.agent.num_skills)):
            self.agent.set_skill(skill, evaluation=True)

            self.logger.info(f"Evaluating skill {skill + 1}/{self.agent.num_skills}")

            if self.record is not None:
                frame = self.env_eval.grab_frame()
                skill_video_label = f"{video_label}_skill_{skill}"
                self.record.start_video(skill_video_label, frame)

                log_path = self.record.current_sub_directory
                self.env_eval.set_log_path(log_path, skill)

            # Run one episode per skill
            episode_results = self._run_single_episode_evaluation(
                episode_counter=skill_counter,
                log_step=log_step,
                record_video=True,
            )

            episode_reward = episode_results["episode_reward"]
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
