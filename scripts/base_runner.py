"""
BaseRunner class containing shared logic for TrainingRunner and EvaluationRunner.

This module provides the common initialization and evaluation functionality that both
training and evaluation runners can inherit from, reducing code duplication.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import cv2
import torch

import execution_logger as logs
import numpy as np
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


@dataclass
class EpisodeStats:
    n_agents: int
    rewards: np.ndarray = field(init=False)
    steps: int = 0

    def __post_init__(self) -> None:
        self.rewards = np.zeros(self.n_agents, dtype=np.float32)

    def step(self) -> None:
        self.steps += 1

    def update_reward(self, reward: float | list[float] | np.ndarray) -> None:
        if np.isscalar(reward):
            # Single-agent case (broadcast to all)
            reward_vector = np.full(self.n_agents, reward, dtype=np.float32)
        else:
            reward_vector = np.asarray(reward, dtype=np.float32)

        self.rewards += np.asarray(reward_vector)

    def get_episode_reward(self) -> float:
        return float(np.sum(self.rewards))

    def summary(self) -> dict[str, float]:
        return {
            "episode_steps": self.steps,
            "episode_reward": self.get_episode_reward(),
            "episode_reward_mean": float(np.mean(self.rewards)),
            **{f"agent_{i}_reward": float(r) for i, r in enumerate(self.rewards)},
        }

    def reset(self) -> None:
        self.rewards[:] = 0
        self.steps = 0


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
        self.logger.info(
            f"[SEED {self.train_seed} | {self.eval_seed}] Setting up Runner"
        )

        # Create factory instances (each process needs its own)
        self.env_factory = EnvironmentFactory()
        self.network_factory = NetworkFactory()
        self.memory_factory = MemoryFactory()

        self.fps = self.env_config.record_video_fps

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

        self.recons_record = Record(
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
        self.recons_record.set_sub_directory(f"{self.train_seed}")

        # Save configurations if requested
        if save_configurations:
            self.record.save_configurations(configurations)

        # Create environments
        self.logger.info(
            f"[SEED {self.train_seed} | {self.eval_seed}] Loading Environment: {self.env_config.gym}"
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
            f"[SEED {self.train_seed} | {self.eval_seed}] Algorithm: {self.alg_config.algorithm}"
        )
        self.agent: Algorithm = self.network_factory.create_network(
            self.env.observation_space, self.env.action_num, self.alg_config, len(self.env.env.tasks)
        )

        # Validate agent creation
        if self.agent is None:
            raise ValueError(
                f"[SEED {self.train_seed} | {self.eval_seed}] Unknown agent for default algorithms {self.alg_config.algorithm}"
            )

        # Set up record with agent
        self.record.set_agent(self.agent)
        self.recons_record.set_agent(self.agent)

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
    ) -> dict[str, Any]:
        """
        Run a single evaluation episode and return detailed results.

        Args:
            episode_counter: Episode number for logging
            log_step: Training/evaluation step for logging context
            record_video: Whether to record video for this episode

        Returns:
            Dictionary with episode results including reward, states, actions, etc.
        """

        episode_states = []
        episode_actions = []
        episode_rewards: list[float] = []

        episode_stats = EpisodeStats(n_agents=self.env_eval.num_agents)

        # Reset environment
        state = self.env_eval.reset(training=False)

        all_done = False
        all_truncated = False

        while not all_done and not all_truncated:
            episode_stats.step()

            # Action selection
            available_actions = self.env_eval.get_available_actions()
            action_context = ActionContext(
                state=state, 
                evaluation=True, 
                available_actions=available_actions, 
                extras={"tasks": self.env_eval.env.tasks}
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

            all_done = all(done) if isinstance(done, list) else done
            all_truncated = all(truncated) if isinstance(truncated, list) else truncated
            episode_end = all_done or all_truncated

            episode_stats.update_reward(reward)

            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(hlp.get_device()).unsqueeze(0)
                recons_state = self.agent.autoencoder(state_tensor)["reconstructed_observation"]
                recons_last_frame = recons_state.squeeze()[-1] * 255 * 8
                recons_image = recons_last_frame.cpu().numpy().astype(np.uint8)
                recons_image = cv2.cvtColor(recons_image, cv2.COLOR_GRAY2BGR)

                # Mario
                # recons_image = cv2.resize(recons_image, (160, 144), interpolation=cv2.INTER_NEAREST)

                self.recons_record.log_video(recons_image)

            # Collect data for bias calculation
            episode_states.append(state)
            episode_actions.append(normalised_action)

            # Just taking the sum reward for processing bias
            episode_rewards.append(episode_stats.get_episode_reward())

            # Record video if requested
            if record_video and self.record is not None:
                frame = self.env_eval.grab_frame()
                overlay = overlay_info(
                    frame,
                    reward=f"{episode_stats.get_episode_reward():.1f}",
                    **self.env_eval.get_overlay_info(),
                )
                self.record.log_video(overlay)

        # Calculate bias and log results
        episode_results = {
            "episode_reward": episode_stats.get_episode_reward(),
            "episode_timesteps": episode_stats.steps,
            "episode_states": episode_states,
            "episode_actions": episode_actions,
            "episode_rewards": episode_rewards,
            "env_info": env_info,
        }

        if episode_end:
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
                    display=True,
                    **env_info,
                    **bias_data,
                    **episode_stats.summary(),
                )

            self.agent.episode_done()

        return episode_results

    def _evaluate_agent_episodes(
        self,
        log_step: int,
        video_label: str,
    ) -> dict[str, Any]:
        """
        Evaluate standard RL agent over multiple episodes.

        Args:
            log_step: Training/evaluation step for logging context
            video_label: Label for video recording

        Returns:
            Dictionary with aggregated evaluation results
        """
        if self.record is not None:
            frame = self.env_eval.grab_frame()
            self.record.start_video(video_label, frame, fps=self.fps)
            # Mario
            # recons_frame = self.env_eval.env.game_area().astype(np.uint8) * 8
            # recons_image = cv2.cvtColor(recons_frame, cv2.COLOR_GRAY2BGR)
            # recons_image = cv2.resize(recons_image, (160, 144), interpolation=cv2.INTER_NEAREST)
            # # Pokemon
            recons_frame = self.env_eval.env.screen.ndarray.transpose(2,0,1)[:1, :, :]
            recons_image = recons_frame.squeeze()
            recons_image = cv2.cvtColor(recons_image, cv2.COLOR_GRAY2BGR)
            self.recons_record.start_video(f"{video_label}_recons", recons_image, fps=self.fps)

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

        if self.record is not None:
            self.record.stop_video()
            self.recons_record.stop_video()

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
    ) -> dict[str, Any]:
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
