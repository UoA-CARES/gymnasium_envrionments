import time
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from base_runner import BaseRunner
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.repetition import EpisodeReplay
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class TrainingRunner(BaseRunner):
    """
    Handles the training loop for a single seed with integrated logging.

    This class encapsulates all training logic for a single seed, providing
    clean separation between orchestration (TrainingCoordinator) and execution (TrainingRunner).
    """

    def __init__(
        self,
        train_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        progress_queue: Queue | None = None,
        resume_path: str | None = None,
        save_configurations: bool = False,
        eval_seed: int | None = None,
    ):
        """
        Initialize TrainingRunner with all component creation and setup.

        Args:
            train_seed: Random seed for this training run
            configurations: Dictionary containing all parsed configurations
            base_log_dir: Base directory for logging
            progress_queue: Queue for progress updates (if any)
            resume_path: Path to resume from (if None, start fresh training)
            save_configurations: Whether to save configurations to disk
            eval_seed: Separate evaluation seed (if None, uses train_seed)
        """
        # Resolve eval_seed before calling super()
        eval_seed = eval_seed if eval_seed is not None else train_seed

        # Initialize the base runner
        super().__init__(
            train_seed=train_seed,
            eval_seed=eval_seed,
            configurations=configurations,
            base_log_dir=base_log_dir,
            save_configurations=save_configurations,
        )

        # TrainingRunner-specific setup
        self.progress_queue = progress_queue
        self.display = bool(self.env_config.display)

        # Create memory (needed for training)
        self.memory = self.memory_factory.create_memory(self.alg_config)

        # Handle resume logic - this must modify our local variables
        self.start_training_step = 0
        if resume_path is not None:
            self.start_training_step, self.memory = self._handle_resume(
                resume_path,
                self.alg_config.algorithm,
            )

        # Set up memory in record
        self.record.set_memory_buffer(self.memory)

        # Algorithm Training parameters
        self.max_steps_training = self.alg_config.max_steps_training
        self.max_steps_exploration = self.alg_config.max_steps_exploration
        self.number_steps_per_train_policy = (
            self.alg_config.number_steps_per_train_policy
        )
        self.batch_size = self.alg_config.batch_size
        self.G = self.alg_config.G  # pylint: disable=invalid-name

        # Evaluation parameters (some inherited from BaseRunner)
        self.number_steps_per_evaluation = (
            self.training_config.number_steps_per_evaluation
        )

        # Episode repetition parameters
        self.repetition_num_episodes = self.alg_config.repetition_num_episodes
        self.use_episode_repetition = self.repetition_num_episodes > 0

        self.logger.info(f"[SEED {self.train_seed}] training instance setup complete")

    def _handle_resume(
        self,
        data_path: str,
        algorithm: str,
    ) -> tuple[int, MemoryBuffer]:
        """
        Handle all resume logic and return starting step and loaded memory.

        Args:
            data_path: Path to the checkpoint data
            algorithm: Algorithm name for loading models

        Returns:
            Tuple of (starting_training_step, loaded_memory)
        """
        restart_path = Path(data_path) / str(self.train_seed)

        # Check if seed directory exists
        if not restart_path.exists():
            self.logger.warning(
                f"[SEED {self.train_seed}] No checkpoint found at {restart_path}, starting fresh training"
            )
            return 0, self.memory

        self.logger.info(
            f"[SEED {self.train_seed}] Restarting from path: {restart_path}"
        )

        self.logger.info(
            f"[SEED {self.train_seed}] Loading training and evaluation data"
        )
        self.record.load(restart_path)

        self.logger.info(f"[SEED {self.train_seed}] Loading memory buffer")
        try:
            loaded_memory = MemoryBuffer.load(restart_path / "memory", "memory")
        except FileNotFoundError:
            self.logger.warning(
                f"[SEED {self.train_seed}] No memory buffer found at {restart_path / 'memory'}, starting with empty memory"
            )
            loaded_memory = self.memory

        self.logger.info(f"[SEED {self.train_seed}] Loading agent models")
        try:
            self.agent.load_models(restart_path / "models" / "checkpoint", algorithm)
        except FileNotFoundError:
            self.logger.warning(
                f"[SEED {self.train_seed}] No agent models found at {restart_path / 'models' / 'checkpoint'}, starting with fresh models"
            )

        start_training_step = self.record.get_last_logged_step()
        self.logger.info(
            f"[SEED {self.train_seed}] Resuming from step: {start_training_step}"
        )

        return start_training_step, loaded_memory

    def _report_progress(self, episode: int, step: int, status: str) -> None:
        """Report progress to the main thread if a queue is provided."""
        if self.progress_queue is not None:
            self.progress_queue.put(
                {
                    "seed": self.train_seed,
                    "episode": episode,
                    "step": step,
                    "total": self.max_steps_training,
                    "status": status,
                }
            )

    def run_training(self) -> None:
        """
        Execute the main training loop with proper cleanup.

        This is the main entry point that orchestrates the entire training process
        for this seed, including exploration, training, and evaluation phases.
        """
        self.logger.info(
            f"Training {self.max_steps_training} Exploration {self.max_steps_exploration} "
            f"Evaluation {self.number_steps_per_evaluation}"
        )

        self._report_progress(0, 0, "starting")

        start_time = time.time()

        # Initialize training state
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        state = self.env.reset()
        episode_start = time.time()

        # Episode repetition tracking
        repeating = False
        repetition_counter = 0
        repetition_buffer = EpisodeReplay()
        repeat = False
        episode_repetitions = 0

        # Main training loop
        train_step_counter = self.start_training_step
        for train_step_counter in range(
            self.start_training_step, int(self.max_steps_training)
        ):
            episode_timesteps += 1

            info: dict = {}

            status = (
                "training"
                if train_step_counter >= self.max_steps_exploration
                else "exploration"
            )
            self._report_progress(episode_num + 1, train_step_counter + 1, status)

            # Determine action based on training phase
            if train_step_counter < self.max_steps_exploration:
                normalised_action, denormalised_action = (
                    self._select_exploration_action(train_step_counter)
                )
            elif (
                self.use_episode_repetition
                and repeat
                and repetition_buffer.has_best_episode()
            ):
                normalised_action, denormalised_action = self._select_repitition_action(
                    episode_num, episode_timesteps, repetition_buffer
                )
                if episode_timesteps >= len(repetition_buffer.best_actions):
                    repeat = False
            else:
                normalised_action, denormalised_action = self._select_policy_action(
                    state
                )

            # Record action and execute step
            repetition_buffer.record_action(denormalised_action)
            info["repeated"] = episode_repetitions

            next_state, reward_extrinsic, done, truncated, env_info = self.env.step(
                denormalised_action
            )

            if self.display:
                self.env.render()

            # Calculate total reward (extrinsic + intrinsic)
            intrinsic_reward = 0
            if train_step_counter > self.max_steps_exploration:
                intrinsic_reward = self.agent.get_intrinsic_reward(
                    state, normalised_action, next_state
                )

            total_reward = reward_extrinsic + intrinsic_reward

            # Store experience in memory
            self.memory.add(state, normalised_action, total_reward, next_state, done)

            state = next_state
            episode_reward += reward_extrinsic
            info["intrinsic_reward"] = intrinsic_reward

            # Train policy if conditions are met
            if (
                train_step_counter >= self.max_steps_exploration
                and (train_step_counter + 1) % self.number_steps_per_train_policy == 0
            ):
                train_info = self._update_policy(
                    train_step_counter,
                    episode_num,
                    episode_timesteps,
                    episode_reward,
                    done or truncated,
                )
                info |= train_info

            # Evaluate agent periodically
            if (train_step_counter + 1) % self.number_steps_per_evaluation == 0:
                self._report_progress(
                    episode_num + 1, train_step_counter + 1, "evaluation"
                )
                self._run_evaluation(train_step_counter)

            # Handle episode completion
            if done or truncated:
                episode_time = time.time() - episode_start

                # Log training data
                self.record.log_train(
                    total_steps=train_step_counter + 1,
                    episode=episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward=episode_reward,
                    episode_time=episode_time,
                    **env_info,
                    **info,
                    display=True,
                )

                # Handle episode repetition logic
                repeating, repeat, repetition_counter, episode_repetitions = (
                    self._finalise_episode(
                        train_step_counter,
                        episode_reward,
                        repetition_buffer,
                        repeating,
                        repeat,
                        repetition_counter,
                        episode_repetitions,
                    )
                )

                # Reset for next episode
                state = self.env.reset()
                episode_timesteps = 0
                episode_reward = 0
                episode_num += 1
                self.agent.episode_done()
                episode_start = time.time()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(
            f"Training completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
        )

        # Save record and report completion
        self.record.save()
        self._report_progress(episode_num + 1, train_step_counter + 1, "done")

    def _select_exploration_action(self, train_step_counter: int) -> tuple:
        """Handle exploration phase action selection."""
        self.logger.info(
            f"Running Exploration Steps {train_step_counter + 1}/{self.max_steps_exploration}"
        )

        denormalised_action = self.env.sample_action()
        normalised_action = denormalised_action

        if self.apply_action_normalisation:
            normalised_action = hlp.normalize(
                denormalised_action,
                self.env.max_action_value,
                self.env.min_action_value,
            )

        return normalised_action, denormalised_action

    def _select_repitition_action(
        self, episode_num: int, episode_timesteps: int, repetition_buffer: EpisodeReplay
    ) -> tuple:
        """Handle episode repetition action selection."""
        self.logger.info(
            f"Repeating Episode {episode_num} Step {episode_timesteps}/{len(repetition_buffer.best_actions)}"
        )

        denormalised_action = repetition_buffer.replay_best_episode(
            episode_timesteps - 1
        )

        # For repetition, assume we stored denormalized actions
        normalised_action = denormalised_action
        if self.apply_action_normalisation:
            normalised_action = hlp.normalize(
                denormalised_action,
                self.env.max_action_value,
                self.env.min_action_value,
            )

        return normalised_action, denormalised_action

    def _select_policy_action(self, state) -> tuple:
        """Handle policy-based action selection."""
        available_actions = self.env.get_available_actions()
        action_context = ActionContext(
            state=state, evaluation=False, available_actions=available_actions
        )
        normalised_action = self.agent.select_action_from_policy(action_context)

        denormalised_action = normalised_action
        if self.apply_action_normalisation:
            denormalised_action = hlp.denormalize(
                normalised_action, self.env.max_action_value, self.env.min_action_value
            )

        return normalised_action, denormalised_action

    def _update_policy(
        self,
        train_step_counter: int,
        episode_num: int,
        episode_timesteps: int,
        episode_reward: float,
        episode_done: bool,
    ) -> dict:
        """Execute policy training step."""
        training_context = TrainingContext(
            memory=self.memory,
            batch_size=self.batch_size,
            training_step=train_step_counter,
            episode=episode_num + 1,
            episode_steps=episode_timesteps,
            episode_reward=episode_reward,
            episode_done=episode_done,
        )

        train_info = {}
        for _ in range(self.G):
            train_info = self.agent.train_policy(training_context)

        return train_info

    def _finalise_episode(
        self,
        train_step_counter: int,
        episode_reward: float,
        repetition_buffer: EpisodeReplay,
        repeating: bool,
        repeat: bool,
        repetition_counter: int,
        episode_repetitions: int,
    ) -> tuple:
        """Handle episode completion and repetition logic."""
        if repeating:
            repeat = True
            repetition_counter += 1
            if repetition_counter >= self.repetition_num_episodes:
                repeat = False
                repeating = False
                repetition_counter = 0
        elif train_step_counter > self.max_steps_exploration:
            repeat = repetition_buffer.finish_episode(episode_reward)
            repeating = repeat
            episode_repetitions = (
                episode_repetitions + 1
                if repeat and self.use_episode_repetition
                else episode_repetitions
            )
        else:
            repetition_buffer.finish_episode(episode_reward)

        return repeating, repeat, repetition_counter, episode_repetitions

    def _run_evaluation(self, train_step_counter: int) -> None:
        """Execute evaluation phase."""
        self.logger.info("*************--Evaluation Loop--*************")

        if self.agent.policy_type == "usd":
            self._evaluate_usd_skills(train_step_counter, f"{train_step_counter + 1}")
        else:
            self._evaluate_agent_episodes(
                train_step_counter, f"{train_step_counter + 1}"
            )

        self.logger.info("--------------------------------------------")
