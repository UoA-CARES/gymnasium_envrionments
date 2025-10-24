import logging
import time
from multiprocessing.queues import Queue

import training_logger as logs
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.util.repetition import EpisodeReplay
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)
from environments.gym_environment import GymEnvironment
from environments.multimodal_wrapper import MultiModalWrapper
from util.overlay import overlay_info
from util.record import Record


class TrainingLoop:
    """
    Handles the training loop for a single seed with integrated logging.

    This class encapsulates all training logic for a single seed, providing
    clean separation between orchestration (TrainingRunner) and execution (TrainingLoop).
    """

    def __init__(
        self,
        env: GymEnvironment | MultiModalWrapper,
        env_eval: GymEnvironment | MultiModalWrapper,
        agent: Algorithm,
        memory: MemoryBuffer,
        record: Record,
        train_config: TrainingConfig,
        alg_config: AlgorithmConfig,
        seed: int,
        progress_queue: Queue | None = None,
        logger: logging.Logger | None = None,
        display: bool = False,
        start_training_step: int = 0,
    ):
        """
        Initialize the training loop for a single seed.

        Args:
            env: Training environment
            env_eval: Evaluation environment
            agent: RL algorithm/agent
            memory: Memory buffer for experiences
            record: Recording system for metrics/videos
            train_config: Training configuration (values extracted and cached)
            alg_config: Algorithm configuration (values extracted and cached)
            seed: Random seed for this training loop
            progress_queue: Queue for progress updates (if any)
            logger: Logger for this training loop (if None, uses seed logger)
            display: Whether to render environment
            start_training_step: Step to start training from (for resume)
        """
        # Core dependencies
        self.env = env
        self.env_eval = env_eval

        self.agent = agent
        self.memory = memory
        self.record = record

        # Runtime behavior
        self.display = display
        self.start_training_step = start_training_step

        self.apply_action_normalisation = self.agent.policy_type in ["policy", "usd"]

        # Set up logging
        self.seed = seed
        self.progress_queue = progress_queue

        if logger is None:
            self.logger = logs.get_seed_logger()  # Use general seed logger for now
        else:
            self.logger = logger

        # Create exploration logger as child of main logger
        # self.exploration_logger = logging.getLogger(f"{self.logger.name}.exploration")

        # Algorithm Training parameters
        self.max_steps_training = alg_config.max_steps_training
        self.max_steps_exploration = alg_config.max_steps_exploration
        self.number_steps_per_train_policy = alg_config.number_steps_per_train_policy

        self.batch_size = alg_config.batch_size
        self.G = alg_config.G  # pylint: disable=invalid-name

        # Evaluation parameters
        self.number_eval_episodes = train_config.number_eval_episodes
        self.number_steps_per_evaluation = train_config.number_steps_per_evaluation

        # Episode repetition parameters
        self.repetition_num_episodes = alg_config.repetition_num_episodes
        self.use_episode_repetition = self.repetition_num_episodes > 0

    def _report_progress(self, episode: int, step: int, status: str) -> None:
        """Report progress to the main thread if a queue is provided."""
        if self.progress_queue is not None:
            self.progress_queue.put(
                {
                    "seed": self.seed,
                    "episode": episode,
                    "step": step,
                    "total": self.max_steps_training,
                    "status": status,
                }
            )

    def run_training(self) -> None:
        """
        Execute the main training loop.

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

        self._report_progress(episode_num + 1, train_step_counter + 1, "complete")

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

    def _run_evaluation(self, train_step_counter: int) -> None:
        """Execute evaluation phase."""
        self.logger.info("*************--Evaluation Loop--*************")

        if self.agent.policy_type == "usd":
            self._evaluate_usd(train_step_counter)
        else:
            self._evaluate_agent(train_step_counter)

        self.logger.info("--------------------------------------------")

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

    def _evaluate_usd(self, total_steps: int) -> None:
        """
        Evaluate USD (Unsupervised Skill Discovery) agent.

        Args:
            total_steps: Current training step count
        """
        state = self.env_eval.reset(training=False)

        for skill_counter, skill in enumerate(range(self.agent.num_skills)):
            episode_timesteps = 0
            episode_reward = 0
            episode_num = 0
            done = False
            truncated = False

            self.agent.set_skill(skill, evaluation=True)

            if self.record is not None:
                frame = self.env_eval.grab_frame()
                self.record.start_video(f"{total_steps+1}-{skill}", frame)

                log_path = self.record.current_sub_directory
                self.env_eval.set_log_path(log_path, total_steps + 1)

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

                if done or truncated:
                    # Log evaluation information
                    if self.record is not None:
                        self.record.log_eval(
                            total_steps=total_steps + 1,
                            episode=skill_counter + 1,
                            episode_reward=episode_reward,
                            display=True,
                            **env_info,
                        )

                        self.record.stop_video()

                    # Reset environment
                    state = self.env_eval.reset()
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                    self.agent.episode_done()

    def _evaluate_agent(self, total_steps: int) -> None:
        """
        Evaluate standard RL agent.

        Args:
            total_steps: Current training step count
        """
        state = self.env_eval.reset(training=False)

        if self.record is not None:
            frame = self.env_eval.grab_frame()
            self.record.start_video(f"{total_steps + 1}", frame)

            log_path = self.record.current_sub_directory
            self.env_eval.set_log_path(log_path, total_steps + 1)

        for eval_episode_counter in range(self.number_eval_episodes):
            episode_timesteps = 0
            episode_reward = 0
            episode_num = 0
            done = False
            truncated = False

            episode_states = []
            episode_actions = []
            episode_rewards: list[float] = []

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
                episode_rewards.append(reward)

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
                        episode_rewards,
                    )

                    # Log evaluation information
                    if self.record is not None:
                        self.record.log_eval(
                            total_steps=total_steps + 1,
                            episode=eval_episode_counter + 1,
                            episode_reward=episode_reward,
                            display=True,
                            **env_info,
                            **bias_data,
                        )

                        self.record.stop_video()

                    # Reset environment
                    state = self.env_eval.reset()
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

                    self.agent.episode_done()
