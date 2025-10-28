"""
EvaluationRunner class for sequential evaluation of multiple model checkpoints.

This module provides a clean interface for loading different model checkpoints
from a training run and evaluating each one, allowing for performance tracking
across training steps.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

from base_runner import BaseRunner
from natsort import natsorted


class EvaluationRunner(BaseRunner):
    """
    Handles sequential evaluation of multiple model checkpoints.

    This class loads different model checkpoints from a training run and evaluates
    each one, providing performance tracking across different training steps.
    """

    def __init__(
        self,
        train_seed: int,
        eval_seed: int,
        configurations: dict[str, Any],
        base_log_dir: str,
        former_base_path: str,
        num_eval_episodes: int | None = None,
        save_configurations: bool = False,
    ):
        """
        Initialize EvaluationRunner for sequential checkpoint evaluation.

        Args:
            train_seed: Random seed for this training run
            eval_seed: Random seed for this evaluation run
            configurations: Dictionary containing all parsed configurations
            base_log_dir: Base directory for logging evaluation results
            former_base_path: Base path to the trained model directory (contains seed subdirs)
            num_episodes: Number of episodes to run per checkpoint (if None, uses config default)
            save_configurations: Whether to save configurations to disk
        """
        # Initialize the base runner with evaluation-specific settings
        super().__init__(
            train_seed=train_seed,
            eval_seed=eval_seed,
            configurations=configurations,
            base_log_dir=base_log_dir,
            save_configurations=save_configurations,
            num_eval_episodes=num_eval_episodes,
        )

        # EvaluationRunner-specific attributes
        self.former_model_base_path = Path(former_base_path)
        self.former_model_seed_path = self.former_model_base_path / str(self.train_seed)

        # Update logging for evaluation context
        self.logger.info(
            f"[SEED {self.train_seed}] will run {self.number_eval_episodes} episodes per checkpoint on [SEED {self.eval_seed}]"
        )

    def _discover_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Discover all available model checkpoints for this seed.

        Returns:
            List of checkpoint info dictionaries with 'path' and 'step' keys
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

    def _load_checkpoint(self, checkpoint_info: Dict[str, Any]) -> bool:
        """
        Load a specific model checkpoint into the agent.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            True if loading succeeded, False otherwise
        """
        checkpoint_path = checkpoint_info["path"]
        step = checkpoint_info["step"]

        self.logger.info(f"[SEED {self.eval_seed}] (step {step})")

        try:
            self.agent.load_models(checkpoint_path, self.alg_config.algorithm)
            self.logger.info(
                f"[SEED {self.eval_seed}] Successfully loaded checkpoint: {step}"
            )
            return True
        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.logger.warning(
                f"[SEED {self.eval_seed}] Failed to load checkpoint {step}: {e}"
            )
            return False

    def _evaluate_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """
        Evaluate a single checkpoint.

        Args:
            checkpoint_info: Checkpoint information dictionary

        Returns:
            Dictionary with evaluation results
        """
        step = checkpoint_info["step"]

        self.logger.info(
            f"[SEED {self.eval_seed}] Evaluating checkpoint: (step {step})"
        )

        start_time = time.time()

        if self.agent.policy_type == "usd":
            results = self._evaluate_usd_skills(checkpoint_info["step"], f"{step}")
        else:
            results = self._evaluate_agent_episodes(checkpoint_info["step"], f"{step}")

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.logger.info(
            f"[SEED {self.eval_seed}] Completed evaluation of {step}: "
            f"Avg reward: {results.get('avg_reward', 'N/A'):.2f}, "
            f"Time: {evaluation_time:.1f}s"
        )

    def run_evaluation(self) -> None:
        """
        Execute evaluation of all discovered checkpoints.
        """
        self.logger.info(f"[SEED {self.eval_seed}] Starting checkpoint evaluation")

        # Discover all checkpoints
        checkpoints = self._discover_checkpoints()

        if not checkpoints:
            self.logger.warning(
                f"[SEED {self.eval_seed}] No checkpoints found to evaluate"
            )
            return

        for i, checkpoint_info in enumerate(checkpoints):
            self.logger.info(
                f"[SEED {self.eval_seed}] Processing checkpoint {i + 1}/{len(checkpoints)}"
            )

            # Load the checkpoint
            if not self._load_checkpoint(checkpoint_info):
                self.logger.error(
                    f"[SEED {self.eval_seed}] Failed to load checkpoint {checkpoint_info['step']}, skipping"
                )
                continue

            self._evaluate_checkpoint(checkpoint_info)

        # Save all results
        self.record.save()

        self.logger.info(f"[SEED {self.eval_seed}] evaluation completed.")

    def run_test(self) -> None:
        """
        Execute testing (alias for evaluation for now).

        This is kept separate in case future differentiation is needed
        between evaluation and testing protocols.
        """
        self.logger.info(
            f"[SEED {self.train_seed}] Starting testing with {self.number_eval_episodes} episodes per checkpoint on [SEED {self.eval_seed}]"
        )
