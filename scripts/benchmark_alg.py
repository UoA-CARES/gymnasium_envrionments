import sys
from run import main as run_main
from io import StringIO
from typing import List

# The equivalent of bash's domain-task map
dmcs_tasks = {
    "acrobot": "swingup",
    "ball_in_cup": "catch",
    "cartpole": "swingup",
    "cheetah": "run",
    "finger": "turn_hard",
    "hopper": "hop",
    "humanoid": "run",
    "reacher": "hard",
    "walker": "walk",
}


def run_with_args_and_stdin(args_list: List[str], stdin_str: str = ""):
    """Simulate CLI args and stdin, then call run.py's main()."""
    old_argv = sys.argv.copy()
    old_stdin = sys.stdin
    try:
        sys.argv = ["run.py"] + args_list
        sys.stdin = StringIO(stdin_str)
        run_main()
    finally:
        sys.argv = old_argv
        sys.stdin = old_stdin


def main():
    # CLI argument parsing (simple mimic of the bash version)
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <required_arg> <run_name> [optional_args...]")
        sys.exit(1)

    required_arg = sys.argv[1]
    input_name = sys.argv[2] if len(sys.argv) >= 3 else ""
    optional_args = sys.argv[3:] if len(sys.argv) > 3 else []

    for domain, task in dmcs_tasks.items():
        print(f"Running: domain={domain}, task={task}")
        args = [
            "train",
            "cli",
            "--gym",
            "dmcs",
            "--domain",
            domain,
            "--task",
            task,
            required_arg,
            "--seeds",
            "10",
            "20",
            "30",
            "40",
            "50",
            *optional_args,
        ]
        run_with_args_and_stdin(args, stdin_str=input_name)


if __name__ == "__main__":
    main()
