
from dm_control import suite


def all_env():
    max_len = max(len(d) for d, _ in suite.BENCHMARKING)
    for domain, task in suite.BENCHMARKING:
        print(f"{domain:<{max_len}}  {task}")

