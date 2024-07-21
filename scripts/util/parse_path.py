from datetime import datetime

def create_path_from_format_string(format_str: str, algorithm: str, domain: str, task: str, gym: str, seed: int, run_name: str) -> str:
    """
    Create a path from a format string
    :param format_str: The format string to use
    :param domain: The domain of the environment
    :param task: The task of the environment
    :param gym: The gym environment
    :param seed: The seed used
    :param run_name: The name of the run
    :return: The path
    """
    
    domain_with_hyphen_or_empty = f"{domain}-" if domain != "" else ""
    domain_task = domain_with_hyphen_or_empty + task
    
    date = datetime.now().strftime('%y_%m_%d_%H-%M-%S')

    run_name_else_date = run_name if run_name != "" else date
    
    log_dir = format_str.format(algorithm=algorithm, domain=domain, task=task, gym=gym, run_name=run_name, run_name_else_date=run_name_else_date, seed=seed, domain_task=domain_task, date=date)
    return log_dir