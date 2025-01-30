# Description: This file contains the code to evaluate the model on the tasks provided in the tasks.yaml file.
# The code is taken from the lm-evaluation-harness repository and modified to be used in the LLM assessment.

from loguru import logger as eval_logger
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(str(Path(__file__).parent), "lm-evaluation-harness"))
print(sys.path)

from lm_eval import evaluator, utils
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table, simple_parse_args_string
from lm_eval.models.huggingface import HFLM

def task_evaluate(model, tasks, batch_size=1, device="cuda:0", num_fewshot=0) -> None:

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = HFLM(model)
    
    task_manager = TaskManager("WARNING", None)

    if os.path.isdir(tasks):
        import glob

        task_names = []
        yaml_path = os.path.join(tasks, "*.yaml")
        for yaml_file in glob.glob(yaml_path):
            config = utils.load_yaml_config(yaml_file)
            task_names.append(config)
    else:
        task_list = tasks.split(",")
        task_names = task_manager.match_tasks(task_list)
        for task in [task for task in task_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n",
            )
            raise ValueError(
                f"Tasks not found: {missing}."
            )
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=None,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        task_manager=task_manager,
        verbosity="WARNING",
    )
    print(make_table(results))
    return results['results']