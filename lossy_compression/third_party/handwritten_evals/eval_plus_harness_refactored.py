# lifted more or less irectly from
# https://github.com/evalplus/evalplus/blob/master/evalplus/evaluate.py

import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.codegen import run_codegen
from evalplus.config import *
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    """Compute or load cached expected outputs for all problems.
    
    Args:
        problems: Dict mapping task_id to problem data
        hashcode: Unique identifier for caching results
        tasks_only_output_not_none: List of tasks that require non-None outputs
        
    Returns:
        Dict mapping task_id to {"base": (output, time), "plus": (output, time)}
    """
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        print(f"Load from ground-truth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Computing expected output...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"]
            in tasks_only_output_not_none,
        )

        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"]
            in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:  # {...}, "base" | "plus" -> (status, details)
    """Test a solution against expected outputs for base and plus test cases.
    
    Args:
        dataset: Dataset name ("humaneval" or "mbpp")
        completion_id: Unique ID for this completion
        problem: Problem data containing inputs and entry point
        solution: Generated code solution to test
        expected_output: Expected outputs from get_groundtruth()
        base_only: Only test base cases, skip plus cases
        fast_check: Use fast checking mode
        identifier: Optional identifier for tracking
        min_time_limit: Minimum execution time limit
        gt_time_limit_factor: Time limit multiplier vs ground truth
        
    Returns:
        Dict with completion metadata and test results for base/plus cases
    """
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def _determine_result_path(samples: str, output_file: Optional[str]) -> str:
    """Determine the path for saving evaluation results.
    
    Args:
        samples: Path to samples file or directory
        output_file: Optional explicit output file path
        
    Returns:
        Path where results should be saved
    """
    if output_file is not None:
        return output_file

    if os.path.isdir(samples):
        return os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        # legacy compatibility
        if os.path.exists(samples.replace(".jsonl", "_eval_results.json")):
            return samples.replace(".jsonl", "_eval_results.json")
        else:
            return samples.replace(".jsonl", ".eval_results.json")


def _load_existing_results(result_path: str) -> Optional[Dict]:
    """Load existing evaluation results if available.
    
    Args:
        result_path: Path to results file
        
    Returns:
        Loaded results dict or None if file doesn't exist
    """
    if os.path.isfile(result_path):
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)
        return compatible_eval_result(results)
    return None


def _load_dataset_and_ground_truth(dataset: str,
                                   mini: bool,
                                   noextreme: bool,
                                   version: str,
                                   max_problems: Optional[int] = None):
    """Load dataset problems and compute ground truth.
    
    Args:
        dataset: Dataset name ("humaneval" or "mbpp")
        mini: Use mini dataset
        noextreme: Exclude extreme cases
        version: Dataset version
        max_problems: Maximum number of problems to load (None for all)
        
    Returns:
        Tuple of (problems, expected_output, dataset_hash)
    """
    if dataset == "humaneval":
        problems = get_human_eval_plus(mini=mini,
                                       noextreme=noextreme,
                                       version=version)
        dataset_hash = get_human_eval_plus_hash(mini=mini,
                                                noextreme=noextreme,
                                                version=version)
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini,
                                 noextreme=noextreme,
                                 version=version)
        dataset_hash = get_mbpp_plus_hash(mini=mini,
                                          noextreme=noextreme,
                                          version=version)
        expected_output = get_groundtruth(problems, dataset_hash,
                                          MBPP_OUTPUT_NOT_NONE_TASKS)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Limit problems if max_problems is specified
    if max_problems is not None:
        problem_items = list(problems.items())
        limited_problems = dict(problem_items[:max_problems])
        limited_expected_output = {
            k: expected_output[k]
            for k in limited_problems.keys()
        }

        print(
            f"Limited dataset to {len(limited_problems)} problems (requested: {max_problems})"
        )
        print(f"Problems: {list(limited_problems.keys())}")

        return limited_problems, limited_expected_output, dataset_hash

    return problems, expected_output, dataset_hash


def _prepare_evaluation_tasks(samples: str, problems: Dict,
                              expected_output: Dict, dataset: str,
                              base_only: bool, test_details: bool,
                              min_time_limit: float,
                              gt_time_limit_factor: float):
    """Prepare evaluation tasks for parallel processing.
    
    Args:
        samples: Path to samples file
        problems: Problem data
        expected_output: Expected outputs
        dataset: Dataset name
        base_only: Only test base cases
        test_details: Include detailed test results
        min_time_limit: Minimum time limit
        gt_time_limit_factor: Time limit factor
        
    Returns:
        Tuple of (futures, completion_id, n_samples, remainings)
    """
    n_workers = max(1, multiprocessing.cpu_count() // 2)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        remainings = set()

        print("Reading samples...")
        for sample in tqdm(load_solutions(samples)):
            task_id = sample["task_id"]
            if task_id not in problems:
                warn(
                    f"Task {task_id} is found in the samples but not found in the dataset"
                )
                continue

            solution = (sample["solution"] if "solution" in sample else
                        problems[task_id]["prompt"] + sample["completion"])
            remainings.add(sample["_identifier"])

            args = (
                dataset,
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                base_only,
                not test_details,  # fast_check
                sample["_identifier"],
                min_time_limit,
                gt_time_limit_factor,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        print(
            f"Prepared {n_samples} evaluation tasks for {len(problems)} problems"
        )

        assert n_samples == len(remainings), "Missing problems in unfinished"
        assert len(completion_id) == len(
            problems), "Missing problems in samples"

        return futures, completion_id, n_samples, remainings


def _monitor_evaluation_progress(remainings: set):
    """Monitor evaluation progress and warn if stuck.
    
    Args:
        remainings: Set of remaining task identifiers
    """

    def stucking_checker():
        while remainings:
            last_size = len(remainings)
            time.sleep(20)
            if last_size != len(remainings) or len(remainings) == 0:
                continue
            # Potential stucking
            warn("No samples had finished testing in the last 20s")
            warn(f"{len(remainings)} samples to be tested: {remainings}")

    threading.Thread(target=stucking_checker).start()


def _process_evaluation_results(futures, n_samples: int, remainings: set):
    """Process evaluation results from futures.
    
    Args:
        futures: List of futures from parallel evaluation
        n_samples: Number of samples being evaluated
        remainings: Set of remaining task identifiers
        
    Returns:
        Dict mapping task_id to list of results
    """
    eval_results = defaultdict(list)

    for future in tqdm(as_completed(futures), total=n_samples):
        result = future.result()
        remainings.remove(result["_identifier"])
        eval_results[result["task_id"]].append(result)

    return eval_results


def _format_evaluation_results(eval_results: Dict, problems: Dict,
                               dataset: str, base_only: bool,
                               test_details: bool):
    """Format evaluation results into final structure.
    
    Args:
        eval_results: Raw evaluation results
        problems: Problem data
        dataset: Dataset name
        base_only: Only test base cases
        test_details: Include detailed test results
        
    Returns:
        Formatted results dict
    """

    def get_failed_tests(stat, details, inputs) -> List[Any]:
        if stat == PASS or not details:
            return []
        if test_details:
            return [inputs[i] for i in range(len(details)) if not details[i]]
        # else => simply return the only and the last fail test
        return [inputs[len(details) - 1]]

    results = {"eval": {}}

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = []

        for res in task_results:
            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(base_stat, base_details,
                                               problems[task_id]["base_input"])

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            # with plus tests
            if not base_only:
                plus_stat, plus_details = res["plus"]
                plus_fail_tests = get_failed_tests(
                    plus_stat, plus_details, problems[task_id]["plus_input"])

            if dataset == "mbpp":
                base_fail_tests = mbpp_serialize_inputs(
                    task_id, base_fail_tests)
                plus_fail_tests = mbpp_serialize_inputs(
                    task_id, plus_fail_tests)

            results["eval"][task_id].append({
                "task_id": task_id,
                "solution": res["solution"],
                "base_status": base_stat,
                "plus_status": plus_stat,
                "base_fail_tests": base_fail_tests,
                "plus_fail_tests": plus_fail_tests,
            })

    return results


def _calculate_pass_at_k_metrics(results: Dict, base_only: bool, dataset: str):
    """Calculate pass@k metrics for evaluation results.
    
    Args:
        results: Evaluation results
        base_only: Only test base cases
        dataset: Dataset name
        
    Returns:
        Updated results dict with pass@k metrics
    """
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum([
                    res[i]["base_status"] == res[i]["plus_status"] == PASS
                    for i in range(len(res))
                ]))

    base_correct = np.array(base_correct)

    # Calculate base pass@k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100] if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    # Calculate plus pass@k if applicable
    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct),
                                            k).mean()
            for k in [1, 10, 100] if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k

    return results


def _save_results_with_backup(results: Dict, result_path: str,
                              i_just_wanna_run: bool):
    """Save results to file with optional backup of existing file.
    
    Args:
        results: Results to save
        result_path: Path to save results
        i_just_wanna_run: Whether to prompt for overwrite
    """
    if os.path.isfile(result_path) and i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(
                f"{result_path} already exists. Press [Y/N] to overwrite or exit..."
            )
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)


def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    output_file: Optional[str] = None,
    max_problems: Optional[int] = None,
    **model_kwargs,
):
    """Main evaluation function - orchestrates the entire evaluation pipeline.
    
    Args:
        dataset: Dataset name ("humaneval" or "mbpp")
        samples: Path to samples file or directory
        base_only: Only test base cases
        parallel: Number of parallel workers
        i_just_wanna_run: Whether to prompt for overwrite
        test_details: Include detailed test results
        min_time_limit: Minimum execution time limit
        gt_time_limit_factor: Time limit multiplier
        mini: Use mini dataset
        noextreme: Exclude extreme cases
        version: Dataset version
        output_file: Output file path
        max_problems: Maximum number of problems to evaluate (None for all)
        **model_kwargs: Additional model arguments
    """
    # Generate samples if model kwargs provided
    if model_kwargs:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        samples = run_codegen(dataset=dataset, **model_kwargs)

    assert samples is not None, "No samples provided"

    # Determine result path
    result_path = _determine_result_path(samples, output_file)

    # Try to load existing results
    results = _load_existing_results(result_path)

    if results is None or i_just_wanna_run:
        # Load dataset and ground truth
        problems, expected_output, dataset_hash = _load_dataset_and_ground_truth(
            dataset, mini, noextreme, version, max_problems)

        # what is the shape of problems
        print(type(problems))
        print(len(problems))
        raise
        # Initialize results structure
        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        # Prepare and run evaluation tasks
        futures, completion_id, n_samples, remainings = _prepare_evaluation_tasks(
            samples, problems, expected_output, dataset, base_only,
            test_details, min_time_limit, gt_time_limit_factor)

        # Monitor progress
        _monitor_evaluation_progress(remainings)

        # Process results
        eval_results = _process_evaluation_results(futures, n_samples,
                                                   remainings)

        # Format results
        results = _format_evaluation_results(eval_results, problems, dataset,
                                             base_only, test_details)

    # Calculate metrics
    results = _calculate_pass_at_k_metrics(results, base_only, dataset)

    # Save results
    _save_results_with_backup(results, result_path, i_just_wanna_run)


def main():
    from fire import Fire
    Fire(evaluate)


if __name__ == "__main__":
    main()
