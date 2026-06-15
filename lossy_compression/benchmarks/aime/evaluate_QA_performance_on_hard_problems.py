#!/usr/bin/env python3
"""
Evaluate question-answering models on medium and hard problems.
This script:
1. Analyzes baseline results from haiku, sonnet, and opus
2. Identifies medium (haiku fails, others pass) and hard (only opus passes) problems
3. Runs QA experiments with different question-generating models
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import argparse
import time

# Import the analysis functions

from lossy_compression.analyze_human_eval_results import load_results_from_folder, categorize_problems


def run_qa_experiment_parallel(task_ids: List[int],
                               llm_model: str = "opus",
                               slm_model: str = "haiku",
                               question_model: str = "opus",
                               num_questions: int = 10,
                               log_dir: Path = None) -> subprocess.Popen:
    """
    Start QA experiment in background, logging to file.
    
    Returns:
        Popen process object
    """
    # Build the command
    cmd = [
        "python", "run_human_eval.py", "--model", "qa", "--task-ids",
        *[str(tid) for tid in task_ids], "--llm-model", llm_model,
        "--slm-model", slm_model, "--question-model", question_model, "-q",
        str(num_questions), "--verbose"
    ]

    # Add parallel execution flags if requested
    if parallel:
        cmd.append("--parallel")
        cmd.extend(["--max-workers", str(max_workers)])

    # Create log file with timestamp
    config_name = f"QA_LLM-{llm_model}_SLM-{slm_model}_Q-{question_model}"
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{config_name}_{timestamp}.log"
        log_handle = open(log_file, 'w')
        print(f"Starting {config_name} -> logging to {log_file}")
    else:
        log_handle = subprocess.DEVNULL
        print(f"Starting {config_name}")

    # Start process in background
    process = subprocess.Popen(cmd,
                               stdout=log_handle,
                               stderr=subprocess.STDOUT,
                               text=True)

    # Store metadata on the process object
    process.config_name = config_name
    process.log_file = log_file if log_dir else None
    process.start_time = time.time()
    process.cmd = ' '.join(cmd)

    return process


def run_qa_experiment(task_ids: List[int],
                      llm_model: str = "opus",
                      slm_model: str = "haiku",
                      question_model: str = "opus",
                      num_questions: int = 10,
                      verbose: bool = True,
                      parallel: bool = False,
                      max_workers: int = 4) -> Tuple[str, float]:
    """
    Run QA experiment on specific task IDs with given model configuration.
    
    Args:
        task_ids: List of task IDs to test (just the numbers, e.g., [1, 2, 3])
        llm_model: LLM model name (opus, sonnet, haiku)
        slm_model: SLM model name
        question_model: Question generation model name
        num_questions: Maximum number of Q&A iterations
        verbose: Whether to show output
        
    Returns:
        Tuple of (path to saved results, execution time in seconds)
    """
    # Convert task IDs to space-separated string
    task_ids_str = " ".join(str(tid) for tid in task_ids)

    # Build the command
    cmd = [
        "python", "run_human_eval.py", "--model", "qa", "--task-ids",
        *[str(tid) for tid in task_ids], "--llm-model", llm_model,
        "--slm-model", slm_model, "--question-model", question_model, "-q",
        str(num_questions), "--verbose"
    ]

    # Add parallel execution flags if requested
    if parallel:
        cmd.append("--parallel")
        cmd.extend(["--max-workers", str(max_workers)])

    print(f"\n{'='*60}")
    print(f"Running QA Experiment")
    print(f"{'='*60}")
    print(f"LLM: {llm_model}")
    print(f"SLM: {slm_model}")
    print(f"Question Model: {question_model}")
    print(f"Tasks: {len(task_ids)} problems")
    print(f"Max iterations: {num_questions}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    # Start timing
    start_time = time.time()

    # Run the experiment with real-time output
    try:
        # Use subprocess.Popen for real-time output
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        results_path = None
        task_count = 0
        task_start_time = None

        # Print output line by line as it comes
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')  # Print each line in real-time

                # Track task progress and timing
                if "Processing task" in line or "Task HumanEval/" in line:
                    if task_start_time:
                        # Print timing for previous task
                        task_duration = time.time() - task_start_time
                        print(f"  ⏱️  Task completed in {task_duration:.1f}s")
                    task_start_time = time.time()
                    task_count += 1

                # Still capture the results path
                if "Results saved to:" in line:
                    results_path = line.split("Results saved to:")[-1].strip()

        # Final task timing
        if task_start_time:
            task_duration = time.time() - task_start_time
            print(f"  ⏱️  Task completed in {task_duration:.1f}s")

        # Wait for process to complete
        process.wait()

        # Calculate total time
        total_time = time.time() - start_time

        if process.returncode == 0:
            if results_path:
                print(f"\n{'='*60}")
                print(f"✅ Experiment completed")
                print(f"  Total time: {total_time:.1f}s")
                if task_count > 0:
                    print(f"  Average per task: {total_time/task_count:.1f}s")
                print(f"  Results: {results_path}")
                print(f"{'='*60}")
            return results_path, total_time
        else:
            print(f"❌ Error: Process exited with code {process.returncode}")
            return None, total_time

    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return None, time.time() - start_time

    return None, 0


def analyze_qa_results(results_paths: Dict[str, str]) -> Dict:
    """
    Analyze and compare results from different QA configurations.
    
    Args:
        results_paths: Dict mapping configuration names to result paths
        
    Returns:
        Dict containing all results and analysis
    """
    print(f"\n{'='*60}")
    print("QA RESULTS ANALYSIS")
    print(f"{'='*60}")

    all_results = {}
    detailed_task_results = {}  # task_id -> {config_name -> result}

    for config_name, results_path in results_paths.items():
        if results_path and Path(results_path).exists():
            # Load summary
            summary_path = Path(results_path) / "summary.json"
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            # Load detailed results
            detailed_path = Path(results_path) / "detailed_results.json"
            with open(detailed_path, 'r') as f:
                detailed = json.load(f)

            all_results[config_name] = {
                'summary':
                summary,
                'detailed':
                detailed,
                'passed_tasks':
                [r['task_id'] for r in detailed['results'] if r['passed']],
                'failed_tasks':
                [r['task_id'] for r in detailed['results'] if not r['passed']]
            }

            # Collect per-task results for comparison
            for result in detailed['results']:
                task_id = result['task_id']
                if task_id not in detailed_task_results:
                    detailed_task_results[task_id] = {}
                detailed_task_results[task_id][config_name] = {
                    'passed': result['passed'],
                    'status': result.get('status', 'unknown'),
                    'qa_metrics': result.get('qa_metrics', {}),
                    'failed_tests': result.get('failed_tests', [])
                }

            print(f"\n{config_name}:")
            print(
                f"  Success rate: {summary['success_rate']:.1%} ({summary['passed_count']}/{summary['total_tasks']})"
            )
            # Extract model name properly
            full_model = summary['model_config']['question_model']
            if 'haiku' in full_model:
                q_model_display = 'haiku'
            elif 'sonnet' in full_model:
                q_model_display = 'sonnet'
            elif 'opus' in full_model:
                q_model_display = 'opus'
            else:
                q_model_display = full_model.split('-')[1]  # Fallback
            print(f"  Model config: Q={q_model_display}")

            # Calculate average bits for successful problems
            qa_results = [
                r for r in detailed['results']
                if 'qa_metrics' in r and r['qa_metrics'] and r['passed']
            ]
            if qa_results:
                avg_bits = sum(r['qa_metrics']['bits_of_information']
                               for r in qa_results) / len(qa_results)
                print(f"  Avg bits (successful): {avg_bits:.1f}")

    # Compare results
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}")

        # Find problems solved by each configuration
        config_names = list(all_results.keys())
        for i, config in enumerate(config_names):
            unique_solved = set(all_results[config]['passed_tasks'])
            for other_config in config_names:
                if other_config != config:
                    unique_solved -= set(
                        all_results[other_config]['passed_tasks'])

            if unique_solved:
                print(
                    f"\nUniquely solved by {config}: {len(unique_solved)} problems"
                )
                for task_id in list(unique_solved)[:5]:
                    print(f"  - {task_id}")

    return all_results, detailed_task_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA models on medium and hard problems")
    parser.add_argument("--haiku",
                        required=True,
                        help="Path to Haiku baseline results")
    parser.add_argument("--sonnet",
                        required=True,
                        help="Path to Sonnet baseline results")
    parser.add_argument("--opus",
                        required=True,
                        help="Path to Opus baseline results")
    parser.add_argument("-q",
                        "--num-questions",
                        type=int,
                        default=10,
                        help="Maximum Q&A iterations (default: 10)")
    parser.add_argument(
        "--question-models",
        nargs="+",
        default=["haiku", "sonnet", "opus"],
        help="Question models to test (default: haiku sonnet opus)")
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments, only analyze existing results")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Show detailed output during experiments")
    parser.add_argument("--output-dir",
                        default="qa_evaluation_results",
                        help="Directory to save analysis results")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel task execution within run_human_eval.py")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for task execution (default: 4)")

    args = parser.parse_args()

    # Step 1: Load baseline results
    print("Loading baseline results...")
    haiku_results = load_results_from_folder(args.haiku)
    sonnet_results = load_results_from_folder(args.sonnet)
    opus_results = load_results_from_folder(args.opus)

    # Step 2: Categorize problems
    categories = categorize_problems(haiku_results, sonnet_results,
                                     opus_results)

    # Step 3: Identify medium and hard problems
    medium_problems = categories['medium']  # Haiku fails, sonnet & opus pass
    hard_problems = categories['hard']  # Only opus passes
    target_problems = medium_problems + hard_problems

    # Extract just the task numbers
    target_task_nums = []
    for task_id in target_problems:
        # Extract number from "HumanEval/123" format
        task_num = int(task_id.split('/')[-1])
        target_task_nums.append(task_num)

    print(f"\n{'='*60}")
    print("TARGET PROBLEMS")
    print(f"{'='*60}")
    print(
        f"Medium problems (haiku fails, others pass): {len(medium_problems)}")
    print(f"Hard problems (only opus passes): {len(hard_problems)}")
    print(f"Total target problems: {len(target_problems)}")
    print(f"\nMedium problem IDs: {medium_problems[:10]}...")
    print(f"Hard problem IDs: {hard_problems[:10]}...")

    # Create output directory early (needed for parallel mode logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Run QA experiments with different question models
    results_paths = {}
    experiment_timings = {}
    overall_start_time = time.time()

    if not args.skip_experiments:
        # Run experiments sequentially, but each with parallel task execution
        for question_model in args.question_models:
            config_name = f"QA_LLM-opus_SLM-haiku_Q-{question_model}"
            print(f"\n\n{'='*80}")
            print(f"EXPERIMENT: {config_name}")
            if args.parallel:
                print(
                    f"Running with parallel task execution ({args.max_workers} workers)"
                )
            print(f"{'='*80}")

            results_path, execution_time = run_qa_experiment(
                task_ids=target_task_nums,
                llm_model="opus",
                slm_model="haiku",
                question_model=question_model,
                num_questions=args.num_questions,
                verbose=args.verbose,
                parallel=args.parallel,
                max_workers=args.max_workers)

            if results_path:
                results_paths[config_name] = results_path
                experiment_timings[config_name] = execution_time

            # Small delay between experiments
            time.sleep(2)

        # Print overall timing summary
        total_experiment_time = time.time() - overall_start_time
        print(f"\n{'='*80}")
        print(f"TIMING SUMMARY")
        print(f"{'='*80}")
        for config_name, timing in experiment_timings.items():
            q_model = config_name.split('_Q-')[-1]
            print(f"  {q_model}: {timing:.1f}s ({timing/60:.1f} min)")
        print(f"{'='*80}")
        print(
            f"Total experiment time: {total_experiment_time:.1f}s ({total_experiment_time/60:.1f} min)"
        )
        if len(experiment_timings) > 0:
            avg_time = sum(
                experiment_timings.values()) / len(experiment_timings)
            print(
                f"Average per configuration: {avg_time:.1f}s ({avg_time/60:.1f} min)"
            )
        print(f"{'='*80}")

    # Step 5: Analyze results
    all_experiment_results = {}
    detailed_task_analysis = {}
    if results_paths:
        all_experiment_results, detailed_task_analysis = analyze_qa_results(
            results_paths)

    # Step 6: Save comprehensive analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = output_dir / f"qa_analysis_{timestamp}.json"
    detailed_file = output_dir / f"qa_detailed_{timestamp}.json"
    task_comparison_file = output_dir / f"qa_task_comparison_{timestamp}.json"

    # Main analysis summary
    analysis_data = {
        "timestamp": timestamp,
        "baseline_results": {
            "haiku": args.haiku,
            "sonnet": args.sonnet,
            "opus": args.opus
        },
        "problem_categories": {
            "medium": medium_problems,
            "hard": hard_problems,
            "medium_count": len(medium_problems),
            "hard_count": len(hard_problems)
        },
        "qa_experiments": results_paths,
        "configuration": {
            "num_questions": args.num_questions,
            "question_models_tested": args.question_models,
            "parallel": args.parallel,
            "max_workers": args.max_workers if args.parallel else None
        },
        "timings": experiment_timings if not args.skip_experiments else {},
        "summary_by_model": {}
    }

    # Add summary statistics for each model
    for config_name, results in all_experiment_results.items():
        if 'summary' in results:
            # Extract model name properly
            full_model = results['summary']['model_config']['question_model']
            if 'haiku' in full_model:
                q_model = 'haiku'
            elif 'sonnet' in full_model:
                q_model = 'sonnet'
            elif 'opus' in full_model:
                q_model = 'opus'
            else:
                q_model = full_model.split('-')[1]  # Fallback
            analysis_data['summary_by_model'][q_model] = {
                'success_rate': results['summary']['success_rate'],
                'passed_count': results['summary']['passed_count'],
                'total_tasks': results['summary']['total_tasks'],
                'passed_tasks': results.get('passed_tasks', []),
                'failed_tasks': results.get('failed_tasks', [])
            }

    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    # Save detailed experiment results (includes all QA metrics, solutions, etc.)
    if all_experiment_results:
        with open(detailed_file, 'w') as f:
            json.dump(all_experiment_results, f, indent=2)

    # Save task-by-task comparison across all models
    if detailed_task_analysis:
        # Add summary statistics per task
        task_summary = {}
        for task_id, model_results in detailed_task_analysis.items():
            models_passed = [
                model for model, result in model_results.items()
                if result['passed']
            ]
            task_summary[task_id] = {
                'models_attempted':
                list(model_results.keys()),
                'models_passed':
                models_passed,
                'success_rate':
                len(models_passed) /
                len(model_results) if model_results else 0,
                'details':
                model_results
            }

        with open(task_comparison_file, 'w') as f:
            json.dump(task_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Files saved:")
    print(f"  1. Summary: {analysis_file}")
    if all_experiment_results:
        print(f"  2. Detailed results: {detailed_file}")
    if detailed_task_analysis:
        print(f"  3. Task comparison: {task_comparison_file}")
    print(f"\nTarget problems tested: {len(target_problems)}")
    print(f"Configurations evaluated: {len(results_paths)}")

    # Print which tasks each model got right/wrong
    if detailed_task_analysis:
        print(f"\n{'='*60}")
        print("TASK-BY-TASK RESULTS")
        print(f"{'='*60}")
        for task_id in sorted(detailed_task_analysis.keys(),
                              key=lambda x: int(x.split('/')[-1])):
            models = detailed_task_analysis[task_id]
            passed = [
                m.split('_Q-')[-1] for m, r in models.items() if r['passed']
            ]
            failed = [
                m.split('_Q-')[-1] for m, r in models.items()
                if not r['passed']
            ]

            status = "✅" if passed else "❌"
            print(
                f"{task_id:15} {status} Passed: {passed if passed else 'none'} | Failed: {failed if failed else 'none'}"
            )

    # Print final summary table
    if results_paths:
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"{'Question Model':<15} {'Success Rate':<15} {'Passed':<10}")
        print(f"{'-'*40}")

        for config_name, path in results_paths.items():
            if path and Path(path).exists():
                with open(Path(path) / "summary.json", 'r') as f:
                    summary = json.load(f)
                # Extract model name properly
                full_model = summary['model_config']['question_model']
                if 'haiku' in full_model:
                    q_model = 'haiku'
                elif 'sonnet' in full_model:
                    q_model = 'sonnet'
                elif 'opus' in full_model:
                    q_model = 'opus'
                else:
                    q_model = full_model.split('-')[1]  # Fallback
                success_rate = f"{summary['success_rate']:.1%}"
                passed = f"{summary['passed_count']}/{summary['total_tasks']}"
                print(f"{q_model:<15} {success_rate:<15} {passed:<10}")


if __name__ == "__main__":
    # Make sure we're in the right directory
    if not Path("run_human_eval.py").exists():
        print(
            "Error: This script must be run from the lossy_compression directory"
        )
        sys.exit(1)

    main()
