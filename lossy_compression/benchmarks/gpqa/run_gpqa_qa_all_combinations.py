#!/usr/bin/env python3
"""
Run all 27 combinations of SLM, LLM, and Question-model for GPQA Q&A compression.
Maximum 3 parallel workers at a time to avoid API rate limits.

Usage (run from lossy_compression directory):
    python run_gpqa_qa_all_combinations.py --difficulty medium+hard --max-questions 10
    python run_gpqa_qa_all_combinations.py --difficulty medium+hard --max-questions 10 --num-problems 5  # test mode
    python run_gpqa_qa_all_combinations.py --difficulty medium+hard --no-timestamp --skip-existing  # resumable
"""

import subprocess
import time
from multiprocessing import Pool, current_process
import argparse
from datetime import datetime
import os
import json


def run_combination(args):
    """Run a single SLM/LLM/Question-model combination."""
    slm, llm, question_model, difficulty, max_questions, num_problems, batch_mode, batch_size, verbose, use_timestamp = args

    # Create output filename - optionally without timestamp for consistent naming
    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"lossy_compression/results/gpqa_qa_{slm}_{llm}_{question_model}_{difficulty}_{timestamp}.json"
    else:
        output_path = f"lossy_compression/results/gpqa_qa_{slm}_{llm}_{question_model}_{difficulty}.json"

    # Build command
    cmd = [
        "python", "evaluate_gpqa_qa_compression.py", "--difficulty",
        difficulty, "--slm", slm, "--llm", llm, "--question-model",
        question_model, "--max-questions",
        str(max_questions), "--output", output_path
    ]

    if num_problems:
        cmd.extend(["--num-problems", str(num_problems)])

    if batch_mode:
        cmd.append("--batch")
        cmd.extend(["--batch-size", str(batch_size)])

    # Always run subprocesses with verbose to see detailed progress
    cmd.append("--verbose")

    # Log start
    worker_id = current_process().name
    print(
        f"\n[{worker_id}] Starting: SLM={slm}, LLM={llm}, Q={question_model}")
    print(f"[{worker_id}] Command: {' '.join(cmd)}")
    start_time = time.time()

    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check if successful
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(
                f"\n[{worker_id}] ✓ Completed: SLM={slm}, LLM={llm}, Q={question_model} ({elapsed:.1f}s)"
            )
            print(f"[{worker_id}] Output saved to: {output_path}")

            # Try to extract accuracy from output
            if "Accuracy:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Accuracy:" in line:
                        print(f"[{worker_id}] {line.strip()}")
                        break

            # Load the results JSON to get accuracy and problem details
            try:
                with open(output_path, 'r') as f:
                    result_data = json.load(f)
                    accuracy = result_data.get('accuracy', 0)
                    correct_count = result_data.get('correct_count', 0)
                    total_problems = result_data.get('total_problems', 0)
                    avg_questions = result_data.get('avg_questions', 0)
                    problem_results = result_data.get('results', [])

                    # Calculate min/max questions if available
                    if problem_results:
                        questions_per_problem = [
                            r.get('num_questions', 0) for r in problem_results
                        ]
                        min_questions = min(questions_per_problem
                                            ) if questions_per_problem else 0
                        max_questions = max(questions_per_problem
                                            ) if questions_per_problem else 0
                    else:
                        min_questions = 0
                        max_questions = 0

                    # Extract quality scores/ratings from metrics if available
                    all_quality_scores = []
                    for r in problem_results:
                        if r.get('metrics') and r['metrics'].get(
                                'quality_scores'):
                            all_quality_scores.extend(
                                r['metrics']['quality_scores'])

                    if all_quality_scores:
                        avg_quality_score = sum(all_quality_scores) / len(
                            all_quality_scores)
                        best_quality_scores = [
                            r['metrics'].get('best_quality_score', 0)
                            for r in problem_results if r.get('metrics')
                            and 'best_quality_score' in r['metrics']
                        ]
                        avg_best_quality = sum(best_quality_scores) / len(
                            best_quality_scores
                        ) if best_quality_scores else None
                    else:
                        avg_quality_score = None
                        avg_best_quality = None

            except:
                accuracy = None
                correct_count = None
                total_problems = None
                avg_questions = None
                min_questions = None
                max_questions = None
                avg_quality_score = None
                avg_best_quality = None
                problem_results = []

            return {
                'status': 'success',
                'slm': slm,
                'llm': llm,
                'question_model': question_model,
                'output_path': output_path,
                'elapsed_time': elapsed,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_problems': total_problems,
                'avg_questions': avg_questions,
                'min_questions': min_questions,
                'max_questions': max_questions,
                'avg_quality_score': avg_quality_score,
                'avg_best_quality': avg_best_quality,
                'problem_results': problem_results
            }
        else:
            print(
                f"\n[{worker_id}] ✗ Failed: SLM={slm}, LLM={llm}, Q={question_model}"
            )
            print(f"[{worker_id}] Error: {result.stderr[:500]}")
            return {
                'status': 'failed',
                'slm': slm,
                'llm': llm,
                'question_model': question_model,
                'error': result.stderr[:500]
            }

    except Exception as e:
        print(
            f"\n[{worker_id}] ✗ Exception: SLM={slm}, LLM={llm}, Q={question_model}"
        )
        print(f"[{worker_id}] Error: {str(e)}")
        return {
            'status': 'error',
            'slm': slm,
            'llm': llm,
            'question_model': question_model,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run all GPQA Q&A compression combinations')
    parser.add_argument(
        '--difficulty',
        type=str,
        default='medium+hard',
        choices=['easy', 'medium', 'hard', 'all', 'medium+hard'],
        help='Problem difficulty (default: medium+hard)')
    parser.add_argument('--max-questions',
                        type=int,
                        default=10,
                        help='Max questions per problem (default: 10)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Limit number of problems (default: all)')
    parser.add_argument('--batch',
                        action='store_true',
                        help='Enable batch mode')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='Batch size if using --batch (default: 10)')
    parser.add_argument('--max-workers',
                        type=int,
                        default=3,
                        help='Maximum parallel workers (default: 3)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose output for each run')
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Which models to use: all, or comma-separated (e.g., haiku,sonnet)'
    )
    parser.add_argument('--skip-existing',
                        action='store_true',
                        help='Skip combinations that already have results')
    parser.add_argument(
        '--summary-output',
        type=str,
        default=None,
        help='Path for summary JSON file (default: auto-generated)')
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help=
        'Do not use timestamps in output filenames (allows consistent naming for resume)'
    )

    args = parser.parse_args()

    # Define models
    if args.models == 'all':
        models = ['haiku', 'sonnet', 'opus']
    else:
        models = args.models.split(',')

    # Generate all 27 combinations (or subset)
    combinations = []
    for slm in models:
        for llm in models:
            for question_model in models:
                # Check if we should skip existing results
                if args.skip_existing:
                    # Look for existing results - check both with and without timestamp
                    if args.no_timestamp:
                        # Check for exact match without timestamp
                        exact_path = f"lossy_compression/results/gpqa_qa_{slm}_{llm}_{question_model}_{args.difficulty}.json"
                        if os.path.exists(exact_path):
                            print(
                                f"Skipping existing: SLM={slm}, LLM={llm}, Q={question_model} ({exact_path})"
                            )
                            continue
                    else:
                        # Check for any timestamped version
                        pattern = f"lossy_compression/results/gpqa_qa_{slm}_{llm}_{question_model}_{args.difficulty}_*.json"
                        import glob
                        existing = glob.glob(pattern)
                        if existing:
                            print(
                                f"Skipping existing: SLM={slm}, LLM={llm}, Q={question_model} ({existing[0]})"
                            )
                            continue

                combinations.append(
                    (slm, llm, question_model, args.difficulty,
                     args.max_questions, args.num_problems, args.batch,
                     args.batch_size, args.verbose, not args.no_timestamp))

    total_combinations = len(combinations)
    print(f"\n{'='*60}")
    print(
        f"Running {total_combinations} combinations with max {args.max_workers} parallel workers"
    )
    print(f"Difficulty: {args.difficulty}")
    print(f"Max questions: {args.max_questions}")
    print(f"Num problems: {args.num_problems if args.num_problems else 'all'}")
    print(f"Batch mode: {args.batch}")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*60}\n")

    # Create results directory if needed
    os.makedirs('results', exist_ok=True)

    # If resuming, load existing results first
    existing_results = []
    if args.skip_existing and args.no_timestamp:
        # Load any existing completed results
        for slm in models:
            for llm in models:
                for question_model in models:
                    exact_path = f"lossy_compression/results/gpqa_qa_{slm}_{llm}_{question_model}_{args.difficulty}.json"
                    if os.path.exists(exact_path):
                        try:
                            with open(exact_path, 'r') as f:
                                result_data = json.load(f)
                                existing_results.append({
                                    'status':
                                    'existing',
                                    'slm':
                                    slm,
                                    'llm':
                                    llm,
                                    'question_model':
                                    question_model,
                                    'output_path':
                                    exact_path,
                                    'elapsed_time':
                                    0,
                                    'accuracy':
                                    result_data.get('accuracy', 0),
                                    'correct_count':
                                    result_data.get('correct_count', 0),
                                    'total_problems':
                                    result_data.get('total_problems', 0),
                                    'avg_questions':
                                    result_data.get('avg_questions', 0),
                                    'problem_results':
                                    result_data.get('results', [])
                                })
                                print(
                                    f"Loaded existing result: SLM={slm}, LLM={llm}, Q={question_model}"
                                )
                        except Exception as e:
                            print(f"Warning: Could not load {exact_path}: {e}")

    # Run with multiprocessing pool
    start_time = time.time()
    if combinations:
        with Pool(args.max_workers) as pool:
            new_results = pool.map(run_combination, combinations)
    else:
        new_results = []

    # Combine existing and new results
    results = existing_results + new_results

    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results
                     if r['status'] in ['success', 'existing'])
    failed = sum(1 for r in results
                 if r['status'] not in ['success', 'existing'])
    existing_count = sum(1 for r in results if r['status'] == 'existing')

    print(f"\n{'='*60}")
    print(f"COMPLETED ALL COMBINATIONS")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {successful}/{total_combinations}")
    if existing_count > 0:
        print(f"  - From existing results: {existing_count}")
        print(f"  - Newly computed: {successful - existing_count}")
    print(f"Failed: {failed}/{total_combinations}")

    # Save summary
    if args.summary_output:
        summary_path = args.summary_output
        # Check if file already exists
        if os.path.exists(summary_path):
            print(f"\nWarning: Summary file already exists: {summary_path}")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Not saving summary.")
                summary_path = None
    else:
        summary_path = f"lossy_compression/results/gpqa_qa_parallel_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if summary_path:
        with open(summary_path, 'w') as f:
            json.dump(
                {
                    'timestamp': datetime.now().isoformat(),
                    'args': vars(args),
                    'total_combinations': total_combinations,
                    'successful': successful,
                    'failed': failed,
                    'total_time_seconds': total_time,
                    'results': results
                },
                f,
                indent=2)
        print(f"\nSummary saved to: {summary_path}")

    # Print failed combinations for retry
    if failed > 0:
        print("\nFailed combinations (for retry):")
        for r in results:
            if r['status'] != 'success':
                print(
                    f"  - SLM={r['slm']}, LLM={r['llm']}, Q={r['question_model']}"
                )
                print(f"    Error: {r.get('error', 'Unknown')[:100]}")


if __name__ == "__main__":
    main()
