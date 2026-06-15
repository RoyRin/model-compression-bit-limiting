#!/usr/bin/env python3
"""
Experiment runner for LLM-SLM compression across multiple questions.
Runs 30 iterations per question and tracks all metrics.
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lossy_compression.core.qa_compression import iterative_SLM_loop

from lossy_compression import LLM, SLM, QUESTION_SLM


class ExperimentRunner:

    def __init__(self,
                 experiment_name: str = None,
                 max_iterations: int = 30,
                 quality_threshold: int = 9,
                 llm_model: str = LLM,
                 slm_model: str = SLM,
                 question_model: str = QUESTION_SLM,
                 use_local_slm: bool = False,
                 open_ended_guidance: bool = False,
                 use_batch: bool = False,
                 enable_parallel: bool = False,
                 verbose: bool = False,
                 output_dir: str = "experiments"):
        """Initialize experiment runner."""
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.llm_model = llm_model
        self.slm_model = slm_model
        self.question_model = question_model
        self.use_local_slm = use_local_slm
        self.open_ended_guidance = open_ended_guidance
        self.use_batch = use_batch
        self.enable_parallel = enable_parallel
        self.verbose = verbose

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = experiment_name or f"run_{timestamp}"
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize results storage
        self.results = []
        self.config = {
            "experiment_name": self.experiment_name,
            "timestamp": timestamp,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "llm_model": llm_model,
            "slm_model": slm_model,
            "question_model": question_model,
            "use_local_slm": use_local_slm,
            "open_ended_guidance": open_ended_guidance,
            "use_batch": use_batch,
            "enable_parallel": enable_parallel
        }

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def run_single_question(self, question_id: int, question_text: str,
                            category: str) -> Dict[str, Any]:
        """Run LLM-SLM loop for a single question."""
        print(f"\n{'='*60}")
        print(f"Question {question_id}: {question_text[:50]}...")
        print(f"Category: {category}")
        print(f"{'='*60}")

        start_time = time.time()

        try:

            # Run the iterative loop
            final_answer, qa_pairs, metrics = iterative_SLM_loop(
                question_text,
                large_model_name=self.llm_model,
                small_model_name=self.slm_model,
                question_model_name=self.question_model,
                use_local_slm=self.use_local_slm,
                max_iterations=self.max_iterations,
                quality_threshold=self.quality_threshold,
                open_ended_guidance=self.open_ended_guidance,
                enable_parallel=self.enable_parallel,
                verbose=self.verbose)

            elapsed_time = time.time() - start_time

            # Extract detailed information
            result = {
                "question_id": question_id,
                "question": question_text,
                "category": category,
                "final_answer": final_answer,
                "qa_pairs": qa_pairs,
                "metrics": metrics,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "error": None
            }

            # Extract key metrics for summary
            if metrics:
                result["summary"] = {
                    "iterations_completed":
                    metrics.get("iterations", 0),
                    "final_quality_score":
                    metrics.get("final_quality_score", 0),
                    "best_quality_score":
                    metrics.get("best_quality_score", 0),
                    "initial_score":
                    metrics.get("quality_scores", [0])[0]
                    if metrics.get("quality_scores") else 0,
                    "improvement":
                    metrics.get("best_quality_score", 0) -
                    (metrics.get("quality_scores", [0])[0]
                     if metrics.get("quality_scores") else 0),
                    "total_qa_pairs":
                    metrics.get("total_qa_pairs", 0),
                    "quality_progression":
                    metrics.get("quality_progression", []),
                    "reached_threshold":
                    metrics.get("final_quality_score",
                                0) >= self.quality_threshold
                }

            print(f"✅ Completed in {elapsed_time:.2f}s")
            print(
                f"📊 Final Score: {result['summary']['final_quality_score']}/10"
            )
            print(f"📈 Improvement: +{result['summary']['improvement']}")

        except Exception as e:
            import traceback
            print(f"❌ Error: {str(e)}")
            print(f"📋 Full traceback:")
            traceback.print_exc()
            result = {
                "question_id": question_id,
                "question": question_text,
                "category": category,
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

        # Save individual result
        result_file = self.logs_dir / f"question_{question_id:03d}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def run_experiment(self,
                       questions_file: str = "experiment_questions.json",
                       question_ids: List[int] = None):
        """Run experiment on multiple questions."""
        # Load questions
        with open(questions_file, "r") as f:
            data = json.load(f)
            questions = data["questions"]

        # Filter questions if specific IDs provided
        if question_ids:
            questions = [q for q in questions if q["id"] in question_ids]

        print(f"\n🚀 Starting experiment: {self.experiment_name}")
        print(f"📝 Running {len(questions)} questions")
        print(f"🔄 Max iterations per question: {self.max_iterations}")
        print(f"🎯 Quality threshold: {self.quality_threshold}/10")

        print(f"🤖 LLM model: {self.llm_model}")
        print(f"🤖 SLM model: {self.slm_model}")
        print(f"🤖 Question model: {self.question_model}")

        # Save questions being used
        with open(self.output_dir / "questions.json", "w") as f:
            json.dump({"questions": questions}, f, indent=2)

        # Run each question
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question {q['id']}...")
            result = self.run_single_question(q["id"], q["question"],
                                              q["category"])
            self.results.append(result)

            # Save intermediate summary
            self._save_summary()

        # Final summary
        self._generate_final_report()

        print("\n✅ Experiment completed!")
        print(f"📁 Results saved to: {self.output_dir}")

    def _save_summary(self):
        """Save summary of results so far."""
        summary = {
            "config": self.config,
            "total_questions": len(self.results),
            "successful":
            sum(1 for r in self.results if r.get("success", False)),
            "failed":
            sum(1 for r in self.results if not r.get("success", False)),
            "results": []
        }

        for r in self.results:
            if r.get("success") and "summary" in r:
                summary["results"].append({
                    "question_id":
                    r["question_id"],
                    "category":
                    r["category"],
                    "initial_score":
                    r["summary"]["initial_score"],
                    "final_score":
                    r["summary"]["final_quality_score"],
                    "improvement":
                    r["summary"]["improvement"],
                    "iterations":
                    r["summary"]["iterations_completed"],
                    "reached_threshold":
                    r["summary"]["reached_threshold"],
                    "elapsed_time":
                    r["elapsed_time"]
                })

        # Calculate aggregate statistics
        if summary["results"]:
            scores = [r["final_score"] for r in summary["results"]]
            improvements = [r["improvement"] for r in summary["results"]]

            summary["statistics"] = {
                "avg_final_score":
                sum(scores) / len(scores),
                "avg_improvement":
                sum(improvements) / len(improvements),
                "max_score":
                max(scores),
                "min_score":
                min(scores),
                "threshold_reached":
                sum(1 for r in summary["results"] if r["reached_threshold"]),
                "avg_time_per_question":
                sum(r["elapsed_time"]
                    for r in summary["results"]) / len(summary["results"])
            }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _generate_final_report(self):
        """Generate HTML report of results."""
        with open(self.output_dir / "summary.json", "r") as f:
            summary = json.load(f)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .improvement {{ color: blue; font-weight: bold; }}
                .statistics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>LLM-SLM Compression Experiment Report</h1>
            <h2>Configuration</h2>
            <ul>
                <li>Experiment: {self.experiment_name}</li>
                <li>Timestamp: {self.config['timestamp']}</li>
                <li>Max Iterations: {self.config['max_iterations']}</li>
                <li>Quality Threshold: {self.config['quality_threshold']}/10</li>
                <li>LLM Model: {self.config['llm_model']}</li>
                <li>SLM Model: {self.config['slm_model']}</li>
            </ul>
            
            <h2>Summary Statistics</h2>
            <div class="statistics">
                <p>Total Questions: {summary['total_questions']}</p>
                <p>Successful: <span class="success">{summary['successful']}</span></p>
                <p>Failed: <span class="failure">{summary['failed']}</span></p>
        """

        if "statistics" in summary:
            stats = summary["statistics"]
            html += f"""
                <p>Average Final Score: {stats['avg_final_score']:.2f}/10</p>
                <p>Average Improvement: <span class="improvement">+{stats['avg_improvement']:.2f}</span></p>
                <p>Questions Reaching Threshold: {stats['threshold_reached']}/{summary['successful']}</p>
                <p>Average Time per Question: {stats['avg_time_per_question']:.2f}s</p>
            """

        html += """
            </div>
            
            <h2>Individual Results</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Initial Score</th>
                    <th>Final Score</th>
                    <th>Improvement</th>
                    <th>Iterations</th>
                    <th>Threshold</th>
                    <th>Time (s)</th>
                </tr>
        """

        for r in summary.get("results", []):
            threshold_class = "success" if r["reached_threshold"] else ""
            html += f"""
                <tr>
                    <td>{r['question_id']}</td>
                    <td>{r['category']}</td>
                    <td>{r['initial_score']}/10</td>
                    <td>{r['final_score']}/10</td>
                    <td class="improvement">+{r['improvement']}</td>
                    <td>{r['iterations']}</td>
                    <td class="{threshold_class}">{'✓' if r['reached_threshold'] else '✗'}</td>
                    <td>{r['elapsed_time']:.2f}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        with open(self.output_dir / "report.html", "w") as f:
            f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-SLM compression experiments")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--iterations",
                        type=int,
                        default=20,
                        help="Max iterations per question")
    parser.add_argument("--threshold",
                        type=int,
                        default=9,
                        help="Quality threshold (1-10)")
    parser.add_argument("--llm", type=str, default=LLM, help="LLM model")
    parser.add_argument("--slm", type=str, default=SLM, help="SLM model")
    parser.add_argument("--question-model",
                        type=str,
                        default=QUESTION_SLM,
                        help="Question generation model")
    parser.add_argument("--local", action="store_true", help="Use local SLM")
    parser.add_argument("--open-ended",
                        action="store_true",
                        help="Enable open-ended guidance")
    parser.add_argument("--batch",
                        action="store_true",
                        help="Use batch question generation mode")
    parser.add_argument("--parallel",
                        action="store_true",
                        help="Enable parallel execution")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--questions",
                        type=str,
                        default="experiment_questions.json",
                        help="Questions file")
    parser.add_argument("--ids",
                        type=int,
                        nargs="+",
                        help="Specific question IDs to run")
    parser.add_argument("--output",
                        type=str,
                        default="experiments",
                        help="Output directory")

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(experiment_name=args.name,
                              max_iterations=args.iterations,
                              quality_threshold=args.threshold,
                              llm_model=args.llm,
                              slm_model=args.slm,
                              question_model=args.question_model,
                              use_local_slm=args.local,
                              open_ended_guidance=args.open_ended,
                              use_batch=args.batch,
                              enable_parallel=args.parallel,
                              verbose=args.verbose,
                              output_dir=args.output)

    # Run experiment
    runner.run_experiment(questions_file=args.questions, question_ids=args.ids)


if __name__ == "__main__":
    main()
