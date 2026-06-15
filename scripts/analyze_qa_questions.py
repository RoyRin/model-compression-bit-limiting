"""
Analyze the types of questions asked in the QA compression protocol.

Compares questions in cases where QA helped the SLM improve ("recovered")
vs. cases where it did not ("not recovered"), looking for patterns in
question type, specificity, and answer distribution.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path("results/iterative-qa-sweep/v4.5")

# Only look at QA files (not BLC), main objective runs (not ablations)
QA_PATTERN = re.compile(r"^(.+)_QA_objective_v4\.5_\d+_\d+\.json$")


def load_all_qa_results():
    """Load all QA objective result files."""
    results = {}
    for f in sorted(RESULTS_DIR.iterdir()):
        m = QA_PATTERN.match(f.name)
        if m:
            dataset = m.group(1)
            with open(f) as fh:
                data = json.load(fh)
            results[dataset] = data
    return results


def classify_problem(p):
    """Classify a problem into recovered / not_recovered / regressed / already_correct."""
    if p["initial_correct"]:
        if p["final_correct"]:
            return "already_correct"
        else:
            return "regressed"
    else:
        if p["final_correct"]:
            return "recovered"
        else:
            return "not_recovered"


def classify_question(q):
    """Heuristic classification of question types."""
    q_lower = q.lower().strip()

    # Arithmetic / computation verification
    arith_keywords = [
        "arithmetic",
        "calculation",
        "computed",
        "compute",
        "sum ",
        "product",
        "divide",
        "multiply",
        "subtract",
        "add ",
        "equals",
        "equal to",
        "yield",
        "result in",
        "evaluates to",
        "simplif",
        "correctly calculated",
        "correctly computed",
    ]
    if any(kw in q_lower for kw in arith_keywords):
        return "arithmetic_check"

    # Interpretation / understanding of problem statement
    interp_keywords = [
        "interpret",
        "understood",
        "understanding",
        "mean ",
        "means ",
        "meaning",
        "phrase",
        "refer to",
        "refers to",
        "definition",
        "defined as",
        "correctly read",
        "problem ask",
        "problem state",
        "question ask",
    ]
    if any(kw in q_lower for kw in interp_keywords):
        return "interpretation"

    # Conceptual / domain knowledge
    concept_keywords = [
        "principle",
        "theorem",
        "formula",
        "law ",
        "property",
        "concept",
        "method",
        "approach",
        "technique",
        "algorithm",
        "correct.*approach",
        "appropriate.*method",
        "valid.*strategy",
    ]
    if any(kw in q_lower for kw in concept_keywords):
        return "conceptual"
    if any(re.search(kw, q_lower) for kw in concept_keywords if ".*" in kw):
        return "conceptual"

    # Step verification (checking intermediate steps)
    step_keywords = [
        "step",
        "intermediate",
        "first ",
        "then ",
        "next ",
        "after ",
        "before ",
        "sequence",
        "order of",
    ]
    if any(kw in q_lower for kw in step_keywords):
        return "step_verification"

    # Completeness / coverage
    complete_keywords = [
        "all ",
        "every ",
        "accounted for",
        "missing",
        "overlooked",
        "considered",
        "included",
        "covered",
        "comprehensive",
        "each ",
        "both ",
        "entire",
    ]
    if any(kw in q_lower for kw in complete_keywords):
        return "completeness"

    # Final answer verification
    final_keywords = [
        "final answer",
        "correct answer",
        "reasonable",
        "consistent",
        "match",
        "verify",
        "verification",
        "confirm",
        "double-check",
    ]
    if any(kw in q_lower for kw in final_keywords):
        return "answer_verification"

    # Edge case / boundary
    edge_keywords = [
        "edge case",
        "boundary",
        "special case",
        "exception",
        "zero",
        "negative",
        "overflow",
        "empty",
    ]
    if any(kw in q_lower for kw in edge_keywords):
        return "edge_case"

    return "other"


def compute_false_rate(questions, answers):
    """What fraction of answers were 'False' (i.e., the LLM corrected the SLM)."""
    if not answers:
        return 0.0
    false_count = sum(1 for a in answers if a.strip().lower() == "false")
    return false_count / len(answers)


def analyze_results(results):
    """Main analysis."""
    # Aggregate across all datasets
    all_recovered = []
    all_not_recovered = []
    all_regressed = []
    all_already_correct = []

    dataset_stats = {}

    for dataset, data in results.items():
        problems = data["problems"]
        recovered = [p for p in problems if classify_problem(p) == "recovered"]
        not_recovered = [
            p for p in problems if classify_problem(p) == "not_recovered"
        ]
        regressed = [p for p in problems if classify_problem(p) == "regressed"]
        already_correct = [
            p for p in problems if classify_problem(p) == "already_correct"
        ]

        all_recovered.extend(recovered)
        all_not_recovered.extend(not_recovered)
        all_regressed.extend(regressed)
        all_already_correct.extend(already_correct)

        dataset_stats[dataset] = {
            "total": len(problems),
            "recovered": len(recovered),
            "not_recovered": len(not_recovered),
            "regressed": len(regressed),
            "already_correct": len(already_correct),
        }

    print("=" * 80)
    print("QA COMPRESSION QUESTION ANALYSIS")
    print("=" * 80)

    # 1. Dataset-level summary
    print("\n--- Per-Dataset Summary ---\n")
    print(
        f"{'Dataset':<20} {'Total':>6} {'Already':>8} {'Recovered':>10} {'Not Rec.':>10} {'Regressed':>10} {'Recovery %':>10}"
    )
    print("-" * 80)
    for ds in sorted(dataset_stats.keys()):
        s = dataset_stats[ds]
        initially_wrong = s["recovered"] + s["not_recovered"]
        recovery_rate = s[
            "recovered"] / initially_wrong * 100 if initially_wrong > 0 else 0
        print(
            f"{ds:<20} {s['total']:>6} {s['already_correct']:>8} {s['recovered']:>10} {s['not_recovered']:>10} {s['regressed']:>10} {recovery_rate:>9.1f}%"
        )

    total_wrong = len(all_recovered) + len(all_not_recovered)
    print(
        f"\n{'TOTAL':<20} {len(all_recovered) + len(all_not_recovered) + len(all_regressed) + len(all_already_correct):>6} "
        f"{len(all_already_correct):>8} {len(all_recovered):>10} {len(all_not_recovered):>10} {len(all_regressed):>10} "
        f"{len(all_recovered) / total_wrong * 100 if total_wrong > 0 else 0:>9.1f}%"
    )

    # 2. Question type distribution
    print("\n--- Question Type Distribution ---\n")
    recovered_types = Counter()
    not_recovered_types = Counter()
    for p in all_recovered:
        for q in p["questions"]:
            recovered_types[classify_question(q)] += 1
    for p in all_not_recovered:
        for q in p["questions"]:
            not_recovered_types[classify_question(q)] += 1

    all_types = sorted(
        set(list(recovered_types.keys()) + list(not_recovered_types.keys())))
    total_rec_q = sum(recovered_types.values())
    total_nrec_q = sum(not_recovered_types.values())

    print(
        f"{'Question Type':<25} {'Recovered':>10} {'(%%)':>6} {'Not Rec.':>10} {'(%%)':>6} {'Diff':>8}"
    )
    print("-" * 70)
    for t in all_types:
        rc = recovered_types[t]
        nrc = not_recovered_types[t]
        rp = rc / total_rec_q * 100 if total_rec_q > 0 else 0
        nrp = nrc / total_nrec_q * 100 if total_nrec_q > 0 else 0
        diff = rp - nrp
        print(
            f"{t:<25} {rc:>10} {rp:>5.1f}% {nrc:>10} {nrp:>5.1f}% {diff:>+7.1f}%"
        )

    # 3. False-answer rate (how often LLM said "False" = corrected the SLM)
    print("\n--- 'False' Answer Rate (LLM corrections) ---\n")
    rec_false_rates = [
        compute_false_rate(p["questions"], p["answers"]) for p in all_recovered
    ]
    nrec_false_rates = [
        compute_false_rate(p["questions"], p["answers"])
        for p in all_not_recovered
    ]

    avg_rec_false = sum(rec_false_rates) / len(
        rec_false_rates) if rec_false_rates else 0
    avg_nrec_false = sum(nrec_false_rates) / len(
        nrec_false_rates) if nrec_false_rates else 0

    print(
        f"Recovered problems:     avg 'False' rate = {avg_rec_false:.1%}  (n={len(all_recovered)})"
    )
    print(
        f"Not-recovered problems: avg 'False' rate = {avg_nrec_false:.1%}  (n={len(all_not_recovered)})"
    )
    print()

    # Distribution of false counts
    rec_false_counts = Counter(
        sum(1 for a in p["answers"] if a.strip().lower() == "false")
        for p in all_recovered)
    nrec_false_counts = Counter(
        sum(1 for a in p["answers"] if a.strip().lower() == "false")
        for p in all_not_recovered)

    print(
        f"{'#False answers':<20} {'Recovered':>10} {'(%%)':>6} {'Not Rec.':>10} {'(%%)':>6}"
    )
    print("-" * 55)
    for n_false in range(
            max(max(rec_false_counts.keys(), default=0),
                max(nrec_false_counts.keys(), default=0)) + 1):
        rc = rec_false_counts.get(n_false, 0)
        nrc = nrec_false_counts.get(n_false, 0)
        print(
            f"{n_false:<20} {rc:>10} {rc / len(all_recovered) * 100 if all_recovered else 0:>5.1f}% {nrc:>10} {nrc / len(all_not_recovered) * 100 if all_not_recovered else 0:>5.1f}%"
        )

    # 4. Early stopping patterns
    print("\n--- Early Stopping Patterns ---\n")
    rec_early = sum(1 for p in all_recovered if p.get("early_stopped", False))
    nrec_early = sum(1 for p in all_not_recovered
                     if p.get("early_stopped", False))
    print(
        f"Recovered:     {rec_early}/{len(all_recovered)} early stopped ({rec_early/len(all_recovered)*100:.1f}%)"
        if all_recovered else "")
    print(
        f"Not recovered: {nrec_early}/{len(all_not_recovered)} early stopped ({nrec_early/len(all_not_recovered)*100:.1f}%)"
        if all_not_recovered else "")

    rec_nq = [
        p.get("n_questions_used", len(p["questions"])) for p in all_recovered
    ]
    nrec_nq = [
        p.get("n_questions_used", len(p["questions"]))
        for p in all_not_recovered
    ]
    print(
        f"\nAvg questions used - Recovered: {sum(rec_nq)/len(rec_nq):.1f}, Not recovered: {sum(nrec_nq)/len(nrec_nq):.1f}"
    )

    # 5. Question length analysis
    print("\n--- Question Length Analysis ---\n")
    rec_lens = [len(q) for p in all_recovered for q in p["questions"]]
    nrec_lens = [len(q) for p in all_not_recovered for q in p["questions"]]
    print(
        f"Recovered:     avg question length = {sum(rec_lens)/len(rec_lens):.0f} chars  (median: {sorted(rec_lens)[len(rec_lens)//2]:.0f})"
    )
    print(
        f"Not recovered: avg question length = {sum(nrec_lens)/len(nrec_lens):.0f} chars  (median: {sorted(nrec_lens)[len(nrec_lens)//2]:.0f})"
    )

    # 6. Quality score analysis
    print("\n--- Quality Score Analysis ---\n")
    rec_scores = [
        s for p in all_recovered for s in (p.get("quality_scores") or [])
    ]
    nrec_scores = [
        s for p in all_not_recovered for s in (p.get("quality_scores") or [])
    ]
    if rec_scores and nrec_scores:
        print(
            f"Recovered:     avg quality score = {sum(rec_scores)/len(rec_scores):.1f}  (n={len(rec_scores)} evaluations)"
        )
        print(
            f"Not recovered: avg quality score = {sum(nrec_scores)/len(nrec_scores):.1f}  (n={len(nrec_scores)} evaluations)"
        )

    # 7. First question that gets a "False" answer - position analysis
    print("\n--- Position of First 'False' Answer ---\n")
    rec_first_false = []
    nrec_first_false = []
    for p in all_recovered:
        for i, a in enumerate(p["answers"]):
            if a.strip().lower() == "false":
                rec_first_false.append(i + 1)  # 1-indexed
                break
        else:
            rec_first_false.append(None)  # no false at all
    for p in all_not_recovered:
        for i, a in enumerate(p["answers"]):
            if a.strip().lower() == "false":
                nrec_first_false.append(i + 1)
                break
        else:
            nrec_first_false.append(None)

    rec_has_false = [x for x in rec_first_false if x is not None]
    nrec_has_false = [x for x in nrec_first_false if x is not None]
    rec_no_false = len(rec_first_false) - len(rec_has_false)
    nrec_no_false = len(nrec_first_false) - len(nrec_has_false)

    print(
        f"Recovered:     {len(rec_has_false)}/{len(all_recovered)} ({len(rec_has_false)/len(all_recovered)*100:.0f}%) had at least one 'False'"
    )
    if rec_has_false:
        print(
            f"               avg position of first False: {sum(rec_has_false)/len(rec_has_false):.1f}"
        )
    print(
        f"Not recovered: {len(nrec_has_false)}/{len(all_not_recovered)} ({len(nrec_has_false)/len(all_not_recovered)*100:.0f}%) had at least one 'False'"
    )
    if nrec_has_false:
        print(
            f"               avg position of first False: {sum(nrec_has_false)/len(nrec_has_false):.1f}"
        )

    # 8. Concrete examples - show a few recovered and not-recovered transcripts
    print("\n" + "=" * 80)
    print("EXAMPLE TRANSCRIPTS")
    print("=" * 80)

    # Pick diverse examples
    for dataset in sorted(results.keys()):
        problems = results[dataset]["problems"]
        recovered = [p for p in problems if classify_problem(p) == "recovered"]
        not_recovered = [
            p for p in problems if classify_problem(p) == "not_recovered"
        ]

        if recovered:
            p = recovered[0]
            false_rate = compute_false_rate(p["questions"], p["answers"])
            n_false = sum(1 for a in p["answers"]
                          if a.strip().lower() == "false")
            print(
                f"\n--- RECOVERED [{dataset}] idx={p['idx']} difficulty={p['difficulty']} false_rate={false_rate:.0%} ({n_false} False) ---"
            )
            for i, (q, a) in enumerate(zip(p["questions"], p["answers"])):
                marker = " <<<" if a.strip().lower() == "false" else ""
                print(f"  Q{i+1}: {q}")
                print(f"  A{i+1}: {a}{marker}")

        if not_recovered:
            p = not_recovered[0]
            false_rate = compute_false_rate(p["questions"], p["answers"])
            n_false = sum(1 for a in p["answers"]
                          if a.strip().lower() == "false")
            print(
                f"\n--- NOT RECOVERED [{dataset}] idx={p['idx']} difficulty={p['difficulty']} false_rate={false_rate:.0%} ({n_false} False) ---"
            )
            for i, (q, a) in enumerate(zip(p["questions"], p["answers"])):
                marker = " <<<" if a.strip().lower() == "false" else ""
                print(f"  Q{i+1}: {q}")
                print(f"  A{i+1}: {a}{marker}")

    # 9. "All True" analysis - when every answer is True
    print("\n" + "=" * 80)
    print("ALL-TRUE ANALYSIS (every answer is 'True')")
    print("=" * 80)
    rec_all_true = sum(1 for p in all_recovered if all(
        a.strip().lower() == "true" for a in p["answers"]))
    nrec_all_true = sum(1 for p in all_not_recovered if all(
        a.strip().lower() == "true" for a in p["answers"]))
    print(
        f"\nRecovered:     {rec_all_true}/{len(all_recovered)} ({rec_all_true/len(all_recovered)*100:.0f}%) had ALL True answers"
    )
    print(
        f"Not recovered: {nrec_all_true}/{len(all_not_recovered)} ({nrec_all_true/len(all_not_recovered)*100:.0f}%) had ALL True answers"
    )
    print()
    print(
        "(All-True = the LLM never corrected the SLM, yet the SLM still changed its answer)"
    )


if __name__ == "__main__":
    results = load_all_qa_results()
    if not results:
        print(f"No QA result files found in {RESULTS_DIR}")
        sys.exit(1)
    print(
        f"Loaded {len(results)} datasets: {', '.join(sorted(results.keys()))}")
    analyze_results(results)
