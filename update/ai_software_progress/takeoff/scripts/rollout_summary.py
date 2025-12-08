#!/usr/bin/env python3
"""
Generate a summary of rollout success/failure statistics.

This script analyzes rollouts.jsonl files to count successful and failed rollouts,
and creates a summary file in the run directory.

Usage:
    python scripts/rollout_summary.py <run_dir>
    python scripts/rollout_summary.py outputs/1105_daniel
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Import RolloutsReader for unified rollout parsing
from plotting_utils.rollouts_reader import RolloutsReader


def analyze_rollouts(rollouts_path: Path) -> Dict[str, Any]:
    """Analyze a rollouts.jsonl file and return statistics.

    Args:
        rollouts_path: Path to rollouts.jsonl file

    Returns:
        Dictionary with statistics about successful and failed rollouts
    """
    total_rollouts = 0
    successful_rollouts = 0
    failed_rollouts = 0
    timeout_rollouts = 0
    error_types: Dict[str, int] = {}
    error_details: List[Dict[str, Any]] = []

    # Use RolloutsReader to iterate over all records (including errors)
    reader = RolloutsReader(rollouts_path)

    for record in reader.iter_all_records():
        total_rollouts += 1

        # Check if this rollout has an error
        if "error" in record:
            failed_rollouts += 1
            error_msg = record["error"]

            # Categorize error type
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_type = "Timeout"
                timeout_rollouts += 1
            elif "validation" in error_msg.lower():
                error_type = "Validation Error"
            elif "infeasible" in error_msg.lower():
                error_type = "Infeasible Parameters"
            else:
                error_type = "Other Error"

            error_types[error_type] = error_types.get(error_type, 0) + 1

            # Store error details
            error_details.append({
                "sample_id": record.get("sample_id", total_rollouts - 1),
                "error_type": error_type,
                "error_message": error_msg[:200]  # Truncate long messages
            })
        else:
            # Check if results are present and valid
            results = record.get("results")
            if results is not None and isinstance(results, dict):
                successful_rollouts += 1
            else:
                # No error field but no valid results either
                failed_rollouts += 1
                error_types["Missing Results"] = error_types.get("Missing Results", 0) + 1
                error_details.append({
                    "sample_id": record.get("sample_id", total_rollouts - 1),
                    "error_type": "Missing Results",
                    "error_message": "Results field is None or invalid"
                })

    return {
        "total_rollouts": total_rollouts,
        "successful_rollouts": successful_rollouts,
        "failed_rollouts": failed_rollouts,
        "timeout_rollouts": timeout_rollouts,
        "success_rate": successful_rollouts / total_rollouts if total_rollouts > 0 else 0.0,
        "failure_rate": failed_rollouts / total_rollouts if total_rollouts > 0 else 0.0,
        "error_types": error_types,
        "error_details": error_details
    }


def format_summary(stats: Dict[str, Any]) -> str:
    """Format statistics as a human-readable summary.

    Args:
        stats: Statistics dictionary from analyze_rollouts

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "ROLLOUT SUMMARY",
        "=" * 60,
        "",
        f"Total Rollouts:      {stats['total_rollouts']:5d}",
        f"Successful:          {stats['successful_rollouts']:5d} ({stats['success_rate']:6.2%})",
        f"Failed:              {stats['failed_rollouts']:5d} ({stats['failure_rate']:6.2%})",
        ""
    ]

    if stats['error_types']:
        lines.append("FAILURE BREAKDOWN:")
        lines.append("-" * 60)
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
            pct = count / stats['total_rollouts'] if stats['total_rollouts'] > 0 else 0.0
            lines.append(f"  {error_type:30s} {count:5d} ({pct:6.2%})")
        lines.append("")

    if stats['error_details']:
        lines.append(f"FIRST {min(10, len(stats['error_details']))} ERRORS:")
        lines.append("-" * 60)
        for i, error in enumerate(stats['error_details'][:10], start=1):
            lines.append(f"{i}. Sample ID {error['sample_id']}: {error['error_type']}")
            lines.append(f"   {error['error_message']}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate rollout summary statistics")
    parser.add_argument("run_dir", type=str, help="Path to run directory containing rollouts.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: run_dir/rollout_summary.txt)")
    parser.add_argument("--json", action="store_true", help="Also output JSON format summary")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rollouts_path = run_dir / "rollouts.jsonl"

    if not rollouts_path.exists():
        print(f"Error: rollouts.jsonl not found at {rollouts_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {rollouts_path}...")
    stats = analyze_rollouts(rollouts_path)

    # Generate summary text
    summary_text = format_summary(stats)

    # Print to console
    print(summary_text)

    # Write to file
    output_path = Path(args.output) if args.output else run_dir / "rollout_summary.txt"
    output_path.write_text(summary_text)
    print(f"\nSummary written to {output_path}")

    # Optionally write JSON summary
    if args.json:
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"JSON summary written to {json_path}")


if __name__ == "__main__":
    main()
