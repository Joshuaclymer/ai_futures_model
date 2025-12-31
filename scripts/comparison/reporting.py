"""
Reporting functions for comparison results.
"""

from datetime import datetime
from pathlib import Path
from typing import List

from .metrics import ComparisonResult


def print_results(results: List[ComparisonResult], show_details: bool = True):
    """
    Print comparison results to console.

    Args:
        results: List of ComparisonResult objects
        show_details: Whether to show detailed statistics
    """
    # Status symbols and colors
    status_symbols = {
        'PASS': '✓',
        'FAIL': '✗',
        'WARN': '⚠',
        'SKIP': '○',
        'INFO': 'ℹ',
    }

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Count by status
    counts = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'SKIP': 0, 'INFO': 0}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        symbol = status_symbols.get(result.status, '?')

        if show_details:
            print(f"\n{symbol} {result.metric_name}: {result.status}")
            if result.status != 'SKIP':
                print(f"    Local:  {result.local_median:,.2f} (p10={result.local_p10:,.2f}, p90={result.local_p90:,.2f})")
                print(f"    Ref:    {result.reference_median:,.2f} (p10={result.reference_p10:,.2f}, p90={result.reference_p90:,.2f})")
                print(f"    Diff:   {result.pct_diff:.2f}% {'(within CI)' if result.within_ci else ''}")
            if result.note:
                print(f"    Note:   {result.note}")
        else:
            note = f" ({result.note})" if result.note else ""
            print(f"  {symbol} {result.metric_name}: {result.pct_diff:.2f}% diff{note}")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  Passed: {counts['PASS']}")
    print(f"  Failed: {counts['FAIL']}")
    print(f"  Warnings: {counts['WARN']}")
    print(f"  Skipped: {counts['SKIP']}")
    if counts['INFO']:
        print(f"  Info (intentional): {counts['INFO']}")

    total_testable = counts['PASS'] + counts['FAIL'] + counts['WARN']
    if total_testable > 0:
        pass_rate = counts['PASS'] / total_testable * 100
        print(f"\n  Pass Rate: {pass_rate:.1f}% ({counts['PASS']}/{total_testable})")

    print("=" * 80)


def generate_markdown_report(
    results: List[ComparisonResult],
    output_path: Path = None,
    title: str = "Model Comparison Report",
) -> str:
    """
    Generate a markdown report of comparison results.

    Args:
        results: List of ComparisonResult objects
        output_path: Optional path to write the report
        title: Report title

    Returns:
        Markdown string
    """
    lines = [
        f"# {title}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Summary",
        f"",
    ]

    # Count by status
    counts = {'PASS': 0, 'FAIL': 0, 'WARN': 0, 'SKIP': 0, 'INFO': 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    total_testable = counts['PASS'] + counts['FAIL'] + counts['WARN']
    pass_rate = counts['PASS'] / total_testable * 100 if total_testable > 0 else 0

    lines.extend([
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✓ Pass | {counts['PASS']} |",
        f"| ✗ Fail | {counts['FAIL']} |",
        f"| ⚠ Warn | {counts['WARN']} |",
        f"| ○ Skip | {counts['SKIP']} |",
    ])

    if counts['INFO']:
        lines.append(f"| ℹ Info | {counts['INFO']} |")

    lines.extend([
        f"",
        f"**Pass Rate: {pass_rate:.1f}%** ({counts['PASS']}/{total_testable})",
        f"",
        f"## Detailed Results",
        f"",
        f"| Metric | Status | Local Median | Ref Median | % Diff | Note |",
        f"|--------|--------|--------------|------------|--------|------|",
    ])

    status_symbols = {
        'PASS': '✓',
        'FAIL': '✗',
        'WARN': '⚠',
        'SKIP': '○',
        'INFO': 'ℹ',
    }

    for result in results:
        symbol = status_symbols.get(result.status, '?')
        if result.status == 'SKIP':
            lines.append(
                f"| {result.metric_name} | {symbol} {result.status} | - | - | - | {result.note} |"
            )
        else:
            lines.append(
                f"| {result.metric_name} | {symbol} {result.status} | "
                f"{result.local_median:,.2f} | {result.reference_median:,.2f} | "
                f"{result.pct_diff:.2f}% | {result.note} |"
            )

    lines.extend([
        f"",
        f"## Interpretation",
        f"",
        f"- **PASS**: Difference within tolerance",
        f"- **WARN**: Difference exceeds tolerance but local median within reference CI",
        f"- **FAIL**: Difference exceeds tolerance and outside reference CI",
        f"- **SKIP**: Metric not available in one or both models",
        f"- **INFO**: Known intentional difference",
        f"",
    ])

    markdown = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(markdown)

    return markdown
