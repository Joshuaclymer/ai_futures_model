"""
Inspect the reference API response to understand data structure.
"""

import sys
import json
import urllib.request
import time


def call_reference_api(num_samples: int = 10, start_year: int = 2029, total_labor: int = 11300, timeout: int = 180):
    url = 'https://dark-compute.onrender.com/run_simulation'
    data = json.dumps({
        'num_samples': num_samples,
        'start_year': start_year,
        'total_labor': total_labor,
    }).encode()

    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            elapsed = time.time() - start_time
            print(f"API response received in {elapsed:.1f}s")
            return result
    except Exception as e:
        print(f"ERROR calling API: {e}")
        return None


def print_structure(d, prefix="", max_depth=4, depth=0):
    """Print the structure of a nested dict, showing keys and value types."""
    if depth >= max_depth:
        return

    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: dict with {len(value)} keys")
                print_structure(value, prefix + "  ", max_depth, depth + 1)
            elif isinstance(value, list):
                if len(value) > 0:
                    sample = value[0]
                    print(f"{prefix}{key}: list[{len(value)}] of {type(sample).__name__}")
                    if isinstance(sample, dict):
                        print_structure(sample, prefix + "  ", max_depth, depth + 1)
                    elif isinstance(sample, (int, float)) and len(value) > 3:
                        print(f"{prefix}  sample values: {value[:3]}...{value[-3:]}")
                else:
                    print(f"{prefix}{key}: list[] (empty)")
            else:
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"{prefix}{key}: {type(value).__name__} = {val_str}")


def main():
    print("Fetching reference API response...")
    response = call_reference_api(num_samples=5, start_year=2029, total_labor=11300)

    if not response:
        print("Failed to get response")
        return

    print("\n" + "=" * 70)
    print("REFERENCE API RESPONSE STRUCTURE")
    print("=" * 70 + "\n")

    print_structure(response)

    # Specifically look for fab-related data
    print("\n" + "=" * 70)
    print("FAB-RELATED DATA")
    print("=" * 70)

    black_fab = response.get('black_fab', {})
    if black_fab:
        print("\nblack_fab section:")
        print_structure(black_fab, "  ")

    black_project_model = response.get('black_project_model', {})
    if black_project_model:
        print("\nblack_project_model section (selected keys):")
        for key in ['years', 'total_black_project', 'operational_compute', 'individual_project_fab_production']:
            if key in black_project_model:
                value = black_project_model[key]
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: list[{len(value)}] of {type(value[0]).__name__}")
                    if isinstance(value[0], (int, float)):
                        print(f"    sample: {value[:3]}...{value[-3:]}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                    for k, v in value.items():
                        if isinstance(v, list) and len(v) > 0:
                            print(f"    {k}: list[{len(v)}] sample={v[:3]}...{v[-3:]}")

    # Check initial stock section
    initial_stock = response.get('initial_stock', {})
    if initial_stock:
        print("\ninitial_stock section (selected keys):")
        for key in ['initial_compute_stock', 'diverted_compute', 'initial_prc_black_project']:
            if key in initial_stock:
                value = initial_stock[key]
                print(f"  {key}: {type(value).__name__}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, list) and len(v) > 0:
                            print(f"    {k}: list[{len(v)}] sample={v[:3]}")

    # Check initial_black_project
    ibp = response.get('initial_black_project', {})
    if ibp:
        print("\ninitial_black_project section:")
        print_structure(ibp, "  ")

    # Print raw total_black_project if available
    tbp = black_project_model.get('total_black_project', {})
    if tbp:
        print("\n\ntotal_black_project.median full values:")
        median = tbp.get('median', [])
        years = black_project_model.get('years', [])
        if median and years:
            for i in [0, len(median)//4, len(median)//2, 3*len(median)//4, len(median)-1]:
                if i < len(median) and i < len(years):
                    print(f"  Year {years[i]:.1f}: {median[i]:.1f} H100e")


if __name__ == "__main__":
    main()
