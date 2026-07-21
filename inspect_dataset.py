"""Quick script to inspect the dataset files."""
import csv
import os

BASE = os.path.dirname(__file__)

for fname in ["generation_validation.csv", "multiple_choice_validation.csv"]:
    path = os.path.join(BASE, "dataset", fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"Columns: {headers}")
        
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 2:
                break
        
        print(f"Total columns: {len(headers)}")
        print(f"\nFirst row details:")
        for h in headers:
            val = rows[0][h][:120] if rows[0][h] else "(empty)"
            print(f"  {h}: {val}")
    
    # Count total rows
    with open(path, encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1
    print(f"\nTotal rows: {total}")
