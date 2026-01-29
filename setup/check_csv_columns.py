"""
Check CSV Column Consistency

Quick script to check if all CSV files have the same number of columns.
Run this after standardize_csv_columns.py to verify.
"""

import pandas as pd
import os

RAW_DATA_DIR = 'data/raw'

def main():
    print("=" * 60)
    print("CSV Column Check")
    print("=" * 60)
    
    csv_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {RAW_DATA_DIR}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files\n")
    
    column_info = {}
    for csv_file in csv_files:
        file_path = os.path.join(RAW_DATA_DIR, csv_file)
        try:
            df = pd.read_csv(file_path, nrows=1)
            num_cols = len(df.columns)
            column_info[csv_file] = num_cols
            print(f"  {csv_file:<55} {num_cols} columns")
        except Exception as e:
            print(f"  {csv_file:<55} ERROR: {e}")
    
    # Check if all have same number
    unique_counts = set(column_info.values())
    
    print("\n" + "=" * 60)
    if len(unique_counts) == 1:
        print("✅ All CSV files have identical column count!")
        print(f"   Common columns: {list(unique_counts)[0]}")
        print("\nReady for training!")
    else:
        print("⚠️  CSV files have different column counts:")
        for count in sorted(unique_counts):
            files_with_count = [f for f, c in column_info.items() if c == count]
            print(f"   {count} columns: {len(files_with_count)} file(s)")
            for f in files_with_count:
                print(f"      - {f}")
        print("\n⚠️  Run: python setup/standardize_csv_columns.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
