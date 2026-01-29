"""
Standardize CSV Columns Across CICIDS2018 Files

This script ensures all CSV files have identical columns by:
1. Finding common columns across all files
2. Removing columns that only exist in some files
3. Creating backups before modifications
"""

import pandas as pd
import os
import shutil
from datetime import datetime

RAW_DATA_DIR = 'data/raw'
BACKUP_DIR = 'data/raw_backup'

def main():
    print("=" * 60)
    print("CICIDS2018 CSV Column Standardization")
    print("=" * 60)
    
    # Find all CSV files
    csv_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {RAW_DATA_DIR}")
        print("Please download CICIDS2018 data and place CSV files in data/raw/")
        return
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  • {csv_file}")
    
    # Step 1: Read column names from all files
    print("\n" + "-" * 60)
    print("Step 1: Reading column names from all files...")
    print("-" * 60)
    
    all_columns = {}
    for csv_file in csv_files:
        file_path = os.path.join(RAW_DATA_DIR, csv_file)
        try:
            df = pd.read_csv(file_path, nrows=1)
            columns = list(df.columns)
            all_columns[csv_file] = columns
            print(f"✓ {csv_file}: {len(columns)} columns")
        except Exception as e:
            print(f"✗ {csv_file}: Error - {e}")
            return
    
    # Step 2: Find common columns
    print("\n" + "-" * 60)
    print("Step 2: Finding common columns...")
    print("-" * 60)
    
    # Get intersection of all column sets
    common_columns = set(all_columns[csv_files[0]])
    for csv_file in csv_files[1:]:
        common_columns = common_columns.intersection(set(all_columns[csv_file]))
    
    common_columns = sorted(list(common_columns))
    print(f"\nCommon columns across all files: {len(common_columns)}")
    
    # Step 3: Identify files with extra columns
    files_with_extra = []
    for csv_file, columns in all_columns.items():
        extra_columns = set(columns) - set(common_columns)
        if extra_columns:
            files_with_extra.append((csv_file, extra_columns))
            print(f"\n⚠️  {csv_file} has {len(extra_columns)} extra column(s):")
            for col in sorted(extra_columns):
                print(f"     - {col}")
    
    if not files_with_extra:
        print("\n✅ All files already have identical columns!")
        print("No modifications needed.")
        return
    
    # Step 4: Create backups
    print("\n" + "-" * 60)
    print("Step 4: Creating backups...")
    print("-" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_DIR}_{timestamp}"
    
    try:
        os.makedirs(backup_path, exist_ok=True)
        for csv_file, extra_cols in files_with_extra:
            src = os.path.join(RAW_DATA_DIR, csv_file)
            dst = os.path.join(backup_path, csv_file)
            shutil.copy2(src, dst)
            print(f"✓ Backed up: {csv_file}")
        print(f"\nBackups saved to: {backup_path}")
    except Exception as e:
        print(f"✗ Backup failed: {e}")
        return
    
    # Step 5: Standardize files
    print("\n" + "-" * 60)
    print("Step 5: Standardizing CSV files...")
    print("-" * 60)
    
    for csv_file, extra_cols in files_with_extra:
        file_path = os.path.join(RAW_DATA_DIR, csv_file)
        try:
            print(f"\nProcessing: {csv_file}")
            
            # Read full file
            df = pd.read_csv(file_path)
            original_rows = len(df)
            original_cols = len(df.columns)
            
            # Keep only common columns
            df_standardized = df[common_columns]
            
            # Save back
            df_standardized.to_csv(file_path, index=False)
            
            print(f"  ✓ Removed {len(extra_cols)} column(s)")
            print(f"  ✓ Before: {original_rows} rows × {original_cols} columns")
            print(f"  ✓ After:  {original_rows} rows × {len(common_columns)} columns")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return
    
    # Step 6: Final verification
    print("\n" + "-" * 60)
    print("Step 6: Verifying standardization...")
    print("-" * 60)
    
    all_same = True
    for csv_file in csv_files:
        file_path = os.path.join(RAW_DATA_DIR, csv_file)
        df = pd.read_csv(file_path, nrows=1)
        if len(df.columns) != len(common_columns):
            print(f"✗ {csv_file}: {len(df.columns)} columns (expected {len(common_columns)})")
            all_same = False
        else:
            print(f"✓ {csv_file}: {len(df.columns)} columns")
    
    if all_same:
        print("\n" + "=" * 60)
        print("✅ SUCCESS! All CSV files now have identical columns.")
        print("=" * 60)
        print(f"\nStandardized to {len(common_columns)} common columns")
        print(f"Backups available in: {backup_path}")
        print("\nYou can now run: python main.py --module 1")
    else:
        print("\n❌ Verification failed. Please check errors above.")

if __name__ == "__main__":
    main()
