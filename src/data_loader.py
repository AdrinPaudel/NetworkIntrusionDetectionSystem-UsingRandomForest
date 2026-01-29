"""
MODULE 1: DATA LOADING
Load CICIDS2018 dataset from multiple CSV files, optimize data types, and validate
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import (
    log_message, print_section_header, print_subsection_header,
    format_time, format_number, calculate_memory_usage, Timer
)
import config


def find_csv_files(directory=None):
    """
    Find all CSV files in the raw data directory.
    
    Parameters:
    -----------
    directory : str, optional
        Directory to search for CSV files
        
    Returns:
    --------
    list : List of CSV file paths
    """
    if directory is None:
        directory = config.DATA_RAW_DIR
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all CSV files
    csv_pattern = os.path.join(directory, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {directory}")
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    log_message(f"Found {len(csv_files)} CSV files in {directory}", level="INFO")
    for i, filepath in enumerate(csv_files, 1):
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath) / (1024 ** 3)  # GB
        log_message(f"  {i}. {filename} ({file_size:.2f} GB)", level="INFO", print_timestamp=False)
    
    return csv_files


def load_single_csv(filepath):
    """
    Load a single CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pandas.DataFrame : Loaded data
    """
    filename = os.path.basename(filepath)
    log_message(f"Loading: {filename}", level="INFO")
    
    try:
        with Timer(f"Load {filename}", verbose=False):
            # Read CSV and let pandas infer numeric types automatically
            # This converts numeric strings to actual numbers while preserving values
            df = pd.read_csv(
                filepath,
                low_memory=False,
                encoding='utf-8',
                skip_blank_lines=True,
                on_bad_lines='warn'  # Warn about bad lines but skip them
            )
            
            # Convert numeric-looking columns to numeric (coerce invalid values to NaN)
            # This preserves data values but ensures proper numeric types for analysis
            for col in df.columns:
                # Skip known non-numeric columns
                if col not in ['Label', 'Protocol', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port']:
                    # Try converting to numeric, non-numeric values become NaN
                    numeric_version = pd.to_numeric(df[col], errors='coerce')
                    # Only replace if at least 50% of values are numeric
                    if numeric_version.notna().sum() / len(df) > 0.5:
                        df[col] = numeric_version
        
        log_message(f"  Loaded {format_number(len(df))} rows, {len(df.columns)} columns", 
                   level="INFO", print_timestamp=False)
        return df
    
    except Exception as e:
        log_message(f"Error loading {filename}: {str(e)}", level="ERROR")
        raise


def load_all_csv_files(csv_files, parallel=True, max_workers=None):
    """
    Load and concatenate all CSV files in parallel.
    
    Parameters:
    -----------
    csv_files : list
        List of CSV file paths
    parallel : bool
        If True, load files in parallel using multiple threads
    max_workers : int, optional
        Maximum number of parallel workers (default: number of CSV files)
        
    Returns:
    --------
    pandas.DataFrame : Combined data
    """
    log_message(f"Loading {len(csv_files)} CSV files...", level="STEP")
    
    if parallel and len(csv_files) > 1:
        # Use parallel loading with smart scheduling (larger files first)
        if max_workers is None:
            max_workers = min(len(csv_files), os.cpu_count() or 4)
        
        log_message(f"Using parallel loading with {max_workers} workers", level="INFO")
        print()
        
        # Sort files by size (descending) so larger files get loaded first
        # This improves overall loading time with ThreadPoolExecutor
        csv_files_with_sizes = [(f, os.path.getsize(f)) for f in csv_files]
        csv_files_sorted = [f[0] for f in sorted(csv_files_with_sizes, key=lambda x: x[1], reverse=True)]
        
        # Create mapping from sorted position to original position (for result ordering)
        index_mapping = {i: csv_files.index(f) for i, f in enumerate(csv_files_sorted)}
        
        dataframes = [None] * len(csv_files)  # Preserve original order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with larger files first
            future_to_index = {
                executor.submit(load_single_csv, filepath): index_mapping[i]
                for i, filepath in enumerate(csv_files_sorted)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    df = future.result()
                    dataframes[original_index] = df
                    log_message(f"[{original_index + 1}/{len(csv_files)}] Completed", level="INFO")
                except Exception as e:
                    log_message(f"Error loading file {original_index + 1}: {str(e)}", level="ERROR")
                    raise
        print()
    else:
        # Sequential loading (fallback)
        log_message("Using sequential loading", level="INFO")
        print()
        
        dataframes = []
        for i, filepath in enumerate(csv_files, 1):
            log_message(f"[File {i}/{len(csv_files)}]", level="SUBSTEP")
            df = load_single_csv(filepath)
            dataframes.append(df)
            print()
    
    log_message("Concatenating all files...", level="INFO")
    
    with Timer("Concatenation", verbose=False):
        combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    
    log_message(f"Combined dataset: {format_number(len(combined_df))} rows, "
               f"{len(combined_df.columns)} columns", level="SUCCESS")
    
    return combined_df


# REMOVED: optimize_dtypes() function
# User has 208GB RAM and wants original data types preserved
# No dtype optimization performed - data loaded AS-IS from CSV files


def validate_data(df):
    """
    Validate dataset and identify label/protocol columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    tuple : (df, label_column_name, protocol_column_name)
    """
    log_message("Validating dataset...", level="STEP")
    
    # Check if DataFrame is empty
    if len(df) == 0:
        raise ValueError("Dataset is empty (0 rows)")
    
    # Find Label column
    label_col = None
    for candidate in config.LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col is None:
        raise ValueError(
            f"Could not find Label column. "
            f"Tried: {config.LABEL_COLUMN_CANDIDATES}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    log_message(f"Label column found: '{label_col}'", level="SUCCESS")
    
    # Remove rows where Label == 'Label' (misplaced header rows)
    # Use memory-efficient approach to avoid "Killed" error
    log_message("Removing rows with Label == 'Label' (misplaced headers)...", level="SUBSTEP")
    
    bad_label_count = (df[label_col] == 'Label').sum()
    
    if bad_label_count > 0:
        # Memory-efficient: use query() to avoid creating large boolean mask
        df = df.query(f"`{label_col}` != 'Label'").reset_index(drop=True)
        log_message(f"Removed {format_number(bad_label_count)} rows with misplaced headers", level="INFO")
    else:
        log_message("No misplaced header rows found", level="INFO")
    print()
    
    # Find Protocol column (optional)
    protocol_col = None
    for candidate in config.PROTOCOL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            protocol_col = candidate
            break
    
    if protocol_col:
        log_message(f"Protocol column found: '{protocol_col}'", level="SUCCESS")
    else:
        log_message("Protocol column not found (optional)", level="WARNING")
    
    # Check for feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    log_message(f"Numeric columns: {len(numeric_cols)}", level="INFO")
    
    if len(numeric_cols) < 50:
        log_message(f"Warning: Expected at least 50 numeric columns, found {len(numeric_cols)}", 
                   level="WARNING")
    
    # Check unique labels
    unique_labels = df[label_col].nunique()
    log_message(f"Unique labels: {unique_labels}", level="INFO")
    
    return df, label_col, protocol_col


def get_initial_statistics(df, label_col):
    """
    Calculate and display initial dataset statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    label_col : str
        Name of label column
        
    Returns:
    --------
    dict : Statistics dictionary
    """
    log_message("Calculating initial statistics...", level="STEP")
    print()
    
    # Basic stats
    n_rows = len(df)
    n_cols = len(df.columns)
    memory_gb = calculate_memory_usage(df)
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    
    # Numeric vs categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Label distribution
    label_counts = df[label_col].value_counts().sort_values(ascending=False)
    label_percentages = (label_counts / len(df) * 100)
    
    # Missing values
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    nan_percentage = (total_nan / (n_rows * n_cols)) * 100
    
    # Infinite values (only for numeric columns)
    inf_counts = pd.Series(0, index=df.columns)
    for col in numeric_cols:
        inf_counts[col] = np.isinf(df[col]).sum()
    total_inf = inf_counts.sum()
    inf_percentage = (total_inf / (n_rows * n_cols)) * 100
    
    # Duplicates
    n_duplicates = df.duplicated().sum()
    dup_percentage = (n_duplicates / n_rows) * 100
    
    # Print statistics
    print_section_header("INITIAL DATASET STATISTICS")
    print(f"Total Rows:          {format_number(n_rows)}")
    print(f"Total Columns:       {n_cols}")
    print(f"Memory Usage:        {memory_gb:.2f} GB")
    print()
    
    print("Data Types Distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  - {dtype}: {count} columns")
    print()
    
    print(f"Numeric Columns:     {len(numeric_cols)}")
    print(f"Categorical Columns: {len(categorical_cols)}")
    print()
    
    print("Label Distribution:")
    for label, count in label_counts.items():
        pct = label_percentages[label]
        print(f"  {label}: {format_number(count)} ({pct:.2f}%)")
    print()
    
    print(f"Missing Values (NaN):")
    print(f"  Total NaN cells:   {format_number(total_nan)} ({nan_percentage:.3f}%)")
    if total_nan > 0:
        nan_cols = nan_counts[nan_counts > 0]
        print(f"  Affected columns:  {len(nan_cols)}")
        for col, count in nan_cols.head(5).items():
            print(f"    - {col}: {format_number(count)}")
    print()
    
    print(f"Infinite Values (Inf):")
    print(f"  Total Inf cells:   {format_number(total_inf)} ({inf_percentage:.3f}%)")
    if total_inf > 0:
        inf_cols = inf_counts[inf_counts > 0]
        print(f"  Affected columns:  {len(inf_cols)}")
        for col, count in inf_cols.head(5).items():
            print(f"    - {col}: {format_number(count)}")
    print()
    
    print(f"Duplicate Rows:      {format_number(n_duplicates)} ({dup_percentage:.3f}%)")
    print("=" * 80)
    print()
    
    # Return statistics dictionary
    stats = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'memory_gb': memory_gb,
        'dtype_counts': dtype_counts.to_dict(),
        'n_numeric_cols': len(numeric_cols),
        'n_categorical_cols': len(categorical_cols),
        'label_counts': label_counts.to_dict(),
        'label_percentages': label_percentages.to_dict(),
        'total_nan': total_nan,
        'nan_percentage': nan_percentage,
        'nan_by_column': nan_counts[nan_counts > 0].to_dict(),
        'total_inf': total_inf,
        'inf_percentage': inf_percentage,
        'inf_by_column': inf_counts[inf_counts > 0].to_dict(),
        'n_duplicates': n_duplicates,
        'dup_percentage': dup_percentage
    }
    
    return stats


def save_module1_checkpoint(df, label_col, protocol_col, stats):
    """
    Save Module 1 output to checkpoint file for reuse.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Loaded dataset
    label_col : str
        Label column name
    protocol_col : str
        Protocol column name
    stats : dict
        Loading statistics
    """
    import joblib
    
    checkpoint_dir = config.DATA_PREPROCESSED_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'module1_checkpoint.joblib')
    
    checkpoint_data = {
        'df': df,
        'label_col': label_col,
        'protocol_col': protocol_col,
        'stats': stats
    }
    
    joblib.dump(checkpoint_data, checkpoint_path)
    log_message(f"✓ Module 1 checkpoint saved: module1_checkpoint.joblib", level="SUCCESS")


def load_module1_checkpoint():
    """
    Load Module 1 output from checkpoint file.
    
    Returns:
    --------
    tuple : (df, label_col, protocol_col, stats) or None if not found
    """
    import joblib
    
    checkpoint_path = os.path.join(config.DATA_PREPROCESSED_DIR, 'module1_checkpoint.joblib')
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint_data = joblib.load(checkpoint_path)
        log_message(f"✓ Loaded Module 1 checkpoint (skipping data loading)", level="SUCCESS")
        return (
            checkpoint_data['df'],
            checkpoint_data['label_col'],
            checkpoint_data['protocol_col'],
            checkpoint_data['stats']
        )
    except Exception as e:
        log_message(f"Failed to load checkpoint: {str(e)}", level="WARNING")
        return None


def load_data(use_checkpoint=True):
    """
    Main function to load and prepare dataset.
    
    Parameters:
    -----------
    use_checkpoint : bool
        If True, try to load from checkpoint first
    
    Returns:
    --------
    tuple : (df, label_col, protocol_col, stats)
    """
    # Try loading from checkpoint first
    if use_checkpoint:
        checkpoint_result = load_module1_checkpoint()
        if checkpoint_result is not None:
            return checkpoint_result
    
    print()
    print_section_header("MODULE 1: DATA LOADING")
    print()
    
    overall_timer = Timer("Module 1: Data Loading", verbose=False)
    overall_timer.__enter__()
    
    try:
        # Step 1: Find CSV files
        csv_files = find_csv_files()
        print()
        
        # Step 2: Load all CSV files (in parallel)
        df = load_all_csv_files(csv_files, parallel=True)
        print()
        
        # Step 3: NO dtype optimization - preserve original types
        log_message("Preserving original data types (no optimization)", level="INFO")
        
        # Step 4: Validate data
        df, label_col, protocol_col = validate_data(df)
        print()
        
        # Step 5: Get initial statistics
        stats = get_initial_statistics(df, label_col)
        
        overall_timer.__exit__()
        
        log_message("Module 1 completed successfully!", level="SUCCESS")
        print()
        
        # Save checkpoint for future use
        save_module1_checkpoint(df, label_col, protocol_col, stats)
        print()
        
        return df, label_col, protocol_col, stats
    
    except Exception as e:
        log_message(f"Module 1 failed: {str(e)}", level="ERROR")
        raise


if __name__ == "__main__":
    # Test the module
    df, label_col, protocol_col, stats = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Label column: {label_col}")
    print(f"Protocol column: {protocol_col}")
