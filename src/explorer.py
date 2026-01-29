"""
MODULE 2: DATA EXPLORATION
Comprehensive exploratory data analysis with visualizations and detailed reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import (
    log_message, print_section_header, format_number, format_time,
    save_figure, write_text_report, calculate_imbalance_ratio, Timer
)
import config


def analyze_class_distribution(df, label_col):
    """
    Analyze class distribution in detail.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    label_col : str
        Label column name
        
    Returns:
    --------
    dict : Class distribution statistics
    """
    log_message("Analyzing class distribution...", level="STEP")
    
    # Count samples per class
    class_counts = df[label_col].value_counts().sort_values(ascending=False)
    total_samples = len(df)
    
    # Calculate percentages
    class_percentages = (class_counts / total_samples * 100)
    
    # Calculate imbalance ratios
    imbalance_ratios = calculate_imbalance_ratio(class_counts)
    
    # Identify majority and minority classes
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    # Calculate Gini coefficient (measure of imbalance)
    proportions = class_counts / total_samples
    gini = 1 - np.sum(proportions ** 2)
    
    log_message(f"Found {len(class_counts)} unique classes", level="INFO")
    log_message(f"Majority class: {majority_class} ({format_number(class_counts[majority_class])} samples)", 
               level="INFO")
    log_message(f"Minority class: {minority_class} ({format_number(class_counts[minority_class])} samples)", 
               level="INFO")
    log_message(f"Imbalance ratio: {format_number(imbalance_ratios[minority_class])}:1", level="INFO")
    log_message(f"Gini coefficient: {gini:.3f}", level="INFO")
    
    # Count classes requiring SMOTE (<1%)
    classes_needing_smote = [cls for cls, pct in class_percentages.items() if pct < 1.0]
    log_message(f"Classes requiring SMOTE (<1%): {len(classes_needing_smote)}", level="INFO")
    
    return {
        'counts': class_counts,
        'percentages': class_percentages,
        'imbalance_ratios': imbalance_ratios,
        'total_samples': total_samples,
        'n_classes': len(class_counts),
        'majority_class': majority_class,
        'minority_class': minority_class,
        'gini_coefficient': gini,
        'classes_needing_smote': classes_needing_smote
    }


def check_missing_data(df):
    """
    Identify and quantify missing values (NaN) with detailed distribution.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict : Missing data statistics
    """
    log_message("Checking for missing values (NaN)...", level="STEP")
    
    # Per-column analysis
    nan_counts = df.isnull().sum()
    nan_percentages = (nan_counts / len(df)) * 100
    
    # Overall statistics
    total_nan = nan_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    overall_nan_pct = (total_nan / total_cells) * 100
    
    # Count columns with NaN
    n_columns_with_nan = (nan_counts > 0).sum()
    pct_columns_with_nan = (n_columns_with_nan / len(df.columns)) * 100
    
    # Rows with any NaN
    rows_with_nan = df.isnull().any(axis=1).sum()
    
    # Distribution: count NaN per row
    nan_per_row = df.isnull().sum(axis=1)
    nan_distribution = nan_per_row.value_counts().sort_index()
    
    # Convert to percentage distribution
    nan_distribution_pct = (nan_distribution / len(df) * 100).to_dict()
    
    # Problematic columns
    problematic_cols = nan_counts[nan_percentages > 1.0].index.tolist()
    critical_cols = nan_counts[nan_percentages > 10.0].index.tolist()
    
    log_message(f"Total NaN cells: {format_number(total_nan)} ({overall_nan_pct:.3f}%)", level="INFO")
    log_message(f"Columns with NaN: {n_columns_with_nan}/{len(df.columns)} ({pct_columns_with_nan:.1f}%)", level="INFO")
    log_message(f"Rows with ANY NaN: {format_number(rows_with_nan)} ({rows_with_nan/len(df)*100:.2f}%)", 
               level="INFO")
    
    # Log distribution of NaN counts per row
    if len(nan_distribution) > 0 and nan_distribution.iloc[0] != len(df):  # If not all rows have 0 NaN
        log_message("NaN count distribution (breakdown of above):", level="INFO")
        for n_nans, count in list(nan_distribution.items())[:10]:  # Show top 10
            if n_nans > 0:  # Skip rows with 0 NaN
                pct = count / len(df) * 100
                log_message(f"  Rows with EXACTLY {n_nans} NaN(s): {format_number(count)} ({pct:.3f}%)", 
                          level="INFO", print_timestamp=False)
    
    if len(problematic_cols) > 0:
        log_message(f"Problematic columns (>1% NaN): {len(problematic_cols)}", level="WARNING")
    
    return {
        'nan_counts_per_column': nan_counts,
        'nan_percentages': nan_percentages,
        'total_nan_cells': total_nan,
        'total_cells': total_cells,
        'overall_nan_percentage': overall_nan_pct,
        'n_columns_with_nan': n_columns_with_nan,
        'pct_columns_with_nan': pct_columns_with_nan,
        'rows_with_nan': rows_with_nan,
        'nan_per_row_distribution': nan_distribution.to_dict(),
        'nan_per_row_distribution_pct': nan_distribution_pct,
        'problematic_columns': problematic_cols,
        'critical_columns': critical_cols
    }


def check_infinite_values(df):
    """
    Identify and quantify infinite values with detailed distribution.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict : Infinite value statistics
    """
    log_message("Checking for infinite values (Inf)...", level="STEP")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Per-column analysis
    inf_counts = pd.Series(0, index=df.columns)
    for col in numeric_cols:
        inf_counts[col] = np.isinf(df[col]).sum()
    
    total_inf = inf_counts.sum()
    affected_cols = inf_counts[inf_counts > 0].index.tolist()
    
    # Count columns with Inf
    n_columns_with_inf = len(affected_cols)
    pct_columns_with_inf = (n_columns_with_inf / len(df.columns)) * 100
    
    # Rows with any Inf
    rows_with_inf = 0
    inf_per_row_distribution = {0: len(df)}  # Default: all rows have 0 Inf
    inf_per_row_distribution_pct = {0: 100.0}
    
    if len(affected_cols) > 0:
        # Count Inf per row
        inf_per_row = df[affected_cols].apply(lambda x: np.isinf(x)).sum(axis=1)
        rows_with_inf = (inf_per_row > 0).sum()
        
        # Distribution of Inf counts per row
        inf_distribution = inf_per_row.value_counts().sort_index()
        inf_per_row_distribution = inf_distribution.to_dict()
        inf_per_row_distribution_pct = (inf_distribution / len(df) * 100).to_dict()
        
        # Log distribution
        log_message("Inf count distribution (breakdown of above):", level="INFO")
        for n_infs, count in list(inf_distribution.items())[:10]:  # Show top 10
            if n_infs > 0:  # Skip rows with 0 Inf
                pct = count / len(df) * 100
                log_message(f"  Rows with EXACTLY {n_infs} Inf(s): {format_number(count)} ({pct:.3f}%)", 
                          level="INFO", print_timestamp=False)
    
    log_message(f"Total Inf cells: {format_number(total_inf)}", level="INFO")
    log_message(f"Columns with Inf: {n_columns_with_inf}/{len(numeric_cols)} numeric ({pct_columns_with_inf:.1f}%)", level="INFO")
    log_message(f"Rows with Inf: {format_number(rows_with_inf)} ({rows_with_inf/len(df)*100:.2f}%)", 
               level="INFO")
    
    return {
        'inf_counts_per_column': inf_counts,
        'total_inf_cells': total_inf,
        'affected_columns': affected_cols,
        'n_columns_with_inf': n_columns_with_inf,
        'pct_columns_with_inf': pct_columns_with_inf,
        'rows_with_inf': rows_with_inf,
        'inf_per_row_distribution': inf_per_row_distribution,
        'inf_per_row_distribution_pct': inf_per_row_distribution_pct
    }


def count_duplicates(df):
    """
    Identify duplicate rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict : Duplicate statistics
    """
    log_message("Counting duplicate rows...", level="STEP")
    
    n_duplicates = df.duplicated().sum()
    dup_percentage = (n_duplicates / len(df)) * 100
    
    log_message(f"Duplicate rows: {format_number(n_duplicates)} ({dup_percentage:.3f}%)", level="INFO")
    
    return {
        'n_duplicates': n_duplicates,
        'duplicate_percentage': dup_percentage
    }


def analyze_column_availability(df):
    """
    Analyze data availability (non-null percentage) for each column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict : Column availability statistics
    """
    log_message("Analyzing column-wise data availability...", level="STEP")
    
    total_rows = len(df)
    column_stats = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        non_null_pct = (non_null_count / total_rows) * 100
        null_pct = (null_count / total_rows) * 100
        
        column_stats.append({
            'column': col,
            'non_null_count': non_null_count,
            'non_null_percentage': non_null_pct,
            'null_count': null_count,
            'null_percentage': null_pct,
            'dtype': str(df[col].dtype)
        })
    
    # Sort by null percentage descending to show problematic columns first
    column_stats.sort(key=lambda x: x['null_percentage'], reverse=True)
    
    # Find columns with high missing rate
    high_missing_cols = [c for c in column_stats if c['null_percentage'] > 10]
    
    log_message(f"Analyzed {len(column_stats)} columns", level="INFO")
    if high_missing_cols:
        log_message(f"WARNING: {len(high_missing_cols)} columns have >10% missing data", level="WARNING")
    else:
        log_message("All columns have <10% missing data", level="INFO")
    
    return {
        'column_stats': column_stats,
        'high_missing_columns': high_missing_cols,
        'total_columns': len(column_stats)
    }


def calculate_correlations(df, label_col, top_n=None):
    """
    Calculate feature correlations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    label_col : str
        Label column name
    top_n : int, optional
        Number of top features to analyze
        
    Returns:
    --------
    dict : Correlation statistics
    """
    if top_n is None:
        top_n = config.TOP_N_FEATURES_CORRELATION
    
    log_message(f"Calculating feature correlations (top {top_n} features)...", level="STEP")
    
    # Select numeric columns (exclude label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    log_message(f"Computing correlation matrix for {len(numeric_cols)} features...", level="INFO")
    
    # Calculate full correlation matrix
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    # Select top N features by variance
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    # Extract top correlation submatrix
    top_corr_matrix = corr_matrix.loc[top_features, top_features]
    
    # Find highly correlated pairs
    high_corr_pairs = []
    threshold = config.HIGH_CORRELATION_THRESHOLD
    
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            corr_val = top_corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append((top_features[i], top_features[j], corr_val))
    
    log_message(f"Identified top {top_n} features by variance", level="INFO")
    log_message(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold})", level="INFO")
    
    return {
        'correlation_matrix': corr_matrix,
        'top_features': top_features,
        'top_correlation_matrix': top_corr_matrix,
        'highly_correlated_pairs': high_corr_pairs,
        'n_numeric_features': len(numeric_cols)
    }


def generate_descriptive_statistics(df, label_col):
    """
    Generate descriptive statistics for numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    label_col : str
        Label column name
        
    Returns:
    --------
    pandas.DataFrame : Descriptive statistics
    """
    log_message("Generating descriptive statistics...", level="STEP")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    # Calculate statistics
    desc_stats = df[numeric_cols].describe().T
    
    # Add skewness and kurtosis
    desc_stats['skewness'] = df[numeric_cols].skew()
    desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
    
    log_message(f"Calculated statistics for {len(numeric_cols)} features", level="SUCCESS")
    
    return desc_stats


def analyze_data_types_memory(df):
    """
    Analyze data types and memory usage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict : Data type and memory statistics
    """
    log_message("Analyzing data types and memory usage...", level="STEP")
    
    dtype_counts = df.dtypes.value_counts()
    
    # Calculate memory per dtype
    memory_per_dtype = {}
    for dtype in dtype_counts.index:
        cols_of_dtype = df.select_dtypes(include=[dtype]).columns
        memory_bytes = df[cols_of_dtype].memory_usage(deep=True).sum()
        memory_per_dtype[str(dtype)] = memory_bytes / (1024 ** 3)  # GB
    
    total_memory_gb = sum(memory_per_dtype.values())
    memory_per_row_kb = (total_memory_gb * 1024 ** 2) / len(df)  # KB per row
    
    # Top memory-consuming columns
    col_memory = df.memory_usage(deep=True).sort_values(ascending=False)
    top_memory_cols = [(col, mem / (1024 ** 2)) for col, mem in col_memory.head(10).items()]
    
    log_message(f"Total memory: {total_memory_gb:.2f} GB", level="INFO")
    log_message(f"Memory per row: {memory_per_row_kb:.2f} KB", level="INFO")
    
    return {
        'dtype_counts': dtype_counts,
        'memory_per_dtype': memory_per_dtype,
        'total_memory_gb': total_memory_gb,
        'memory_per_row_kb': memory_per_row_kb,
        'top_memory_columns': top_memory_cols
    }


def create_class_distribution_chart(class_stats, output_dir):
    """
    Create vertical bar chart of class distribution.
    
    Parameters:
    -----------
    class_stats : dict
        Class distribution statistics
    output_dir : str
        Output directory for plots
    """
    log_message("Creating class distribution chart...", level="INFO")
    
    counts = class_stats['counts']
    percentages = class_stats['percentages']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create VERTICAL bar chart (labels at bottom, counts on left)
    x_pos = np.arange(len(counts))
    bars = ax.bar(x_pos, counts.values, color=plt.cm.tab10(np.linspace(0, 1, len(counts))))
    
    # Add labels at BOTTOM (x-axis)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in CICIDS2018 Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count annotations ON TOP of bars (no percentage)
    for i, count in enumerate(counts.values):
        ax.text(i, count, f'{format_number(count)}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'class_distribution.png')
    save_figure(fig, filepath)


def create_imbalance_log_chart(class_stats, output_dir):
    """
    Create log-scale bar chart to visualize extreme imbalance.
    
    Parameters:
    -----------
    class_stats : dict
        Class distribution statistics
    output_dir : str
        Output directory
    """
    log_message("Creating class imbalance log-scale chart...", level="INFO")
    
    counts = class_stats['counts']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart with log scale
    x_pos = np.arange(len(counts))
    bars = ax.bar(x_pos, counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(counts))))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    ax.set_ylabel('Sample Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Class Imbalance (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count annotations
    for i, count in enumerate(counts.values):
        ax.text(i, count, f'{format_number(count)}', ha='center', va='bottom', fontsize=9)
    
    # Add median reference line
    median_count = counts.median()
    ax.axhline(y=median_count, color='r', linestyle='--', alpha=0.5, label=f'Median: {format_number(median_count)}')
    ax.legend()
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'class_imbalance_log_scale.png')
    save_figure(fig, filepath)


def create_correlation_heatmap(corr_stats, output_dir):
    """
    Create HIGH RESOLUTION correlation heatmap for top features.
    
    Parameters:
    -----------
    corr_stats : dict
        Correlation statistics
    output_dir : str
        Output directory
    """
    log_message("Creating high-resolution correlation heatmap...", level="INFO")
    
    corr_matrix = corr_stats['top_correlation_matrix']
    
    # HIGH RESOLUTION figure with DPI 150
    fig, ax = plt.subplots(figsize=(18, 16), dpi=150)
    
    # Create heatmap with larger annotations
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax,
                annot_kws={'fontsize': 8})
    
    ax.set_title(f'Feature Correlation Matrix (Top {len(corr_matrix)} Features by Variance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'correlation_heatmap.png')
    # Save with high DPI
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_missing_data_chart(missing_stats, inf_stats, output_dir):
    """
    Create visualization of missing and infinite values.
    
    Parameters:
    -----------
    missing_stats : dict
        Missing data statistics
    inf_stats : dict
        Infinite value statistics
    output_dir : str
        Output directory
    """
    log_message("Creating missing data summary chart...", level="INFO")
    
    nan_counts = missing_stats['nan_counts_per_column']
    inf_counts = inf_stats['inf_counts_per_column']
    
    # Get columns with issues
    cols_with_nan = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)
    cols_with_inf = inf_counts[inf_counts > 0].sort_values(ascending=False).head(10)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # NaN plot
    if len(cols_with_nan) > 0:
        axes[0].barh(range(len(cols_with_nan)), cols_with_nan.values, color='orange')
        axes[0].set_yticks(range(len(cols_with_nan)))
        axes[0].set_yticklabels(cols_with_nan.index)
        axes[0].set_xlabel('Count', fontweight='bold')
        axes[0].set_title('Top 10 Columns with Missing Values (NaN)', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No missing values found', ha='center', va='center', 
                    fontsize=14, transform=axes[0].transAxes)
        axes[0].axis('off')
    
    # Inf plot
    if len(cols_with_inf) > 0:
        axes[1].barh(range(len(cols_with_inf)), cols_with_inf.values, color='red')
        axes[1].set_yticks(range(len(cols_with_inf)))
        axes[1].set_yticklabels(cols_with_inf.index)
        axes[1].set_xlabel('Count', fontweight='bold')
        axes[1].set_title('Top 10 Columns with Infinite Values (Inf)', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No infinite values found', ha='center', va='center', 
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'missing_data_summary.png')
    save_figure(fig, filepath)


def create_memory_usage_chart(memory_stats, output_dir):
    """
    Create pie chart with side legend table showing memory usage by data type.
    
    Parameters:
    -----------
    memory_stats : dict
        Memory usage statistics
    output_dir : str
        Output directory
    """
    log_message("Creating memory usage chart with legend table...", level="INFO")
    
    memory_per_dtype = memory_stats['memory_per_dtype']
    
    # Create figure with two subplots: pie chart and table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Prepare data
    dtypes = list(memory_per_dtype.keys())
    sizes = list(memory_per_dtype.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    
    # Explode largest slice
    explode = [0.05 if size == max(sizes) else 0 for size in sizes]
    
    # Create pie chart WITHOUT labels (clean look, no white lines)
    ax1.pie(sizes, colors=colors, explode=explode,
           startangle=90, wedgeprops={'linewidth': 0, 'edgecolor': 'none'})
    ax1.set_title('Memory Usage by Data Type', fontsize=14, fontweight='bold', pad=20)
    
    # Create side legend TABLE
    ax2.axis('off')
    table_data = []
    for dtype, mem in memory_per_dtype.items():
        pct = mem / memory_stats['total_memory_gb'] * 100
        table_data.append([dtype, f'{mem:.2f} GB', f'{pct:.1f}%'])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Data Type', 'Memory', 'Percentage'],
        cellLoc='left',
        loc='center',
        colWidths=[0.35, 0.3, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color table rows to match pie slices
    for i, color in enumerate(colors):
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 1)].set_facecolor(color)
        table[(i+1, 2)].set_facecolor(color)
    
    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'data_types_memory.png')
    save_figure(fig, filepath)


def analyze_label_consolidation_impact(df, label_col, output_dir):
    """
    Analyze the impact of label consolidation (before and after mapping).
    Shows imbalance ratios, percentages, and prepares visualization data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset with original labels
    label_col : str
        Label column name
    output_dir : str
        Output directory
        
    Returns:
    --------
    dict : Consolidation impact statistics
    """
    log_message("Analyzing label consolidation impact...", level="STEP")
    
    # BEFORE consolidation (original labels)
    original_counts = df[label_col].value_counts().sort_values(ascending=False)
    original_percentages = (original_counts / len(df) * 100)
    original_imbalance = calculate_imbalance_ratio(original_counts)
    
    # Calculate Gini before consolidation
    original_proportions = original_counts / len(df)
    original_gini = 1 - np.sum(original_proportions ** 2)
    
    log_message(f"BEFORE Consolidation:", level="INFO")
    log_message(f"  Classes: {len(original_counts)}", level="INFO")
    log_message(f"  Gini Coefficient: {original_gini:.4f}", level="INFO")
    log_message(f"  Majority class: {original_counts.index[0]} ({original_counts.iloc[0]:,} samples, {original_percentages.iloc[0]:.2f}%)", 
               level="INFO")
    log_message(f"  Minority class: {original_counts.index[-1]} ({original_counts.iloc[-1]:,} samples, {original_percentages.iloc[-1]:.2f}%)", 
               level="INFO")
    log_message(f"  Imbalance ratio: {original_imbalance[original_counts.index[-1]]:.2f}:1", level="INFO")
    print()
    
    # AFTER consolidation (consolidated labels)
    df_consolidated = df.copy()
    df_consolidated[label_col] = df_consolidated[label_col].map(config.LABEL_MAPPING).fillna(df_consolidated[label_col])
    
    # Remove dropped classes
    df_consolidated = df_consolidated[df_consolidated[label_col] != '__DROP__'].copy()
    
    consolidated_counts = df_consolidated[label_col].value_counts().sort_values(ascending=False)
    consolidated_percentages = (consolidated_counts / len(df_consolidated) * 100)
    consolidated_imbalance = calculate_imbalance_ratio(consolidated_counts)
    
    # Calculate Gini after consolidation
    consolidated_proportions = consolidated_counts / len(df_consolidated)
    consolidated_gini = 1 - np.sum(consolidated_proportions ** 2)
    
    log_message(f"AFTER Consolidation:", level="INFO")
    log_message(f"  Classes: {len(consolidated_counts)}", level="INFO")
    log_message(f"  Gini Coefficient: {consolidated_gini:.4f}", level="INFO")
    log_message(f"  Majority class: {consolidated_counts.index[0]} ({consolidated_counts.iloc[0]:,} samples, {consolidated_percentages.iloc[0]:.2f}%)", 
               level="INFO")
    log_message(f"  Minority class: {consolidated_counts.index[-1]} ({consolidated_counts.iloc[-1]:,} samples, {consolidated_percentages.iloc[-1]:.2f}%)", 
               level="INFO")
    log_message(f"  Imbalance ratio: {consolidated_imbalance[consolidated_counts.index[-1]]:.2f}:1", level="INFO")
    print()
    
    # Calculate improvements
    gini_improvement = (original_gini - consolidated_gini) / original_gini * 100 if original_gini > 0 else 0
    classes_removed = len(original_counts) - len(consolidated_counts)
    rows_removed = len(df) - len(df_consolidated)
    
    log_message(f"Consolidation Impact:", level="INFO")
    log_message(f"  Gini Improvement: {gini_improvement:+.2f}%", level="INFO")
    log_message(f"  Classes Reduced: {classes_removed} ({len(original_counts)} → {len(consolidated_counts)})", level="INFO")
    log_message(f"  Rows Removed (SQL Injection, etc): {format_number(rows_removed)} ({rows_removed/len(df)*100:.2f}%)", level="INFO")
    print()
    
    # Create visualization
    create_consolidation_imbalance_chart(
        original_counts, original_percentages, original_gini,
        consolidated_counts, consolidated_percentages, consolidated_gini,
        output_dir
    )
    
    return {
        'original_counts': original_counts,
        'original_percentages': original_percentages,
        'original_imbalance_ratios': original_imbalance,
        'original_gini': original_gini,
        'consolidated_counts': consolidated_counts,
        'consolidated_percentages': consolidated_percentages,
        'consolidated_imbalance_ratios': consolidated_imbalance,
        'consolidated_gini': consolidated_gini,
        'gini_improvement_pct': gini_improvement,
        'classes_removed': classes_removed,
        'rows_removed': rows_removed
    }


def create_consolidation_imbalance_chart(original_counts, original_pcts, original_gini,
                                         consolidated_counts, consolidated_pcts, consolidated_gini,
                                         output_dir):
    """
    Create 4 separate comparison charts for label consolidation impact.
    
    Creates:
    1. Bar chart BEFORE consolidation
    2. Bar chart AFTER consolidation  
    3. Pie chart BEFORE with side table
    4. Pie chart AFTER with side table
    
    Parameters:
    -----------
    original_counts : Series
        Original class counts
    original_pcts : Series
        Original percentages
    original_gini : float
        Original Gini coefficient
    consolidated_counts : Series
        Consolidated class counts
    consolidated_pcts : Series
        Consolidated percentages
    consolidated_gini : float
        Consolidated Gini coefficient
    output_dir : str
        Output directory
    """
    try:
        # 1. Bar chart BEFORE consolidation
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        colors_before = plt.cm.Set2(np.linspace(0, 1, len(original_counts)))
        bars1 = ax1.bar(range(len(original_counts)), original_counts.values, color=colors_before, edgecolor='black', linewidth=2)
        ax1.set_xlabel('Class Labels', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        ax1.set_title(f'BEFORE Consolidation\n({len(original_counts)} classes, Gini={original_gini:.4f})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(range(len(original_counts)))
        ax1.set_xticklabels(original_counts.index, rotation=45, ha='right', fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars1, original_counts.values, original_pcts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.2f}%\n({format_number(int(count))})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig1.tight_layout()
        save_figure(fig1, os.path.join(output_dir, 'consolidation_bar_before.png'), dpi=config.FIGURE_DPI)
        plt.close()
        log_message(f"✓ Saved: consolidation_bar_before.png", level="SUCCESS")
        
        # 2. Bar chart AFTER consolidation
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        colors_after = plt.cm.Set3(np.linspace(0, 1, len(consolidated_counts)))
        bars2 = ax2.bar(range(len(consolidated_counts)), consolidated_counts.values, color=colors_after, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Class Labels', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        ax2.set_title(f'AFTER Consolidation\n({len(consolidated_counts)} classes, Gini={consolidated_gini:.4f})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(range(len(consolidated_counts)))
        ax2.set_xticklabels(consolidated_counts.index, rotation=45, ha='right', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, count, pct in zip(bars2, consolidated_counts.values, consolidated_pcts.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.2f}%\n({format_number(int(count))})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig2.tight_layout()
        save_figure(fig2, os.path.join(output_dir, 'consolidation_bar_after.png'), dpi=config.FIGURE_DPI)
        plt.close()
        log_message(f"✓ Saved: consolidation_bar_after.png", level="SUCCESS")
        
        # 3. Pie chart BEFORE with side table
        fig3 = plt.figure(figsize=(14, 7))
        ax3_pie = fig3.add_subplot(121)
        ax3_table = fig3.add_subplot(122)
        
        # Pie chart (no labels on pie)
        pie_result = ax3_pie.pie(original_pcts.values, colors=colors_before, startangle=90,
                                 textprops={'fontsize': 11})
        # Handle both matplotlib versions: some return (wedges, texts, autotexts), others just (wedges, texts)
        if len(pie_result) == 3:
            wedges, texts, autotexts = pie_result
        else:
            wedges, texts = pie_result
            autotexts = []
        
        ax3_pie.set_title(f'BEFORE Consolidation\nGini: {original_gini:.4f}', fontsize=13, fontweight='bold')
        
        # Remove percentage labels from pie
        for autotext in autotexts:
            autotext.set_visible(False)
        for text in texts:
            text.set_visible(False)
        
        # Side table with class names and percentages
        ax3_table.axis('off')
        table_data = []
        table_data.append(['Class Name', 'Count', 'Percentage'])
        for cls, count, pct in zip(original_counts.index, original_counts.values, original_pcts.values):
            table_data.append([str(cls), format_number(int(count)), f'{pct:.2f}%'])
        
        table3 = ax3_table.table(cellText=table_data, cellLoc='left', loc='center',
                                colWidths=[0.45, 0.25, 0.25])
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 2.2)
        
        # Style header
        for j in range(3):
            table3[(0, j)].set_facecolor('#4CAF50')
            table3[(0, j)].set_text_props(weight='bold', color='white')
        
        # Color code table rows to match pie
        for i in range(1, len(table_data)):
            for j in range(3):
                table3[(i, j)].set_facecolor(colors_before[i-1])
        
        fig3.suptitle('Class Distribution - BEFORE Consolidation', fontsize=14, fontweight='bold', y=0.98)
        fig3.tight_layout()
        save_figure(fig3, os.path.join(output_dir, 'consolidation_pie_before.png'), dpi=config.FIGURE_DPI)
        plt.close()
        log_message(f"✓ Saved: consolidation_pie_before.png", level="SUCCESS")
        
        # 4. Pie chart AFTER with side table
        fig4 = plt.figure(figsize=(12, 7))
        ax4_pie = fig4.add_subplot(121)
        ax4_table = fig4.add_subplot(122)
        
        # Pie chart (no labels on pie)
        pie_result = ax4_pie.pie(consolidated_pcts.values, colors=colors_after, startangle=90,
                                 textprops={'fontsize': 11})
        # Handle both matplotlib versions: some return (wedges, texts, autotexts), others just (wedges, texts)
        if len(pie_result) == 3:
            wedges, texts, autotexts = pie_result
        else:
            wedges, texts = pie_result
            autotexts = []
        
        ax4_pie.set_title(f'AFTER Consolidation\nGini: {consolidated_gini:.4f}', fontsize=13, fontweight='bold')
        
        # Remove percentage labels from pie
        for autotext in autotexts:
            autotext.set_visible(False)
        for text in texts:
            text.set_visible(False)
        
        # Side table with class names and percentages
        ax4_table.axis('off')
        table_data = []
        table_data.append(['Class Name', 'Count', 'Percentage'])
        for cls, count, pct in zip(consolidated_counts.index, consolidated_counts.values, consolidated_pcts.values):
            table_data.append([str(cls), format_number(int(count)), f'{pct:.2f}%'])
        
        table4 = ax4_table.table(cellText=table_data, cellLoc='left', loc='center',
                                colWidths=[0.45, 0.25, 0.25])
        table4.auto_set_font_size(False)
        table4.set_fontsize(10)
        table4.scale(1, 2.2)
        
        # Style header
        for j in range(3):
            table4[(0, j)].set_facecolor('#2196F3')
            table4[(0, j)].set_text_props(weight='bold', color='white')
        
        # Color code table rows to match pie
        for i in range(1, len(table_data)):
            for j in range(3):
                table4[(i, j)].set_facecolor(colors_after[i-1])
        
        fig4.suptitle('Class Distribution - AFTER Consolidation', fontsize=14, fontweight='bold', y=0.98)
        fig4.tight_layout()
        save_figure(fig4, os.path.join(output_dir, 'consolidation_pie_after.png'), dpi=config.FIGURE_DPI)
        plt.close()
        log_message(f"✓ Saved: consolidation_pie_after.png", level="SUCCESS")
        
    except Exception as e:
        log_message(f"Failed to create consolidation charts: {str(e)}", level="WARNING")


def generate_exploration_report(all_stats, df, label_col, output_dir):
    """
    Generate comprehensive text report.
    
    Parameters:
    -----------
    all_stats : dict
        All exploration statistics
    df : pandas.DataFrame
        Dataset
    label_col : str
        Label column name
    output_dir : str
        Output directory
    """
    log_message("Generating comprehensive text report...", level="STEP")
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append(" " * 20 + "DATA EXPLORATION REPORT")
    report_lines.append(" " * 25 + "CICIDS2018 Dataset")
    report_lines.append(f" " * 20 + f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Dataset Overview
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("   " + "-" * 16)
    report_lines.append(f"   Total Rows:          {format_number(len(df))}")
    report_lines.append(f"   Total Columns:       {len(df.columns)}")
    report_lines.append(f"   Memory Usage:        {all_stats['memory']['total_memory_gb']:.2f} GB")
    report_lines.append("")
    
    report_lines.append("   Data Types Distribution:")
    for dtype, count in all_stats['memory']['dtype_counts'].items():
        pct = count / len(df.columns) * 100
        report_lines.append(f"     - {dtype}: {count} columns ({pct:.1f}%)")
    report_lines.append("")
    
    report_lines.append(f"   Numeric Columns:     {all_stats['correlation']['n_numeric_features']}")
    report_lines.append(f"   Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}")
    report_lines.append("")
    
    # 2. Class Distribution
    report_lines.append("2. CLASS DISTRIBUTION")
    report_lines.append("   " + "-" * 18)
    class_stats = all_stats['class_distribution']
    report_lines.append(f"   Total Classes: {class_stats['n_classes']}")
    report_lines.append("")
    
    report_lines.append("   Class Name                    Count         Percentage    Imbalance Ratio")
    report_lines.append("   " + "-" * 73)
    
    for cls in class_stats['counts'].index:
        count = class_stats['counts'][cls]
        pct = class_stats['percentages'][cls]
        ratio = class_stats['imbalance_ratios'][cls]
        report_lines.append(f"   {cls:25} {count:13,}    {pct:8.2f}%    {ratio:12.2f}")
    
    report_lines.append("")
    report_lines.append(f"   Imbalance Severity: EXTREME")
    report_lines.append(f"   Gini Coefficient: {class_stats['gini_coefficient']:.3f}")
    report_lines.append(f"   Classes requiring SMOTE (<1%): {len(class_stats['classes_needing_smote'])}")
    report_lines.append("")
    
    # 2.5 Label Consolidation Impact
    if 'consolidation' in all_stats:
        report_lines.append("2.5 LABEL CONSOLIDATION IMPACT")
        report_lines.append("   " + "-" * 30)
        report_lines.append("")
        
        cons = all_stats['consolidation']
        
        report_lines.append("   BEFORE Consolidation:")
        report_lines.append(f"     Classes: {len(cons['original_counts'])}")
        report_lines.append(f"     Gini Coefficient: {cons['original_gini']:.4f}")
        report_lines.append("")
        report_lines.append("     Original Class Distribution:")
        report_lines.append("     Class Name                    Count         Percentage")
        report_lines.append("     " + "-" * 65)
        
        for cls in cons['original_counts'].index:
            count = cons['original_counts'][cls]
            pct = cons['original_percentages'][cls]
            report_lines.append(f"     {cls:25} {count:13,}    {pct:8.2f}%")
        
        report_lines.append("")
        report_lines.append("   AFTER Consolidation (SQL Injection dropped):")
        report_lines.append(f"     Classes: {len(cons['consolidated_counts'])}")
        report_lines.append(f"     Gini Coefficient: {cons['consolidated_gini']:.4f}")
        report_lines.append("")
        report_lines.append("     Consolidated Class Distribution:")
        report_lines.append("     Class Name                    Count         Percentage    Imbalance Ratio")
        report_lines.append("     " + "-" * 73)
        
        for cls in cons['consolidated_counts'].index:
            count = cons['consolidated_counts'][cls]
            pct = cons['consolidated_percentages'][cls]
            ratio = cons['consolidated_imbalance_ratios'][cls]
            report_lines.append(f"     {cls:25} {count:13,}    {pct:8.2f}%    {ratio:12.2f}")
        
        report_lines.append("")
        report_lines.append("   Consolidation Summary:")
        report_lines.append(f"     Classes Reduced: {cons['classes_removed']} ({len(cons['original_counts'])} → {len(cons['consolidated_counts'])})")
        report_lines.append(f"     Rows Removed (SQL Injection, etc): {format_number(cons['rows_removed'])} ({cons['rows_removed']/len(df)*100:.2f}%)")
        report_lines.append(f"     Gini Improvement: {cons['gini_improvement_pct']:+.2f}%")
        report_lines.append("")
    
    # 3. Data Quality Assessment
    report_lines.append("3. DATA QUALITY ASSESSMENT")
    report_lines.append("   " + "-" * 24)
    report_lines.append("")
    
    # Missing values
    missing_stats = all_stats['missing_data']
    report_lines.append("   3.1 Missing Values (NaN)")
    report_lines.append(f"       Total NaN cells: {format_number(missing_stats['total_nan_cells'])}")
    report_lines.append(f"       Percentage of dataset: {missing_stats['overall_nan_percentage']:.3f}%")
    report_lines.append(f"       Columns with NaN: {missing_stats['n_columns_with_nan']}/{len(df.columns)} "
                       f"({missing_stats['pct_columns_with_nan']:.1f}%)")
    report_lines.append(f"       Rows with NaN: {format_number(missing_stats['rows_with_nan'])} "
                       f"({missing_stats['rows_with_nan']/len(df)*100:.2f}%)")
    report_lines.append("")
    
    # NaN distribution per row
    if 'nan_per_row_distribution_pct' in missing_stats:
        nan_dist = missing_stats['nan_per_row_distribution']
        report_lines.append("       NaN Distribution (rows by NaN count):")
        for n_nans in sorted(nan_dist.keys())[:15]:  # Show top 15
            if n_nans > 0:
                count = nan_dist[n_nans]
                pct = missing_stats['nan_per_row_distribution_pct'][n_nans]
                report_lines.append(f"         {n_nans} NaN(s): {format_number(count)} rows ({pct:.3f}%)")
        report_lines.append("")
    
    # Infinite values
    inf_stats = all_stats['inf_values']
    report_lines.append("   3.2 Infinite Values (Inf/-Inf)")
    report_lines.append(f"       Total Inf cells: {format_number(inf_stats['total_inf_cells'])}")
    report_lines.append(f"       Columns with Inf: {inf_stats['n_columns_with_inf']}/{all_stats['correlation']['n_numeric_features']} numeric "
                       f"({inf_stats['pct_columns_with_inf']:.1f}%)")
    report_lines.append(f"       Rows with Inf: {format_number(inf_stats['rows_with_inf'])} "
                       f"({inf_stats['rows_with_inf']/len(df)*100:.2f}%)")
    report_lines.append("")
    
    # Inf distribution per row
    if 'inf_per_row_distribution_pct' in inf_stats:
        inf_dist = inf_stats['inf_per_row_distribution']
        report_lines.append("       Inf Distribution (rows by Inf count):")
        for n_infs in sorted(inf_dist.keys())[:15]:  # Show top 15
            if n_infs > 0:
                count = inf_dist[n_infs]
                pct = inf_stats['inf_per_row_distribution_pct'][n_infs]
                report_lines.append(f"         {n_infs} Inf(s): {format_number(count)} rows ({pct:.3f}%)")
        report_lines.append("")
    
    # Duplicates
    dup_stats = all_stats['duplicates']
    report_lines.append("   3.3 Duplicate Rows")
    report_lines.append(f"       Duplicate count: {format_number(dup_stats['n_duplicates'])}")
    report_lines.append(f"       Percentage: {dup_stats['duplicate_percentage']:.3f}%")
    report_lines.append("")
    
    # Total rows to remove
    total_to_remove = missing_stats['rows_with_nan'] + inf_stats['rows_with_inf'] + dup_stats['n_duplicates']
    pct_to_remove = total_to_remove / len(df) * 100
    report_lines.append(f"   Total rows to be removed: ~{format_number(total_to_remove)} ({pct_to_remove:.2f}%)")
    report_lines.append(f"   Expected clean dataset size: ~{format_number(len(df) - total_to_remove)} rows")
    report_lines.append("")
    
    # 4. Feature Correlation
    report_lines.append("4. FEATURE CORRELATION ANALYSIS")
    report_lines.append("   " + "-" * 28)
    corr_stats = all_stats['correlation']
    report_lines.append(f"   Total numeric features: {corr_stats['n_numeric_features']}")
    report_lines.append(f"   Top features analyzed (by variance): {len(corr_stats['top_features'])}")
    report_lines.append("")
    report_lines.append(f"   Top {len(corr_stats['top_features'])} Most Variable Features:")
    for i, feat in enumerate(corr_stats['top_features'], 1):
        report_lines.append(f"   {i:2d}. {feat}")
    report_lines.append("")
    
    # Add correlation matrix details
    report_lines.append("   Correlation Matrix Values (Top 20 Features):")
    report_lines.append("   Range from -1 (perfect negative) to +1 (perfect positive)")
    report_lines.append("")
    
    # Extract correlation pairs from top_correlation_matrix
    top_corr_matrix = corr_stats.get('top_correlation_matrix')
    if top_corr_matrix is not None:
        corr_pairs = []
        for i in range(len(top_corr_matrix.columns)):
            for j in range(i+1, len(top_corr_matrix.columns)):
                corr_pairs.append({
                    'feat1': top_corr_matrix.columns[i],
                    'feat2': top_corr_matrix.columns[j],
                    'corr': top_corr_matrix.iloc[i, j]
                })
        
        # Top 10 positive correlations
        positive_corr = sorted([p for p in corr_pairs if p['corr'] > 0], 
                              key=lambda x: x['corr'], reverse=True)[:10]
        report_lines.append("   Strongest Positive Correlations:")
        for i, pair in enumerate(positive_corr, 1):
            report_lines.append(f"   {i:2d}. {pair['feat1'][:30]:<30} <-> {pair['feat2'][:30]:<30} {pair['corr']:>7.3f}")
        report_lines.append("")
        
        # Top 10 negative correlations
        negative_corr = sorted([p for p in corr_pairs if p['corr'] < 0], 
                              key=lambda x: x['corr'])[:10]
        if negative_corr:
            report_lines.append("   Strongest Negative Correlations:")
            for i, pair in enumerate(negative_corr, 1):
                report_lines.append(f"   {i:2d}. {pair['feat1'][:30]:<30} <-> {pair['feat2'][:30]:<30} {pair['corr']:>7.3f}")
            report_lines.append("")
    
    if len(corr_stats['highly_correlated_pairs']) > 0:
        report_lines.append(f"   Highly Correlated Feature Pairs (|r| > {config.HIGH_CORRELATION_THRESHOLD}):")
        report_lines.append("   " + "-" * 75)
        for feat1, feat2, corr_val in corr_stats['highly_correlated_pairs'][:10]:
            report_lines.append(f"   {feat1[:30]:30} <-> {feat2[:30]:30} : {corr_val:6.3f}")
    report_lines.append("")
    
    # 5. Column-wise Data Availability
    report_lines.append("5. COLUMN-WISE DATA AVAILABILITY")
    report_lines.append("   " + "-" * 33)
    col_avail = all_stats.get('column_availability', {})
    if col_avail:
        report_lines.append("")
        report_lines.append("   Data Availability for All Columns:")
        report_lines.append("   " + "-" * 90)
        report_lines.append(f"   {'Column Name':<40} {'Non-Null %':>12} {'Null %':>12} {'Data Type':<15}")
        report_lines.append("   " + "-" * 90)
        
        for col_stat in col_avail.get('column_stats', []):
            report_lines.append(
                f"   {col_stat['column']:<40} "
                f"{col_stat['non_null_percentage']:>11.2f}% "
                f"{col_stat['null_percentage']:>11.2f}% "
                f"{col_stat['dtype']:<15}"
            )
        
        high_missing = col_avail.get('high_missing_columns', [])
        if high_missing:
            report_lines.append("")
            report_lines.append(f"   ⚠  WARNING: {len(high_missing)} columns have >10% missing data:")
            for col_stat in high_missing:
                report_lines.append(
                    f"      - {col_stat['column']}: {col_stat['null_percentage']:.2f}% missing"
                )
            report_lines.append("")
            report_lines.append("   RECOMMENDATION: According to spec, REMOVE rows with NaN during preprocessing")
        else:
            report_lines.append("")
            report_lines.append("   ✓ All columns have <10% missing data - good quality dataset")
    report_lines.append("")
    
    # 6. Memory Usage
    report_lines.append("6. MEMORY USAGE ANALYSIS")
    report_lines.append("   " + "-" * 20)
    mem_stats = all_stats['memory']
    report_lines.append(f"   Total Memory: {mem_stats['total_memory_gb']:.2f} GB")
    report_lines.append(f"   Memory per row: {mem_stats['memory_per_row_kb']:.2f} KB")
    report_lines.append("")
    
    report_lines.append("   Memory by Data Type:")
    for dtype, mem_gb in mem_stats['memory_per_dtype'].items():
        pct = mem_gb / mem_stats['total_memory_gb'] * 100
        report_lines.append(f"     {dtype}: {mem_gb:.2f} GB ({pct:.1f}%)")
    report_lines.append("")
    
    # 7. Preprocessing Recommendations (based on spec)
    report_lines.append("7. PREPROCESSING RECOMMENDATIONS")
    report_lines.append("   " + "-" * 33)
    report_lines.append("")
    report_lines.append("   Based on exploration findings and CICIDS2018 specification:")
    report_lines.append("")
    report_lines.append("   1. Data Cleaning:")
    missing_stats = all_stats['missing_data']
    inf_stats = all_stats['inf_values']
    dup_stats = all_stats['duplicates']
    total_to_remove = missing_stats['rows_with_nan'] + inf_stats['rows_with_inf'] + dup_stats['n_duplicates']
    pct_to_remove = total_to_remove / len(df) * 100
    report_lines.append(f"      ✓ REMOVE rows with NaN: {format_number(missing_stats['rows_with_nan'])} rows ({missing_stats['rows_with_nan']/len(df)*100:.2f}%)")
    report_lines.append(f"      ✓ REMOVE rows with Inf: {format_number(inf_stats['rows_with_inf'])} rows ({inf_stats['rows_with_inf']/len(df)*100:.2f}%)")
    report_lines.append(f"      ✓ REMOVE duplicate rows: {format_number(dup_stats['n_duplicates'])} rows ({dup_stats['duplicate_percentage']:.3f}%)")
    report_lines.append(f"      Expected loss: ~{format_number(total_to_remove)} rows ({pct_to_remove:.2f}%) - ACCEPTABLE")
    report_lines.append(f"      Expected clean dataset: ~{format_number(len(df) - total_to_remove)} rows")
    report_lines.append("")
    report_lines.append("   2. Label Consolidation:")
    report_lines.append("      ✓ Merge DDoS subcategories → \"DDoS\"")
    report_lines.append("      ✓ Merge DoS subcategories → \"DoS\"")
    report_lines.append("      ✓ Merge Brute Force variants → \"Brute Force\"")
    report_lines.append("      Result: 15 classes → 8 classes")
    report_lines.append("")
    report_lines.append("   3. Feature Scaling:")
    report_lines.append("      ✓ Use StandardScaler (mean=0, std=1)")
    report_lines.append("      Reason: Features have vastly different scales")
    report_lines.append("")
    report_lines.append("   4. Class Imbalance Handling:")
    report_lines.append("      ✓ Apply SMOTE to minority classes (<1%)")
    report_lines.append("      Target: Bring minorities to ~1-2% of dataset")
    report_lines.append("")
    report_lines.append("   5. Feature Selection:")
    report_lines.append("      ✓ Use Recursive Feature Elimination (RFE)")
    report_lines.append("      ✓ Target: 30-40 features (optimize macro F1-score)")
    report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append(" " * 25 + "END OF EXPLORATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Report generated by: NIDS CICIDS2018 Project")
    report_lines.append("Module: Data Exploration (Module 2)")
    report_lines.append(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("Next step: Data Preprocessing (Module 3)")
    report_lines.append("=" * 80)
    
    # Write report
    report_content = "\n".join(report_lines)
    filepath = os.path.join(output_dir, 'exploration_results.txt')
    write_text_report(report_content, filepath)


def generate_exploration_steps_log(all_stats, df, output_dir):
    """
    Generate step-by-step execution log for exploration module.
    
    Parameters:
    -----------
    all_stats : dict
        All exploration statistics
    df : pandas.DataFrame
        Dataset
    output_dir : str
        Output directory
    """
    log_message("Generating step-by-step execution log...", level="INFO")
    
    steps_lines = []
    
    # Header
    steps_lines.append("=" * 80)
    steps_lines.append(" " * 20 + "MODULE 2: DATA EXPLORATION")
    steps_lines.append(" " * 25 + "STEP-BY-STEP LOG")
    steps_lines.append(" " * 20 + f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    steps_lines.append("=" * 80)
    steps_lines.append("")
    
    # Step 1
    steps_lines.append("STEP 1: ANALYZE CLASS DISTRIBUTION")
    steps_lines.append("-" * 80)
    class_stats = all_stats['class_distribution']
    majority_count = class_stats['counts'][class_stats['majority_class']]
    minority_count = class_stats['counts'][class_stats['minority_class']]
    imbalance_ratio = class_stats['imbalance_ratios'][class_stats['minority_class']]
    steps_lines.append(f"• Identified {class_stats['n_classes']} unique classes in Label column")
    steps_lines.append(f"• Majority class: {class_stats['majority_class']} with {format_number(majority_count)} samples")
    steps_lines.append(f"• Minority class: {class_stats['minority_class']} with {format_number(minority_count)} samples")
    steps_lines.append(f"• Imbalance ratio: {format_number(imbalance_ratio)}:1")
    steps_lines.append(f"• Gini coefficient: {class_stats['gini_coefficient']:.3f}")
    steps_lines.append(f"• Classes requiring SMOTE (<1%): {len(class_stats['classes_needing_smote'])}")
    steps_lines.append("✓ Class distribution analysis completed")
    steps_lines.append("")
    
    # Step 2
    steps_lines.append("STEP 2: CHECK MISSING VALUES (NaN)")
    steps_lines.append("-" * 80)
    missing_stats = all_stats['missing_data']
    steps_lines.append(f"• Scanned all {len(df.columns)} columns for missing values")
    steps_lines.append(f"• Total NaN cells found: {format_number(missing_stats['total_nan_cells'])} ({missing_stats['overall_nan_percentage']:.3f}%)")
    steps_lines.append(f"• Rows with at least one NaN: {format_number(missing_stats['rows_with_nan'])} ({missing_stats['rows_with_nan']/len(df)*100:.2f}%)")
    steps_lines.append(f"• Problematic columns (>1% NaN): {len(missing_stats['problematic_columns'])}")
    steps_lines.append(f"• Critical columns (>10% NaN): {len(missing_stats['critical_columns'])}")
    if missing_stats['critical_columns']:
        for col in missing_stats['critical_columns'][:5]:
            pct = missing_stats['nan_percentages'][col]
            steps_lines.append(f"  - {col}: {pct:.2f}% missing")
    steps_lines.append("✓ Missing value check completed")
    steps_lines.append("")
    
    # Step 3
    steps_lines.append("STEP 3: CHECK INFINITE VALUES (Inf)")
    steps_lines.append("-" * 80)
    inf_stats = all_stats['inf_values']
    steps_lines.append(f"• Scanned all numeric columns for infinite values")
    steps_lines.append(f"• Total Inf cells found: {format_number(inf_stats['total_inf_cells'])}")
    steps_lines.append(f"• Rows with at least one Inf: {format_number(inf_stats['rows_with_inf'])} ({inf_stats['rows_with_inf']/len(df)*100:.2f}%)")
    steps_lines.append(f"• Affected columns: {len(inf_stats['affected_columns'])}")
    if inf_stats['affected_columns']:
        for col in inf_stats['affected_columns']:
            count = inf_stats['inf_counts_per_column'][col]
            steps_lines.append(f"  - {col}: {format_number(count)} Inf values")
    steps_lines.append("✓ Infinite value check completed")
    steps_lines.append("")
    
    # Step 4
    steps_lines.append("STEP 4: COUNT DUPLICATE ROWS")
    steps_lines.append("-" * 80)
    dup_stats = all_stats['duplicates']
    steps_lines.append(f"• Checked for duplicate rows across all columns")
    steps_lines.append(f"• Duplicate rows found: {format_number(dup_stats['n_duplicates'])} ({dup_stats['duplicate_percentage']:.3f}%)")
    steps_lines.append("✓ Duplicate count completed")
    steps_lines.append("")
    
    # Step 5
    steps_lines.append("STEP 5: ANALYZE COLUMN-WISE DATA AVAILABILITY")
    steps_lines.append("-" * 80)
    col_avail = all_stats.get('column_availability', {})
    if col_avail:
        steps_lines.append(f"• Analyzed data availability for all {col_avail['total_columns']} columns")
        steps_lines.append(f"• Calculated non-null percentage for each column")
        high_missing = col_avail.get('high_missing_columns', [])
        steps_lines.append(f"• Columns with >10% missing: {len(high_missing)}")
        if high_missing:
            for col_stat in high_missing[:5]:
                steps_lines.append(f"  - {col_stat['column']}: {col_stat['non_null_percentage']:.2f}% available")
        steps_lines.append("✓ Column availability analysis completed")
    steps_lines.append("")
    
    # Step 6
    steps_lines.append("STEP 6: CALCULATE FEATURE CORRELATIONS")
    steps_lines.append("-" * 80)
    corr_stats = all_stats['correlation']
    steps_lines.append(f"• Identified {corr_stats['n_numeric_features']} numeric features")
    steps_lines.append(f"• Selected top {len(corr_stats['top_features'])} features by variance for correlation analysis")
    steps_lines.append(f"• Calculated {len(corr_stats['top_features'])}×{len(corr_stats['top_features'])} correlation matrix")
    steps_lines.append(f"• Found {len(corr_stats['highly_correlated_pairs'])} highly correlated pairs (|r| > {config.HIGH_CORRELATION_THRESHOLD})")
    steps_lines.append("✓ Correlation analysis completed")
    steps_lines.append("")
    
    # Step 7
    steps_lines.append("STEP 7: GENERATE DESCRIPTIVE STATISTICS")
    steps_lines.append("-" * 80)
    desc_stats = all_stats.get('descriptive_stats', None)
    if desc_stats is not None and not desc_stats.empty:
        steps_lines.append(f"• Calculated mean, std, min, max, percentiles for all numeric features")
        steps_lines.append(f"• Generated statistics for {corr_stats['n_numeric_features']} features")
        steps_lines.append("✓ Descriptive statistics completed")
    steps_lines.append("")
    
    # Step 8
    steps_lines.append("STEP 8: ANALYZE DATA TYPES AND MEMORY USAGE")
    steps_lines.append("-" * 80)
    mem_stats = all_stats['memory']
    steps_lines.append(f"• Total dataset memory: {mem_stats['total_memory_gb']:.2f} GB")
    steps_lines.append(f"• Memory per row: {mem_stats['memory_per_row_kb']:.2f} KB")
    steps_lines.append("• Memory breakdown by data type:")
    for dtype, mem_gb in mem_stats['memory_per_dtype'].items():
        pct = mem_gb / mem_stats['total_memory_gb'] * 100
        steps_lines.append(f"  - {dtype}: {mem_gb:.2f} GB ({pct:.1f}%)")
    steps_lines.append("✓ Memory analysis completed")
    steps_lines.append("")
    
    # Step 9
    steps_lines.append("STEP 9: CREATE VISUALIZATIONS")
    steps_lines.append("-" * 80)
    steps_lines.append("• Generated class_distribution.png")
    steps_lines.append("  - Vertical bar chart with class counts")
    steps_lines.append("• Generated class_imbalance_log_scale.png")
    steps_lines.append("  - Log scale chart to visualize extreme imbalance")
    steps_lines.append("• Generated correlation_heatmap.png")
    steps_lines.append("  - High-resolution 20×20 correlation matrix (DPI 150)")
    steps_lines.append("• Generated missing_data_summary.png")
    steps_lines.append("  - NaN and Inf values visualization")
    steps_lines.append("• Generated data_types_memory.png")
    steps_lines.append("  - Pie chart with side legend table")
    steps_lines.append("✓ All visualizations created")
    steps_lines.append("")
    
    # Step 10
    steps_lines.append("STEP 10: GENERATE COMPREHENSIVE TEXT REPORT")
    steps_lines.append("-" * 80)
    steps_lines.append("• Created exploration_results.txt with:")
    steps_lines.append("  - Dataset overview")
    steps_lines.append("  - Class distribution details")
    steps_lines.append("  - Data quality assessment")
    steps_lines.append("  - Feature correlation analysis")
    steps_lines.append("  - Column-wise data availability")
    steps_lines.append("  - Memory usage breakdown")
    steps_lines.append("  - Preprocessing recommendations")
    steps_lines.append("✓ Report generation completed")
    steps_lines.append("")
    
    # Summary
    steps_lines.append("=" * 80)
    steps_lines.append(" " * 25 + "EXPLORATION SUMMARY")
    steps_lines.append("=" * 80)
    steps_lines.append(f"Dataset Size: {format_number(len(df))} rows × {len(df.columns)} columns")
    steps_lines.append(f"Memory Usage: {mem_stats['total_memory_gb']:.2f} GB")
    steps_lines.append(f"Numeric Features: {corr_stats['n_numeric_features']}")
    steps_lines.append(f"Data Quality: {missing_stats['overall_nan_percentage']:.3f}% NaN, {dup_stats['duplicate_percentage']:.3f}% duplicates")
    steps_lines.append(f"")
    steps_lines.append("Next Step: Module 3 - Data Preprocessing")
    steps_lines.append("  1. Remove NaN rows")
    steps_lines.append("  2. Remove Inf rows")
    steps_lines.append("  3. Remove duplicate rows")
    steps_lines.append("  4. Consolidate labels (15 → 8 classes)")
    steps_lines.append("  5. Scale features (StandardScaler)")
    steps_lines.append("  6. Apply SMOTE for class imbalance")
    steps_lines.append("  7. Feature selection with RFE")
    steps_lines.append("=" * 80)
    
    # Write steps log
    steps_content = "\n".join(steps_lines)
    filepath = os.path.join(output_dir, 'exploration_steps.txt')
    write_text_report(steps_content, filepath)
    log_message(f"  ✓ Saved step-by-step log", "SUCCESS")


def explore_data(df, label_col, protocol_col=None):
    """
    Main function to perform comprehensive data exploration.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    label_col : str
        Label column name
    protocol_col : str, optional
        Protocol column name
        
    Returns:
    --------
    dict : All exploration statistics
    """
    print()
    print_section_header("MODULE 2: DATA EXPLORATION")
    print()
    
    overall_timer = Timer("Module 2: Data Exploration", verbose=False)
    overall_timer.__enter__()
    
    output_dir = config.REPORTS_EXPLORATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Analyze class distribution
        class_stats = analyze_class_distribution(df, label_col)
        print()
        
        # 1.5. Analyze label consolidation impact
        consolidation_stats = analyze_label_consolidation_impact(df, label_col, output_dir)
        print()
        
        # 2. Check missing data
        missing_stats = check_missing_data(df)
        print()
        
        # 3. Check infinite values
        inf_stats = check_infinite_values(df)
        print()
        
        # 4. Count duplicates
        dup_stats = count_duplicates(df)
        print()
        
        # 5. Analyze column availability
        col_avail_stats = analyze_column_availability(df)
        print()
        
        # 6. Calculate correlations
        corr_stats = calculate_correlations(df, label_col)
        print()
        
        # 7. Generate descriptive statistics
        desc_stats = generate_descriptive_statistics(df, label_col)
        print()
        
        # 8. Analyze data types and memory
        memory_stats = analyze_data_types_memory(df)
        print()
        
        # 9. Create visualizations
        log_message("Creating visualizations...", level="STEP")
        create_class_distribution_chart(class_stats, output_dir)
        create_imbalance_log_chart(class_stats, output_dir)
        create_correlation_heatmap(corr_stats, output_dir)
        create_missing_data_chart(missing_stats, inf_stats, output_dir)
        create_memory_usage_chart(memory_stats, output_dir)
        print()
        
        # 10. Generate comprehensive report
        all_stats = {
            'class_distribution': class_stats,
            'consolidation': consolidation_stats,
            'missing_data': missing_stats,
            'inf_values': inf_stats,
            'duplicates': dup_stats,
            'column_availability': col_avail_stats,
            'correlation': corr_stats,
            'descriptive_stats': desc_stats,
            'memory': memory_stats
        }
        
        generate_exploration_report(all_stats, df, label_col, output_dir)
        generate_exploration_steps_log(all_stats, df, output_dir)
        print()
        
        overall_timer.__exit__()
        
        log_message("Module 2 completed successfully!", level="SUCCESS")
        log_message(f"All reports saved to: {output_dir}", level="SUCCESS")
        print()
        
        return all_stats
    
    except Exception as e:
        log_message(f"Module 2 failed: {str(e)}", level="ERROR")
        raise


if __name__ == "__main__":
    # Test the module
    from src.data_loader import load_data
    
    df, label_col, protocol_col, _ = load_data()
    stats = explore_data(df, label_col, protocol_col)
    print("\nExploration completed!")
