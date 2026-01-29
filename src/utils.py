"""
Utility Functions for NIDS CICIDS2018 Project
Shared helper functions used across all modules
"""

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import config


def create_directory_structure():
    """
    Create all necessary directories for the project.
    Directories are created if they don't exist.
    """
    directories = [
        config.DATA_RAW_DIR,
        config.DATA_PREPROCESSED_DIR,
        config.TRAINED_MODEL_DIR,
        config.REPORTS_EXPLORATION_DIR,
        config.REPORTS_PREPROCESSING_DIR,
        config.REPORTS_TRAINING_DIR,
        config.REPORTS_TESTING_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    log_message("[SETUP] All directories created/verified", level="SUCCESS")


def log_message(message, level="INFO", print_timestamp=True):
    """
    Print a timestamped log message to console.
    
    Parameters:
    -----------
    message : str
        Message to log
    level : str
        Log level: INFO, SUCCESS, WARNING, ERROR, STEP, SUBSTEP
    print_timestamp : bool
        Whether to include timestamp
    """
    timestamp = datetime.now().strftime(config.LOG_TIMESTAMP_FORMAT)
    
    # Color codes for different log levels
    colors = {
        'INFO': '\033[94m',      # Blue
        'SUCCESS': '\033[92m',   # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'STEP': '\033[96m',      # Cyan
        'SUBSTEP': '\033[95m',   # Magenta
    }
    reset = '\033[0m'
    
    color = colors.get(level, '')
    
    if print_timestamp:
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")
    else:
        print(f"{color}[{level}] {message}{reset}")


def format_time(seconds):
    """
    Convert seconds to human-readable format.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    str : Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes ({seconds:.1f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f} hours ({minutes:.1f} minutes)"


def format_number(number):
    """
    Format large numbers with commas.
    
    Parameters:
    -----------
    number : int or float
        Number to format
        
    Returns:
    --------
    str : Formatted number string
    """
    if isinstance(number, float):
        return f"{number:,.2f}"
    else:
        return f"{number:,}"


def calculate_memory_usage(df):
    """
    Calculate DataFrame memory usage in GB.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    float : Memory usage in GB
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_gb = memory_bytes / (1024 ** 3)
    return memory_gb


def save_figure(fig, filepath, dpi=None):
    """
    Save matplotlib figure to file.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Output file path
    dpi : int, optional
        Resolution (dots per inch)
    """
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=config.FIGURE_FORMAT)
    plt.close(fig)
    log_message(f"Saved figure: {os.path.basename(filepath)}", level="SUCCESS")


def print_separator(char='=', length=80):
    """
    Print a separator line.
    
    Parameters:
    -----------
    char : str
        Character to use for separator
    length : int
        Length of separator line
    """
    print(char * length)


def print_section_header(title, length=80):
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Section title
    length : int
        Width of header
    """
    print_separator('=', length)
    print(f"  {title}")
    print_separator('=', length)


def print_subsection_header(title, length=80):
    """
    Print a formatted subsection header.
    
    Parameters:
    -----------
    title : str
        Subsection title
    length : int
        Width of header
    """
    print_separator('-', length)
    print(f"  {title}")
    print_separator('-', length)


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
    ------
    with Timer("Operation name"):
        # code to time
        pass
    """
    
    def __init__(self, name="Operation", verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            log_message(f"Starting: {self.name}", level="INFO")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            log_message(f"Completed: {self.name} in {format_time(self.elapsed)}", 
                       level="SUCCESS")


def get_file_size(filepath):
    """
    Get file size in human-readable format.
    
    Parameters:
    -----------
    filepath : str
        Path to file
        
    Returns:
    --------
    str : Formatted file size
    """
    if not os.path.exists(filepath):
        return "File not found"
    
    size_bytes = os.path.getsize(filepath)
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def write_text_report(content, filepath):
    """
    Write text report to file.
    
    Parameters:
    -----------
    content : str
        Report content
    filepath : str
        Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    log_message(f"Saved report: {os.path.basename(filepath)}", level="SUCCESS")


def calculate_imbalance_ratio(class_counts):
    """
    Calculate imbalance ratios relative to majority class.
    
    Parameters:
    -----------
    class_counts : pandas.Series
        Value counts per class
        
    Returns:
    --------
    dict : Imbalance ratios
    """
    max_count = class_counts.max()
    ratios = {cls: max_count / count for cls, count in class_counts.items()}
    return ratios
