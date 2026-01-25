"""
NIDS CICIDS2018 Project - Main CLI
Command-line interface for running the complete NIDS pipeline
"""

import sys
import argparse
from src.utils import create_directory_structure, log_message, print_section_header
from src.data_loader import load_data
from src.explorer import explore_data
import config


def run_module_1():
    """Run Module 1: Data Loading"""
    log_message("Starting Module 1: Data Loading", level="INFO")
    df, label_col, protocol_col, stats = load_data()
    return df, label_col, protocol_col, stats


def run_module_2(df, label_col, protocol_col):
    """Run Module 2: Data Exploration"""
    log_message("Starting Module 2: Data Exploration", level="INFO")
    exploration_stats = explore_data(df, label_col, protocol_col)
    return exploration_stats


def run_module_3(df, label_col, protocol_col, resume_from=None):
    """Run Module 3: Data Preprocessing"""
    log_message("Starting Module 3: Data Preprocessing", level="INFO")
    from src.preprocessor import preprocess_data
    preprocessing_result = preprocess_data(df, label_col, protocol_col, resume_from=resume_from)
    return preprocessing_result


def run_module_4():
    """Run Module 4: Model Training"""
    log_message("Starting Module 4: Model Training", level="INFO")
    from src.trainer import train_model
    training_results = train_model(
        data_dir='data/preprocessed',
        model_dir='trained_model',
        reports_dir='reports/training',
        n_iter=config.N_ITER_SEARCH,  # Use config setting (50 iterations)
        cv=config.CV_FOLDS,
        random_state=config.RANDOM_STATE
    )
    return training_results


def run_module_5():
    """Run Module 5: Model Testing"""
    log_message("Starting Module 5: Model Testing", level="INFO")
    from src.tester import test_model
    testing_results = test_model()
    return testing_results


def run_full_pipeline():
    """Run the complete pipeline from start to finish"""
    import os
    
    print()
    print_section_header("NIDS CICIDS2018 PROJECT - FULL PIPELINE")
    print()
    
    log_message("Starting full pipeline execution...", level="INFO")
    print()
    
    try:
        # Setup
        create_directory_structure()
        print()
        
        # Check if exploration already complete
        exploration_report = os.path.join(config.REPORTS_EXPLORATION_DIR, 'exploration_results.txt')
        exploration_exists = os.path.exists(exploration_report)
        
        if exploration_exists:
            log_message("✓ Exploration reports found - skipping Module 2", level="INFO")
            log_message(f"Using existing reports from: {exploration_report}", level="INFO")
            print()
        
        # Module 1: Data Loading
        df, label_col, protocol_col, load_stats = run_module_1()
        
        # Module 2: Data Exploration (skip if already done)
        if not exploration_exists:
            exploration_stats = run_module_2(df, label_col, protocol_col)
        else:
            log_message("Skipping Module 2 (already completed)", level="INFO")
            print()
        
        # Module 3: Data Preprocessing
        preprocessing_results = run_module_3(df, label_col, protocol_col)
        
        # Module 4: Model Training
        training_results = run_module_4()
        
        # Module 5: Model Testing
        testing_results = run_module_5()
        
        print_section_header("PIPELINE EXECUTION COMPLETED")
        log_message("All modules (1-5) completed successfully!", level="SUCCESS")
        log_message(f"Final Macro F1-Score: {testing_results['multiclass_results']['aggregate_metrics']['macro_f1']:.4f}", level="SUCCESS")
        print()
        
    except KeyboardInterrupt:
        log_message("\nPipeline interrupted by user", level="WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"\nPipeline failed: {str(e)}", level="ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NIDS CICIDS2018 Project - Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --full
  
  # Run specific module
  python main.py --module 1
  python main.py --module 2
  
  # Run modules 1 and 2
  python main.py --module 1 --module 2
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run the complete pipeline (all modules)'
    )
    
    parser.add_argument(
        '--module',
        type=int,
        action='append',
        choices=[1, 2, 3, 4, 5],
        help='Run specific module(s). Can be specified multiple times.'
    )
    
    parser.add_argument(
        '--resume-from',
        type=int,
        choices=[1, 2, 3],
        help='Resume Module 3 from checkpoint (1=cleaned, 2=encoded, 3=smoted)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not args.full and not args.module:
        parser.print_help()
        sys.exit(0)
    
    # Setup directories
    create_directory_structure()
    print()
    
    # Run full pipeline
    if args.full:
        run_full_pipeline()
    
    # Run specific modules
    elif args.module:
        modules_to_run = sorted(set(args.module))
        
        df = None
        label_col = None
        protocol_col = None
        exploration_stats = None
        
        for module_num in modules_to_run:
            if module_num == 1:
                df, label_col, protocol_col, load_stats = run_module_1()
            
            elif module_num == 2:
                if df is None:
                    log_message("Module 2 requires Module 1 to run first. Running Module 1...", 
                               level="INFO")
                    df, label_col, protocol_col, load_stats = run_module_1()
                
                exploration_stats = run_module_2(df, label_col, protocol_col)
            
            elif module_num == 3:
                resume_from = getattr(args, 'resume_from', None)
                
                if resume_from:
                    log_message(f"Resuming Module 3 from checkpoint {resume_from}", level="INFO")
                    # Don't load Module 1 data if resuming
                    df = None
                    label_col = None
                    protocol_col = None
                else:
                    if df is None:
                        log_message("Module 3 requires Module 1 data. Loading...", 
                                   level="INFO")
                        df, label_col, protocol_col, load_stats = run_module_1()
                        # Skip Module 2 - exploration not needed for preprocessing
                
                preprocessing_result = run_module_3(df, label_col, protocol_col, resume_from=resume_from)
            
            elif module_num == 4:
                # Module 4 loads data from preprocessed files
                training_results = run_module_4()
            
            elif module_num == 5:
                # Module 5 loads trained model and test data from files
                testing_results = run_module_5()
        
        print()
        log_message("Requested modules completed!", level="SUCCESS")
        print()


if __name__ == "__main__":
    main()
