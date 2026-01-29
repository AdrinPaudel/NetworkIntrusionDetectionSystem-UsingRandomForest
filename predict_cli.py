#!/usr/bin/env python
"""
CLI for NIDS Prediction - Easy interface for predictions and monitoring
Usage:
    python predict_cli.py --batch file.csv
    python predict_cli.py --single
    python predict_cli.py --monitor [--duration 120]
    python predict_cli.py --simulate [--duration 180]
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import pandas as pd

from prediction.realtime_monitor_clean import RealtimeMonitor
from prediction.realtime_simulation import RealtimeSimulation
from prediction.predictor import NetworkPredictor
from prediction.predict_preprocess import PredictionPreprocessor
from prediction.report_generator import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)



def batch_prediction(csv_file):
    """Batch prediction from CSV file - Direct pipeline (no realtime_engine)"""
    print(f"\n{'='*80}")
    print(f"BATCH PREDICTION - Processing: {csv_file}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize components
        preprocessor = PredictionPreprocessor()
        predictor = NetworkPredictor()
        report_generator = ReportGenerator()
        
        # Step 1: Load and preprocess CSV
        logger.info(f"Loading CSV: {csv_file}")
        df = preprocessor.load_csv(csv_file)
        
        logger.info(f"Cleaning data...")
        df_clean = preprocessor._validate_and_clean(df)
        
        # Step 2: Make predictions
        logger.info(f"Making predictions on {len(df_clean)} flows...")
        results_df = predictor.predict_batch(df_clean)
        
        # Step 3: Generate reports (no response actions for batch)
        attack_count = results_df['Is_Attack'].sum()
        logger.info(f"Detected {attack_count} attacks")
        
        report_files = report_generator.generate_reports(results_df, csv_file)
        
        # Display summary
        benign_count = (~results_df['Is_Attack']).sum()
        
        print(f"\n{'='*80}")
        print(f"BATCH PREDICTION RESULTS")
        print(f"{'='*80}")
        print(f"Total samples:     {len(results_df)}")
        print(f"Attacks detected:  {attack_count} ({attack_count/len(results_df)*100:.2f}%)")
        print(f"Benign traffic:    {benign_count} ({benign_count/len(results_df)*100:.2f}%)")
        
        # Show attack breakdown
        if attack_count > 0:
            print(f"\nAttack breakdown:")
            attack_types = results_df[results_df['Is_Attack']]['Predicted_Class'].value_counts()
            for attack_type, count in attack_types.items():
                print(f"  {attack_type}: {count} ({count/attack_count*100:.1f}%)")
        
        # Display sample predictions
        print(f"\nSample predictions (first 5 attacks):")
        attack_samples = results_df[results_df['Is_Attack']].head(5)
        for idx, row in attack_samples.iterrows():
            print(f"  [{idx}] {row['Predicted_Class']:<15} (confidence: {row['Confidence']:.4f})")
        
        print(f"\nResults saved:")
        print(f"  - RED log:         {report_files['red']}")
        print(f"  - YELLOW log:      {report_files['yellow']}")
        print(f"  - Full log:        {report_files['full']}")
        print(f"  - Summary report:  {report_files['report']}")
        
        print(f"\n{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def single_prediction():
    """Interactive single prediction"""
    print(f"\n{'='*80}")
    print(f"SINGLE PREDICTION - Interactive Mode")
    print(f"{'='*80}\n")
    
    try:
        predictor = NetworkPredictor()
        
        # Show required features
        features = predictor.get_feature_requirements()
        print(f"Enter network flow data. Required features: {len(features)}")
        print(f"(Press Enter with empty value to use default 0)")
        print(f"\nKey features to enter:")
        key_features = ['Dst Port', 'Src Port', 'Protocol', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
                       'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Duration']
        for f in key_features:
            print(f"  - {f}")
        
        # Collect input
        data_dict = {}
        for feature in key_features:
            while True:
                try:
                    value = input(f"{feature}: ").strip()
                    if value == '':
                        data_dict[feature] = 0.0
                    else:
                        data_dict[feature] = float(value)
                    break
                except ValueError:
                    print("  Invalid input. Enter a number.")
        
        # Fill remaining features with 0
        for feature in features:
            if feature not in data_dict:
                data_dict[feature] = 0.0
        
        # Make prediction
        df = pd.DataFrame([data_dict])
        result = predictor.predict_batch(df)
        
        # Display result
        print(f"\n{'='*80}")
        print(f"PREDICTION RESULT")
        print(f"{'='*80}")
        print(f"Classification: {result['predictions'][0]}")
        print(f"Confidence: {result['confidences'][0]:.4f} ({result['confidences'][0]*100:.2f}%)")
        
        is_attack = result['predictions'][0] != 'Benign'
        print(f"Is Attack: {is_attack}")
        
        if is_attack:
            print(f"\n[ALERT] ATTACK DETECTED")
        else:
            print(f"\n[OK] Benign traffic detected")
        
        print(f"\n{'='*80}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Single prediction failed: {e}")
        return False


def realtime_monitoring(interface=None, duration=None):
    """Start real-time threat detection - integrated network monitoring + threat prediction
    
    Args:
        interface: Network interface to monitor (optional)
        duration: Duration in seconds. Default: 300 (5 minutes). None = 5 min. Set explicitly for different duration.
    """
    print(f"\n{'='*80}")
    print(f"REAL-TIME THREAT DETECTION")
    print(f"{'='*80}\n")
    
    try:
        # Create realtime monitor
        monitor = RealtimeMonitor()
        
        # Set default to 5 minutes if not specified
        if duration is None:
            duration = 300
            duration_str = "5 minutes"
        else:
            duration_str = f"{duration} seconds ({duration/60:.1f} minutes)"
        
        print(f"Starting real-time threat detection...")
        print(f"Duration: {duration_str}")
        print(f"\nMonitoring in progress. Press Ctrl+C to stop.\n")
        
        # Start the monitor
        monitor.start(duration=duration)
        
        return True
        
    except Exception as e:
        logger.error(f"Real-time monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def simulated_traffic(duration=None):
    """Start simulated traffic analysis using CICIDS2018 CSV data
    
    Args:
        duration: Duration in seconds. Default: 180 (3 minutes)
    """
    print(f"\n{'='*80}")
    print(f"SIMULATED TRAFFIC ANALYSIS")
    print(f"{'='*80}\n")
    
    try:
        # Find CSV file in data/realtrafficsimul/
        csv_dir = Path('data/realtrafficsimul')
        
        # Prefer test_sample_1000.csv (smaller, faster) over the huge 8.8M row file
        csv_file = csv_dir / 'test_sample_1000.csv'
        if not csv_file.exists():
            csv_files = list(csv_dir.glob('*.csv'))
            if not csv_files:
                logger.error("No CSV files found in data/realtrafficsimul/")
                return False
            csv_file = csv_files[0]
        
        logger.info(f"Using CSV: {csv_file}")
        
        # Create simulator with sample size (1000 rows from test_sample, or 10k from larger file)
        sample_size = 1000 if 'test_sample' in str(csv_file) else 10000
        simulator = RealtimeSimulation(str(csv_file), sample_size=sample_size)
        
        # Set default to 3 minutes if not specified
        if duration is None:
            duration = 180
            duration_str = "3 minutes"
        else:
            duration_str = f"{duration} seconds ({duration/60:.1f} minutes)"
        
        print(f"Starting simulated traffic analysis...")
        print(f"Duration: {duration_str}")
        print(f"Rate: 5 flows/sec")
        print(f"\nSimulation in progress. Press Ctrl+C to stop.\n")
        
        # Start simulation
        simulator.start(duration=duration, rate=5)
        
        return True
        
    except Exception as e:
        logger.error(f"Simulated traffic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_features():
    """Show required features for prediction"""
    try:
        predictor = NetworkPredictor()
        features = predictor.get_feature_requirements()
        
        print(f"\n{'='*80}")
        print(f"REQUIRED FEATURES FOR PREDICTION")
        print(f"{'='*80}\n")
        
        print(f"Total features: {len(features)}\n")
        
        for i, feature in enumerate(features, 1):
            print(f"{i:2d}. {feature}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Error showing features: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='NIDS Prediction CLI - Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch prediction from CSV
  python predict_cli.py --batch data.csv
  
  # Single interactive prediction
  python predict_cli.py --single
  
  # Real-time monitoring
  python predict_cli.py --monitor
  
  # Real-time on specific interface with custom duration
  python predict_cli.py --monitor --interface eth0 --duration 600
  
  # Simulated traffic analysis (3 minutes default)
  python predict_cli.py --simulate
  
  # Simulated traffic analysis (1 minute)
  python predict_cli.py --simulate --duration 60
  
  # Show required features
  python predict_cli.py --features
        """
    )
    
    parser.add_argument('--batch', type=str, 
                       help='Process CSV file with batch predictions')
    parser.add_argument('--single', action='store_true',
                       help='Interactive single prediction mode')
    parser.add_argument('--monitor', action='store_true',
                       help='Start real-time network threat detection')
    parser.add_argument('--interface', type=str, default=None,
                       help='Network interface for monitoring (e.g., eth0, en0)')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulated traffic analysis using CICIDS2018 CSV data')
    parser.add_argument('--duration', type=int, default=None,
                       help='Monitoring/simulation duration in seconds')
    parser.add_argument('--features', action='store_true',
                       help='Show all required features')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.batch:
        if not Path(args.batch).exists():
            logger.error(f"File not found: {args.batch}")
            return 1
        success = batch_prediction(args.batch)
        return 0 if success else 1
    
    elif args.single:
        success = single_prediction()
        return 0 if success else 1
    
    elif args.simulate:
        success = simulated_traffic(duration=args.duration)
        return 0 if success else 1
    
    elif args.monitor:
        success = realtime_monitoring(interface=args.interface, duration=args.duration)
        return 0 if success else 1
    
    elif args.features:
        show_features()
        return 0
    
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
