"""
Real-time Traffic Simulation using CICIDS2018 CSV Data
Simulates real network traffic by reading pre-extracted flows from CSV
Uses same threading architecture as realtime monitoring
"""

import time
import threading
import queue
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os

from .predictor import NetworkPredictor
from .predict_preprocess import PredictionPreprocessor


class RealtimeSimulation:
    """Simulate real-time monitoring using historical flow data from CSV"""
    
    def __init__(self, csv_file, sample_size=10000):
        """Initialize simulation with CSV file
        
        Args:
            csv_file: Path to CSV with flow data
            sample_size: Max rows to load (avoid huge file issues)
        """
        print("[*] Loading CSV data...")
        self.csv_file = Path(csv_file)
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        
        # Load CSV (with chunking for large files)
        print(f"[*] Reading CSV (max {sample_size} rows)...")
        self.df = pd.read_csv(csv_file, nrows=sample_size)
        print(f"[*] Loaded {len(self.df)} flows from CSV")
        
        # Shuffle and save to temp file
        self.shuffled_file = self._shuffle_and_save()
        print(f"[*] Shuffled data saved to temp file: {self.shuffled_file}")
        
        # Load predictor
        print("[*] Loading predictor...")
        self.predictor = NetworkPredictor()
        self.preprocessor = PredictionPreprocessor()
        
        # Threading
        self.stop_event = threading.Event()
        self.sniffer_queue = queue.Queue(maxsize=500)  # CSV reader → predictor
        self.predictor_queue = queue.Queue(maxsize=500)  # Predictor → reporter
        
        # Tracking
        self.predictions = []
        self.flow_count = 0
        self.start_time = None
        self.session_dir = None
        self.current_minute = None
        self.minute_dir = None
        self.minute_predictions = []
        
        # Counters
        self.red_count = 0
        self.yellow_count = 0
        self.green_count = 0
        
        print(f"[*] Model loaded: {self.predictor.model is not None}")
        print(f"[*] Ready to simulate traffic")
    
    def _shuffle_and_save(self):
        """Shuffle CSV and save to temp file (no seed - random each time)"""
        # Shuffle without seed (truly random)
        shuffled_df = self.df.sample(frac=1, random_state=None).reset_index(drop=True)
        
        # Save to data/temp/ directory
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"nids_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shuffled_df.to_csv(temp_file, index=False)
        
        return str(temp_file)
    
    def csv_reader_worker(self, rate=5, max_flows=None):
        """Thread 1: Read shuffled CSV and queue flows
        
        Args:
            rate: Flows per second
            max_flows: Max flows to process (None = all)
        """
        print(f"CSV Reader worker started (rate={rate}/sec)...")
        
        flow_count = 0
        
        try:
            # Read shuffled CSV
            df = pd.read_csv(self.shuffled_file)
            
            if max_flows:
                df = df.head(max_flows)
            
            start_time = time.time()
            
            # Get feature columns (all except Timestamp)
            feature_cols = [col for col in df.columns if col != 'Timestamp']
            
            for idx, row in df.iterrows():
                if self.stop_event.is_set():
                    break
                
                # Rate limiting: sleep to maintain rate
                expected_time = start_time + (flow_count / rate)
                sleep_time = expected_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Extract features and timestamp
                timestamp = row.get('Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                features_dict = row[feature_cols].to_dict()
                
                # Queue the data
                try:
                    self.sniffer_queue.put_nowait({
                        'timestamp': timestamp,
                        'features': features_dict,
                        'flow_id': idx
                    })
                    flow_count += 1
                    self.flow_count += 1
                except queue.Full:
                    pass  # Drop if queue full
        
        except Exception as e:
            print(f"[!] CSV Reader error: {e}")
        
        finally:
            # Signal end
            self.sniffer_queue.put(None)
            print(f"CSV Reader worker stopped. Processed {flow_count} flows.")
    
    def predictor_worker(self):
        """Thread 2: Predict on flows from CSV"""
        print("Predictor worker started...")
        
        while not self.stop_event.is_set():
            try:
                item = self.sniffer_queue.get(timeout=1)
                if item is None:
                    break
                
                self.predict_flow(item)
            
            except queue.Empty:
                continue
            except Exception:
                pass
        
        self.predictor_queue.put(None)
        print("Predictor worker stopped.")
    
    def predict_flow(self, flow_data):
        """Extract features and predict"""
        try:
            timestamp = flow_data['timestamp']
            features_dict = flow_data['features']
            
            # Create DataFrame with features in correct order
            feature_list = self.predictor.feature_list
            feature_values = [features_dict.get(f, 0) for f in feature_list]
            df = pd.DataFrame([feature_values], columns=feature_list)
            
            # Preprocess
            df_clean = self.preprocessor._validate_and_clean(df)
            
            # Predict
            result = self.predictor.predict_batch(df_clean)
            
            if result is not None and len(result) > 0:
                pred = result.iloc[0]
                
                prediction_data = {
                    'timestamp': timestamp,
                    'flow_id': flow_data['flow_id'],
                    'predicted_class': pred['Predicted_Class'],
                    'confidence': pred['Confidence'],
                    'secondary_class': pred.get('Secondary_Class', 'N/A'),
                    'secondary_confidence': pred.get('Secondary_Confidence', 0.0),
                    'alert_level': pred['Alert_Level'],
                    'is_attack': pred['Is_Attack']
                }
                
                try:
                    self.predictor_queue.put_nowait(prediction_data)
                except queue.Full:
                    pass
        
        except Exception:
            pass
    
    def reporter_worker(self):
        """Thread 3: Report predictions"""
        print("Reporter worker started...")
        
        while not self.stop_event.is_set():
            try:
                item = self.predictor_queue.get(timeout=1)
                if item is None:
                    break
                
                self.log_prediction(item)
                self.check_minute_rotation()
            
            except queue.Empty:
                self.check_minute_rotation()
                continue
            except Exception:
                pass
        
        print("Reporter worker stopped.")
    
    def log_prediction(self, pred_data):
        """Write prediction to log files"""
        if not self.minute_dir:
            return
        
        alert_level = pred_data['alert_level']
        timestamp = pred_data['timestamp']
        
        # Update counters
        if alert_level == 'RED':
            self.red_count += 1
        elif alert_level == 'YELLOW':
            self.yellow_count += 1
        elif alert_level == 'GREEN':
            self.green_count += 1
        
        # Add to minute predictions
        self.minute_predictions.append(pred_data)
        
        # Determine log files
        log_files = []
        if alert_level == 'RED':
            log_files = ['red.log', 'full.log']
        elif alert_level == 'YELLOW':
            log_files = ['yellow.log', 'full.log']
        else:
            log_files = ['full.log']
        
        # Format log line
        log_line = (f"{timestamp} | {alert_level:6} | {pred_data['predicted_class']:15} | "
                   f"{pred_data['confidence']:>10.2%} | {pred_data.get('secondary_class', 'N/A'):15} | "
                   f"{pred_data.get('secondary_confidence', 0.0):>10.2%} | Flow #{pred_data['flow_id']}\n")
        
        # Write to logs
        for log_file in log_files:
            log_path = self.minute_dir / log_file
            with open(log_path, 'a') as f:
                f.write(log_line)
        
        # Console output
        if alert_level in ['RED', 'YELLOW']:
            print(f"[{alert_level}] {timestamp} | {pred_data['predicted_class']} ({pred_data['confidence']:.1%})")
    
    def check_minute_rotation(self):
        """Create new minute folder if needed"""
        current_minute = datetime.now().strftime("%Y%m%d_%H%M")
        
        if self.current_minute != current_minute:
            # Save previous minute
            if self.minute_dir and self.minute_predictions:
                self.save_minute_report()
            
            # Create new minute folder
            self.current_minute = current_minute
            self.minute_dir = self.session_dir / f"minute_{current_minute}"
            self.minute_dir.mkdir(parents=True, exist_ok=True)
            self.minute_predictions = []
            
            # Init log files
            for log_file in ['red.log', 'yellow.log', 'full.log']:
                log_path = self.minute_dir / log_file
                with open(log_path, 'w') as f:
                    f.write(f"Simulation Log - Started: {datetime.now().isoformat()}\n")
                    f.write("="*160 + "\n")
                    f.write("Timestamp | Alert Level | Attack Type | Confidence | Secondary Type | Secondary Conf | Flow ID\n")
                    f.write("="*160 + "\n")
    
    def save_minute_report(self):
        """Generate minute summary"""
        if not self.minute_predictions:
            return
        
        report_path = self.minute_dir / "report.txt"
        
        red = len([p for p in self.minute_predictions if p['alert_level'] == 'RED'])
        yellow = len([p for p in self.minute_predictions if p['alert_level'] == 'YELLOW'])
        green = len([p for p in self.minute_predictions if p['alert_level'] == 'GREEN'])
        total = len(self.minute_predictions)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"SIMULATION MINUTE REPORT - {self.current_minute}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("ALERT SUMMARY:\n")
            f.write(f"  Total Flows:  {total}\n")
            f.write(f"  RED (Attacks):   {red:3} ({red/total*100 if total > 0 else 0:.1f}%)\n")
            f.write(f"  YELLOW (Warn):   {yellow:3} ({yellow/total*100 if total > 0 else 0:.1f}%)\n")
            f.write(f"  GREEN (Safe):    {green:3} ({green/total*100 if total > 0 else 0:.1f}%)\n\n")
            
            if red > 0:
                f.write("ATTACK TYPES:\n")
                attacks = [p for p in self.minute_predictions if p['alert_level'] == 'RED']
                attack_types = {}
                for a in attacks:
                    at = a['predicted_class']
                    attack_types[at] = attack_types.get(at, 0) + 1
                
                for at, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {at:15} {count}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def start(self, duration=180, rate=5):
        """Start simulation
        
        Args:
            duration: Duration in seconds (default 180 = 3 min)
            rate: Flows per second (default 5)
        """
        self.start_time = time.time()
        
        # Create session directory
        results_dir = Path('results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = results_dir / f"simulated_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"SIMULATED TRAFFIC ANALYSIS (from CICIDS2018 CSV)")
        print(f"{'='*80}")
        print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
        print(f"Rate: {rate} flows/sec")
        print(f"Output: {self.session_dir}")
        print(f"{'='*80}\n")
        
        # Calculate max flows based on duration and rate
        max_flows = duration * rate
        
        # Start workers
        reader_thread = threading.Thread(target=self.csv_reader_worker, args=(rate, max_flows), daemon=False)
        predictor_thread = threading.Thread(target=self.predictor_worker, daemon=False)
        reporter_thread = threading.Thread(target=self.reporter_worker, daemon=False)
        
        try:
            reader_thread.start()
            predictor_thread.start()
            reporter_thread.start()
            
            # Wait for completion
            reader_thread.join()
            predictor_thread.join()
            self.stop_event.set()
            reporter_thread.join()
        
        except KeyboardInterrupt:
            print("\n[*] Stopping simulation...")
            self.stop_event.set()
            reader_thread.join(timeout=2)
            predictor_thread.join(timeout=2)
            reporter_thread.join(timeout=2)
        
        # Finalize
        self.finalize_session()
    
    def finalize_session(self):
        """Generate final session summary"""
        elapsed = time.time() - self.start_time
        
        # Save last minute
        if self.minute_predictions:
            self.save_minute_report()
        
        print(f"\n{'='*80}")
        print(f"SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"Duration:     {elapsed:.1f}s")
        print(f"Flows:        {self.flow_count}")
        print(f"Predictions:  {len(self.predictions)}")
        print(f"RED:          {self.red_count}")
        print(f"YELLOW:       {self.yellow_count}")
        print(f"GREEN:        {self.green_count}")
        print(f"Output:       {self.session_dir}")
        print(f"{'='*80}\n")
        
        # Create session summary
        session_report = self.session_dir / "session_summary.txt"
        with open(session_report, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SIMULATED MONITORING SESSION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {elapsed:.1f} seconds\n")
            f.write(f"Session Directory: {self.session_dir}\n")
            f.write(f"Source CSV: {self.csv_file}\n\n")
            
            f.write(f"STATISTICS:\n")
            f.write(f"  Total Flows:     {self.flow_count}\n")
            f.write(f"  Total Analyzed:  {len(self.predictions)}\n\n")
            
            f.write(f"ALERT BREAKDOWN:\n")
            f.write(f"  RED (Attacks):   {self.red_count}\n")
            f.write(f"  YELLOW (Warn):   {self.yellow_count}\n")
            f.write(f"  GREEN (Safe):    {self.green_count}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"  Per-minute folders: {self.session_dir}/minute_*\n")
            f.write(f"  Each minute contains:\n")
            f.write(f"    - red.log (attacks only)\n")
            f.write(f"    - yellow.log (warnings only)\n")
            f.write(f"    - full.log (all flows)\n")
            f.write(f"    - report.txt (summary)\n")
            f.write("\n" + "="*80 + "\n")
        
        # Clean up temp file
        if Path(self.shuffled_file).exists():
            try:
                os.remove(self.shuffled_file)
                print(f"[*] Cleaned up temp file: {self.shuffled_file}")
            except:
                pass
