"""
Clean Real-time Network Monitor with Threading
Parallel pipeline: Sniff -> Queue A -> Predict -> Queue B -> Report
"""

import time
import threading
import queue
import signal
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP
from pathlib import Path

from .predictor import NetworkPredictor
from .predict_preprocess import PredictionPreprocessor


class FlowData:
    """Track packets for a network flow"""
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.start_time = time.time()
        self.end_time = time.time()
        self.packet_count = 0
        
        # Forward/backward tracking
        self.fwd_packets = []
        self.bwd_packets = []
        self.fwd_bytes = []
        self.bwd_bytes = []
        self.fwd_iat = []
        self.bwd_iat = []
        self.fwd_flags = defaultdict(int)
        self.bwd_flags = defaultdict(int)
        self.fwd_window = []
        self.bwd_window = []
        self.last_packet_time = time.time()
    
    def add_packet(self, is_forward, length, flags, window):
        """Add packet to flow"""
        current_time = time.time()
        self.packet_count += 1
        self.end_time = current_time
        
        if is_forward:
            self.fwd_packets.append(length)
            self.fwd_bytes.append(length)
            if self.fwd_iat:
                self.fwd_iat.append(current_time - self.last_packet_time)
            if window:
                self.fwd_window.append(window)
            for flag in flags.split(','):
                if flag.strip():
                    self.fwd_flags[flag.strip()] += 1
        else:
            self.bwd_packets.append(length)
            self.bwd_bytes.append(length)
            if self.bwd_iat:
                self.bwd_iat.append(current_time - self.last_packet_time)
            if window:
                self.bwd_window.append(window)
            for flag in flags.split(','):
                if flag.strip():
                    self.bwd_flags[flag.strip()] += 1
        
        self.last_packet_time = current_time
    
    def get_features(self):
        """Extract 45 features"""
        duration = max(self.end_time - self.start_time, 0.001)
        total_fwd = len(self.fwd_packets)
        total_bwd = len(self.bwd_packets)
        
        features = {
            'Dst Port': self.dst_port,
            'Fwd Header Len': 20 if total_fwd > 0 else 0,
            'Fwd Seg Size Min': min(self.fwd_packets) if self.fwd_packets else 0,
            'Init Fwd Win Byts': self.fwd_window[0] if self.fwd_window else 0,
            'Bwd Seg Size Avg': np.mean(self.bwd_packets) if self.bwd_packets else 0,
            'Pkt Len Max': max(self.fwd_packets + self.bwd_packets) if (self.fwd_packets or self.bwd_packets) else 0,
            'TotLen Fwd Pkts': sum(self.fwd_bytes),
            'Subflow Bwd Byts': sum(self.bwd_bytes),
            'Bwd Pkt Len Std': np.std(self.bwd_packets) if self.bwd_packets else 0,
            'Tot Fwd Pkts': total_fwd,
            'Fwd IAT Tot': sum(self.fwd_iat) if self.fwd_iat else 0,
            'Fwd IAT Max': max(self.fwd_iat) if self.fwd_iat else 0,
            'Bwd Pkt Len Max': max(self.bwd_packets) if self.bwd_packets else 0,
            'Bwd Pkt Len Mean': np.mean(self.bwd_packets) if self.bwd_packets else 0,
            'Fwd Act Data Pkts': len([p for p in self.fwd_packets if p > 0]),
            'TotLen Bwd Pkts': sum(self.bwd_bytes),
            'Subflow Bwd Pkts': total_bwd,
            'Flow IAT Max': max(self.fwd_iat + self.bwd_iat) if (self.fwd_iat or self.bwd_iat) else 0,
            'Flow Duration': duration * 1000000,
            'Flow IAT Mean': np.mean(self.fwd_iat + self.bwd_iat) if (self.fwd_iat or self.bwd_iat) else 0,
            'Fwd IAT Mean': np.mean(self.fwd_iat) if self.fwd_iat else 0,
            'Pkt Len Var': np.var(self.fwd_packets + self.bwd_packets) if (self.fwd_packets or self.bwd_packets) else 0,
            'Pkt Len Std': np.std(self.fwd_packets + self.bwd_packets) if (self.fwd_packets or self.bwd_packets) else 0,
            'Bwd Header Len': 20 if total_bwd > 0 else 0,
            'Subflow Fwd Pkts': total_fwd,
            'Fwd Pkt Len Max': max(self.fwd_packets) if self.fwd_packets else 0,
            'Subflow Fwd Byts': sum(self.fwd_bytes),
            'Tot Bwd Pkts': total_bwd,
            'Fwd Pkts/s': total_fwd / duration if duration > 0 else 0,
            'Fwd Seg Size Avg': np.mean(self.fwd_packets) if self.fwd_packets else 0,
            'Fwd Pkt Len Std': np.std(self.fwd_packets) if self.fwd_packets else 0,
            'Bwd Pkts/s': total_bwd / duration if duration > 0 else 0,
            'Fwd IAT Min': min(self.fwd_iat) if self.fwd_iat else 0,
            'Pkt Len Mean': np.mean(self.fwd_packets + self.bwd_packets) if (self.fwd_packets or self.bwd_packets) else 0,
            'Flow IAT Min': min(self.fwd_iat + self.bwd_iat) if (self.fwd_iat or self.bwd_iat) else 0,
            'Fwd Pkt Len Mean': np.mean(self.fwd_packets) if self.fwd_packets else 0,
            'Flow Pkts/s': self.packet_count / duration if duration > 0 else 0,
            'Idle Mean': 0,
            'Pkt Size Avg': np.mean(self.fwd_bytes + self.bwd_bytes) if (self.fwd_bytes or self.bwd_bytes) else 0,
            'ACK Flag Cnt': self.fwd_flags.get('ACK', 0) + self.bwd_flags.get('ACK', 0),
            'Init Bwd Win Byts': self.bwd_window[0] if self.bwd_window else 0,
            'Idle Min': 0,
            'Flow IAT Std': np.std(self.fwd_iat + self.bwd_iat) if (self.fwd_iat or self.bwd_iat) else 0,
            'RST Flag Cnt': self.fwd_flags.get('RST', 0) + self.bwd_flags.get('RST', 0),
            'Bwd IAT Std': np.std(self.bwd_iat) if self.bwd_iat else 0,
        }
        return features


class RealtimeMonitor:
    """Multi-threaded realtime network monitor with queue-based pipeline"""
    
    def __init__(self):
        print("[*] Loading predictor...")
        self.predictor = NetworkPredictor()
        self.preprocessor = PredictionPreprocessor()
        
        # Threading
        self.stop_event = threading.Event()
        self.sniffer_queue = queue.Queue(maxsize=500)  # Completed flows waiting to predict
        self.predictor_queue = queue.Queue(maxsize=500)  # Predictions waiting to report
        
        # Tracking
        self.flows = {}
        self.predictions = []
        self.packet_count = 0
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
        print(f"[*] Ready to monitor network traffic")
    
    def process_packet(self, packet):
        """Process each captured packet (runs in sniffer thread)"""
        try:
            if not (IP in packet):
                return
            
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = "OTHER"
            src_port = 0
            dst_port = 0
            length = len(packet)
            flags = ""
            window = 0
            
            if TCP in packet:
                protocol = "TCP"
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                window = packet[TCP].window
                flag_list = []
                if packet[TCP].flags.F: flag_list.append('FIN')
                if packet[TCP].flags.S: flag_list.append('SYN')
                if packet[TCP].flags.R: flag_list.append('RST')
                if packet[TCP].flags.P: flag_list.append('PSH')
                if packet[TCP].flags.A: flag_list.append('ACK')
                if packet[TCP].flags.U: flag_list.append('URG')
                flags = ','.join(flag_list)
            elif UDP in packet:
                protocol = "UDP"
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                return
            
            self.packet_count += 1
            
            # Create flow key
            flow_key = (src_ip, src_port, dst_ip, dst_port)
            
            # Create or get flow
            if flow_key not in self.flows:
                self.flows[flow_key] = FlowData(src_ip, dst_ip, src_port, dst_port, protocol)
            
            flow = self.flows[flow_key]
            is_forward = (src_ip == flow.src_ip)
            flow.add_packet(is_forward, length, flags, window)
            
            # Predict once per flow when it reaches 3 packets
            if flow.packet_count == 3:
                self.flow_count += 1
                try:
                    self.sniffer_queue.put_nowait((flow_key, flow))
                except queue.Full:
                    pass  # Drop if queue is full
        
        except Exception:
            pass
    
    def sniffer_worker(self, duration):
        """Thread 1: Sniff packets and put completed flows in queue"""
        print("Sniffer worker started...")
        sniff(prn=self.process_packet, timeout=duration, store=False)
        self.sniffer_queue.put(None)  # Signal end of sniffing
        print("Sniffer worker stopped.")
    
    def predictor_worker(self):
        """Thread 2: Take flows from queue A, predict, put results in queue B"""
        print("Predictor worker started...")
        while not self.stop_event.is_set():
            try:
                item = self.sniffer_queue.get(timeout=1)
                if item is None:
                    break
                
                flow_key, flow = item
                self.analyze_flow(flow_key, flow)
                
            except queue.Empty:
                continue
            except Exception:
                pass
        
        self.predictor_queue.put(None)  # Signal end of predictions
        print("Predictor worker stopped.")
    
    def reporter_worker(self):
        """Thread 3: Take predictions from queue B, write to logs and console"""
        print("Reporter worker started...")
        while not self.stop_event.is_set():
            try:
                item = self.predictor_queue.get(timeout=1)
                if item is None:
                    break
                
                pred_data = item
                self.log_prediction(pred_data)
                self.check_minute_rotation()
                
            except queue.Empty:
                self.check_minute_rotation()  # Check even if no prediction
                continue
            except Exception:
                pass
        
        print("Reporter worker stopped.")
    
    def analyze_flow(self, flow_key, flow):
        """Extract features and predict (runs in predictor thread)"""
        try:
            features_dict = flow.get_features()
            feature_list = self.predictor.feature_list
            feature_values = [features_dict.get(f, 0) for f in feature_list]
            df = pd.DataFrame([feature_values], columns=feature_list)
            
            df_clean = self.preprocessor._validate_and_clean(df)
            result = self.predictor.predict_batch(df_clean)
            
            if result is not None and len(result) > 0:
                pred = result.iloc[0]
                
                src_ip, src_port, dst_ip, dst_port = flow_key
                prediction_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'src_ip': src_ip,
                    'src_port': src_port,
                    'dst_ip': dst_ip,
                    'dst_port': dst_port,
                    'protocol': flow.protocol,
                    'packets': flow.packet_count,
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
    
    def log_prediction(self, pred_data):
        """Write prediction to appropriate log file and console"""
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
        
        # Determine which log file(s) to write to
        log_files = []
        if alert_level == 'RED':
            log_files = ['red.log', 'full.log']
        elif alert_level == 'YELLOW':
            log_files = ['yellow.log', 'full.log']
        else:  # GREEN
            log_files = ['full.log']
        
        # Format log line
        log_line = (f"{timestamp} | {pred_data['src_ip']:15} | {pred_data['src_port']:5} | "
                   f"{pred_data['dst_ip']:15} | {pred_data['dst_port']:5} | "
                   f"{pred_data['protocol']:4} | {alert_level:6} | "
                   f"{pred_data['predicted_class']:15} | {pred_data['confidence']:>10.2%} | "
                   f"{pred_data.get('secondary_class', 'N/A'):15} | {pred_data.get('secondary_confidence', 0.0):>10.2%}\n")
        
        # Write to log files
        for log_file in log_files:
            log_path = self.minute_dir / log_file
            with open(log_path, 'a') as f:
                f.write(log_line)
        
        # Console output for RED/YELLOW
        if alert_level in ['RED', 'YELLOW']:
            print(f"[{alert_level}] {timestamp} | {pred_data['src_ip']}:{pred_data['src_port']} -> "
                  f"{pred_data['dst_ip']}:{pred_data['dst_port']} | "
                  f"{pred_data['predicted_class']} ({pred_data['confidence']:.1%})")
    
    def check_minute_rotation(self):
        """Check if we need to create a new minute folder and report"""
        current_minute = datetime.now().strftime("%Y%m%d_%H%M")
        
        if self.current_minute != current_minute:
            # Save previous minute report if exists
            if self.minute_dir and self.minute_predictions:
                self.save_minute_report()
            
            # Create new minute folder
            self.current_minute = current_minute
            self.minute_dir = self.session_dir / f"minute_{current_minute}"
            self.minute_dir.mkdir(parents=True, exist_ok=True)
            self.minute_predictions = []
            
            # Initialize log files
            for log_file in ['red.log', 'yellow.log', 'full.log']:
                log_path = self.minute_dir / log_file
                with open(log_path, 'w') as f:
                    f.write(f"Log started: {datetime.now().isoformat()}\n")
                    f.write("="*170 + "\n")
                    f.write("Timestamp | Src IP | Src Port | Dst IP | Dst Port | Protocol | Alert | Attack Type | Confidence | Secondary Type | Secondary Conf\n")
                    f.write("="*170 + "\n")
    
    def save_minute_report(self):
        """Save summary report for completed minute"""
        if not self.minute_predictions:
            return
        
        report_path = self.minute_dir / "report.txt"
        
        red = len([p for p in self.minute_predictions if p['alert_level'] == 'RED'])
        yellow = len([p for p in self.minute_predictions if p['alert_level'] == 'YELLOW'])
        green = len([p for p in self.minute_predictions if p['alert_level'] == 'GREEN'])
        total = len(self.minute_predictions)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MINUTE SUMMARY REPORT - {self.current_minute}\n")
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
    
    def start(self, duration=300):
        """Start monitoring with threaded pipeline"""
        self.start_time = time.time()
        
        # Create session directory
        results_dir = Path('results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = results_dir / f"realtime_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Default to 5 minutes if not specified
        if duration is None:
            duration = 300
        
        print(f"\n{'='*80}")
        print(f"REALTIME NETWORK THREAT MONITOR (THREADED)")
        print(f"{'='*80}")
        print(f"Duration: {duration} seconds ({duration/60:.1f} minutes)")
        print(f"Output: {self.session_dir}")
        print(f"New folder every minute: yes")
        print(f"{'='*80}\n")
        
        # Start the three worker threads
        sniffer_thread = threading.Thread(target=self.sniffer_worker, args=(duration,), daemon=False)
        predictor_thread = threading.Thread(target=self.predictor_worker, daemon=False)
        reporter_thread = threading.Thread(target=self.reporter_worker, daemon=False)
        
        try:
            sniffer_thread.start()
            predictor_thread.start()
            reporter_thread.start()
            
            # Wait for all threads to complete
            sniffer_thread.join()
            predictor_thread.join()
            self.stop_event.set()
            reporter_thread.join()
            
        except KeyboardInterrupt:
            print("\n[*] Stopping monitoring...")
            self.stop_event.set()
            sniffer_thread.join(timeout=2)
            predictor_thread.join(timeout=2)
            reporter_thread.join(timeout=2)
        
        # Finalize
        self.finalize_session()
    
    def finalize_session(self):
        """Generate final summary report"""
        elapsed = time.time() - self.start_time
        
        # Save last minute report
        if self.minute_predictions:
            self.save_minute_report()
        
        print(f"\n{'='*80}")
        print(f"SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"Duration:     {elapsed:.1f}s")
        print(f"Packets:      {self.packet_count}")
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
            f.write("MONITORING SESSION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {elapsed:.1f} seconds\n")
            f.write(f"Session Directory: {self.session_dir}\n\n")
            
            f.write(f"STATISTICS:\n")
            f.write(f"  Total Packets:   {self.packet_count}\n")
            f.write(f"  Total Flows:     {self.flow_count}\n")
            f.write(f"  Total Analyzed:  {len(self.predictions)}\n\n")
            
            f.write(f"ALERT BREAKDOWN:\n")
            f.write(f"  RED (Attacks):   {self.red_count}\n")
            f.write(f"  YELLOW (Warn):   {self.yellow_count}\n")
            f.write(f"  GREEN (Safe):    {self.green_count}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"  Per-minute folders: {self.session_dir}/minute_*\n")
            f.write(f"  Each minute contains:\n")
            f.write(f"    - red.log (RED alerts only)\n")
            f.write(f"    - yellow.log (YELLOW warnings only)\n")
            f.write(f"    - full.log (ALL flows)\n")
            f.write(f"    - report.txt (summary)\n")
            f.write("\n" + "="*80 + "\n")


def main():
    monitor = RealtimeMonitor()
    monitor.start(duration=10)


if __name__ == '__main__':
    main()
