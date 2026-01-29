"""
Prediction Report Generator
Analyzes predictions and generates detailed reports for both batch and realtime modes
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports from predictions (batch and realtime modes)"""
    
    def __init__(self):
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.realtime_mode = False
        self.realtime_dir = None
        self.alerts = []
    
    def start_realtime(self):
        """Initialize realtime mode with session directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_dir = self.results_dir / f"realtime_{timestamp}"
        self.realtime_dir.mkdir(parents=True, exist_ok=True)
        self.realtime_mode = True
        self.alerts = []
        
        # Create empty log file
        attack_log = self.realtime_dir / "attacks.log"
        with open(attack_log, 'w') as f:
            f.write("Timestamp | Source IP | Source Port | Dest IP | Dest Port | Protocol | Alert Level | Primary | Primary Conf | Secondary | Secondary Conf | Reason\n")
            f.write("=" * 160 + "\n")
        
        return str(self.realtime_dir)
    
    def log_realtime_alert(self, alert):
        """Log a threat alert in realtime mode (RED and YELLOW only)"""
        print(f"[log_realtime_alert] Called with mode={self.realtime_mode}, dir={self.realtime_dir}")
        if not self.realtime_mode or not self.realtime_dir:
            print(f"[log_realtime_alert] Early return")
            return
        
        print(f"[log_realtime_alert] Adding alert. Before: {len(self.alerts)}")
        self.alerts.append(alert)
        print(f"[log_realtime_alert] Added alert. After: {len(self.alerts)}")
        
        # Only log RED and YELLOW alerts (not GREEN)
        if alert.get('alert_level') in ['RED', 'YELLOW']:
            attack_log = self.realtime_dir / "attacks.log"
            log_line = (f"{alert['timestamp']} | "
                       f"{alert['src_ip']} | {alert['src_port']} | "
                       f"{alert['dst_ip']} | {alert['dst_port']} | "
                       f"{alert['protocol']} | "
                       f"{alert['alert_level']} | "
                       f"{alert['primary']} | {alert['primary_conf']:.1%} | "
                       f"{alert['secondary']} | {alert['secondary_conf']:.1%} | "
                       f"{alert['reason']}\n")
            
            with open(attack_log, 'a') as f:
                f.write(log_line)
    
    def finalize_realtime(self):
        """Generate summary report at end of realtime session"""
        if not self.realtime_mode or not self.realtime_dir:
            return
        
        report_path = self.realtime_dir / "report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("REALTIME MONITORING SUMMARY\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Session Directory: {self.realtime_dir}\n\n")
            
            # Count alerts by level
            red_alerts = [a for a in self.alerts if a.get('alert_level') == 'RED']
            yellow_alerts = [a for a in self.alerts if a.get('alert_level') == 'YELLOW']
            green_alerts = [a for a in self.alerts if a.get('alert_level') == 'GREEN']
            
            f.write("ALERT SUMMARY:\n")
            f.write(f"  RED (Attacks):       {len(red_alerts)}\n")
            f.write(f"  YELLOW (Warnings):   {len(yellow_alerts)}\n")
            f.write(f"  GREEN (Safe):        {len(green_alerts)}\n")
            f.write(f"  Total Flows:         {len(self.alerts)}\n\n")
            
            # Only show details for RED and YELLOW
            logged_alerts = red_alerts + yellow_alerts
            if logged_alerts:
                f.write("THREAT DETAILS (RED & YELLOW):\n")
                f.write("-"*100 + "\n")
                for alert in logged_alerts:
                    f.write(f"\n[{alert['timestamp']}] {alert['src_ip']}:{alert['src_port']} → {alert['dst_ip']}:{alert['dst_port']}\n")
                    f.write(f"  Alert Level: {alert['alert_level']}\n")
                    f.write(f"  Protocol:    {alert['protocol']}\n")
                    f.write(f"  Primary:     {alert['primary']} ({alert['primary_conf']:.1%})\n")
                    f.write(f"  Secondary:   {alert['secondary']} ({alert['secondary_conf']:.1%})\n")
                    f.write(f"  Reason:      {alert['reason']}\n")
            
            f.write("\n" + "="*100 + "\n")
    
    def generate_reports(self, results_df, input_file):
        """
        Generate 4 separate log files + summary report
        Files: red.log, yellow.log, full.log, report.txt
        
        Args:
            results_df: DataFrame with predictions
            input_file: Original input CSV file name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(input_file).stem
        
        # Create batch-specific folder
        batch_dir = self.results_dir / f"batch_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. RED Log (attacks only)
        red_log_path = batch_dir / "red.log"
        self._generate_red_log(results_df, red_log_path)
        
        # 2. YELLOW Log (warnings only)
        yellow_log_path = batch_dir / "yellow.log"
        self._generate_yellow_log(results_df, yellow_log_path)
        
        # 3. FULL Log (all predictions)
        full_log_path = batch_dir / "full.log"
        self._generate_full_log(results_df, full_log_path)
        
        # 4. Summary Report (TXT)
        report_path = batch_dir / "report.txt"
        self._generate_report(results_df, report_path)
        
        return {
            'red': str(red_log_path),
            'yellow': str(yellow_log_path),
            'full': str(full_log_path),
            'report': str(report_path)
        }
    
    def _generate_attack_log(self, results_df, output_path):
        """Generate log file with RED and YELLOW alerts (deprecated - use separate RED/YELLOW logs)"""
        alerts = results_df[results_df['Alert_Level'].isin(['RED', 'YELLOW'])].copy()
        
        with open(output_path, 'w') as f:
            f.write(f"ATTACK & WARNING LOG - {datetime.now().isoformat()}\n")
            f.write(f"Total Logged: {len(alerts)} (RED: {(alerts['Alert_Level']=='RED').sum()}, YELLOW: {(alerts['Alert_Level']=='YELLOW').sum()})\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in alerts.iterrows():
                f.write(f"Row ID: {idx} | Alert Level: {row['Alert_Level']}\n")
                
                # Timestamp
                if 'Timestamp' in row:
                    f.write(f"  Timestamp:           {row['Timestamp']}\n")
                
                # Source and Destination IPs
                if 'Src IP' in row:
                    f.write(f"  Source IP:           {row['Src IP']}\n")
                if 'Dst IP' in row:
                    f.write(f"  Destination IP:      {row['Dst IP']}\n")
                
                # Predictions
                f.write(f"  Primary Prediction:   {row['Predicted_Class']:20} ({row['Confidence']:.2%})\n")
                
                # Get second highest probability
                prob_cols = [c for c in row.index if c.startswith('Prob_')]
                if prob_cols:
                    probs = {col.replace('Prob_', ''): row[col] for col in prob_cols}
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_probs) > 1:
                        f.write(f"  Secondary Prediction: {sorted_probs[1][0]:20} ({sorted_probs[1][1]:.2%})\n")
                
                # Port information
                if 'Dst Port' in row:
                    f.write(f"  Destination Port:    {int(row['Dst Port'])}\n")
                if 'Src Port' in row:
                    f.write(f"  Source Port:         {int(row['Src Port'])}\n")
                
                f.write("\n")
        
        logger.info(f"Attack log saved: {output_path}")
    
    def _generate_red_log(self, results_df, output_path):
        """Generate log file with RED alerts only (attacks)"""
        alerts = results_df[results_df['Alert_Level'] == 'RED'].copy()
        
        with open(output_path, 'w') as f:
            f.write(f"RED ALERT LOG (ATTACKS) - {datetime.now().isoformat()}\n")
            f.write(f"Total Logged: {len(alerts)} RED alerts\n")
            f.write("="*160 + "\n")
            f.write("Timestamp | Src IP | Src Port | Dst IP | Dst Port | Protocol | Attack Type | Confidence | Secondary Type | Secondary Conf\n")
            f.write("="*160 + "\n\n")
            
            for idx, row in alerts.iterrows():
                timestamp_str = row.get('Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                src_ip = row.get('Src IP', 'N/A')
                src_port = int(row.get('Src Port', 0)) if 'Src Port' in row else 0
                dst_ip = row.get('Dst IP', 'N/A')
                dst_port = int(row.get('Dst Port', 0)) if 'Dst Port' in row else 0
                protocol = row.get('Protocol', 'TCP')
                attack_type = row['Predicted_Class']
                confidence = row['Confidence']
                secondary_type = row.get('Secondary_Class', 'N/A')
                secondary_conf = row.get('Secondary_Confidence', 0.0)
                
                f.write(f"{timestamp_str} | {src_ip:15} | {src_port:5} | {dst_ip:15} | "
                       f"{dst_port:5} | {protocol:4} | {attack_type:15} | {confidence:>10.2%} | {secondary_type:15} | {secondary_conf:>10.2%}\n")
        
        logger.info(f"RED log saved: {output_path}")
    
    def _generate_yellow_log(self, results_df, output_path):
        """Generate log file with YELLOW alerts only (warnings/low confidence attacks)"""
        alerts = results_df[results_df['Alert_Level'] == 'YELLOW'].copy()
        
        with open(output_path, 'w') as f:
            f.write(f"YELLOW ALERT LOG (WARNINGS) - {datetime.now().isoformat()}\n")
            f.write(f"Total Logged: {len(alerts)} YELLOW alerts\n")
            f.write("="*160 + "\n")
            f.write("Timestamp | Src IP | Src Port | Dst IP | Dst Port | Protocol | Attack Type | Confidence | Secondary Type | Secondary Conf\n")
            f.write("="*160 + "\n\n")
            
            for idx, row in alerts.iterrows():
                timestamp_str = row.get('Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                src_ip = row.get('Src IP', 'N/A')
                src_port = int(row.get('Src Port', 0)) if 'Src Port' in row else 0
                dst_ip = row.get('Dst IP', 'N/A')
                dst_port = int(row.get('Dst Port', 0)) if 'Dst Port' in row else 0
                protocol = row.get('Protocol', 'TCP')
                attack_type = row['Predicted_Class']
                confidence = row['Confidence']
                secondary_type = row.get('Secondary_Class', 'N/A')
                secondary_conf = row.get('Secondary_Confidence', 0.0)
                
                f.write(f"{timestamp_str} | {src_ip:15} | {src_port:5} | {dst_ip:15} | "
                       f"{dst_port:5} | {protocol:4} | {attack_type:15} | {confidence:>10.2%} | {secondary_type:15} | {secondary_conf:>10.2%}\n")
        
        logger.info(f"YELLOW log saved: {output_path}")
    
    
    def _generate_full_log(self, results_df, output_path):
        """Generate log file with ALL predictions (RED, YELLOW, GREEN)"""
        with open(output_path, 'w') as f:
            f.write(f"FULL PREDICTION LOG (ALL FLOWS) - {datetime.now().isoformat()}\n")
            f.write(f"Total Rows: {len(results_df)}\n")
            f.write(f"RED: {(results_df['Alert_Level']=='RED').sum()}, "
                   f"YELLOW: {(results_df['Alert_Level']=='YELLOW').sum()}, "
                   f"GREEN: {(results_df['Alert_Level']=='GREEN').sum()}\n")
            f.write("="*170 + "\n")
            f.write("Timestamp | Src IP | Src Port | Dst IP | Dst Port | Protocol | Alert | Attack Type | Confidence | Secondary Type | Secondary Conf\n")
            f.write("="*170 + "\n\n")
            
            for idx, row in results_df.iterrows():
                timestamp_str = row.get('Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                src_ip = row.get('Src IP', 'N/A')
                src_port = int(row.get('Src Port', 0)) if 'Src Port' in row else 0
                dst_ip = row.get('Dst IP', 'N/A')
                dst_port = int(row.get('Dst Port', 0)) if 'Dst Port' in row else 0
                protocol = row.get('Protocol', 'TCP')
                alert = row['Alert_Level']
                attack_type = row['Predicted_Class']
                confidence = row['Confidence']
                secondary_type = row.get('Secondary_Class', 'N/A')
                secondary_conf = row.get('Secondary_Confidence', 0.0)
                
                f.write(f"{timestamp_str} | {src_ip:15} | {src_port:5} | {dst_ip:15} | "
                       f"{dst_port:5} | {protocol:4} | {alert:6} | {attack_type:15} | {confidence:>10.2%} | {secondary_type:15} | {secondary_conf:>10.2%}\n")
        
        logger.info(f"Full log saved: {output_path}")
    
    def _generate_report(self, results_df, output_path):
        """Generate summary report with statistics"""
        with open(output_path, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("BATCH PREDICTION SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Overall Statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            total = len(results_df)
            
            # Count by alert level
            red_count = (results_df['Alert_Level'] == 'RED').sum()
            yellow_count = (results_df['Alert_Level'] == 'YELLOW').sum()
            green_count = (results_df['Alert_Level'] == 'GREEN').sum()
            
            f.write(f"Total Samples Analyzed: {total}\n\n")
            f.write(f"RED (Attacks):       {red_count:5} ({red_count/total*100:5.1f}%)\n")
            f.write(f"YELLOW (Warnings):   {yellow_count:5} ({yellow_count/total*100:5.1f}%)\n")
            f.write(f"GREEN (Safe):        {green_count:5} ({green_count/total*100:5.1f}%)\n\n")
            
            # Attack Breakdown (RED alerts only)
            if red_count > 0:
                f.write("ATTACK BREAKDOWN (RED ALERTS)\n")
                f.write("-"*80 + "\n")
                
                # Get only RED alerts
                attack_rows = results_df[results_df['Alert_Level'] == 'RED'].copy()
                threat_types = []
                for idx, row in attack_rows.iterrows():
                    primary = row['Predicted_Class']
                    if primary != 'Benign':
                        threat_types.append(primary)
                    else:
                        # Get secondary threat
                        prob_cols = [c for c in row.index if c.startswith('Prob_')]
                        if prob_cols:
                            probs = {col.replace('Prob_', ''): row[col] for col in prob_cols}
                            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                            if len(sorted_probs) > 1:
                                threat_types.append(sorted_probs[1][0])
                            else:
                                threat_types.append(primary)
                        else:
                            threat_types.append(primary)
                
                attack_rows['Threat_Type'] = threat_types
                attack_breakdown = attack_rows['Threat_Type'].value_counts()
                for threat_type, count in attack_breakdown.items():
                    f.write(f"{threat_type:20} {count:5} ({count/red_count*100:5.1f}%)\n")
                f.write("\n")
            
            # Top Attacks by Confidence (RED only)
            f.write("TOP 10 HIGHEST CONFIDENCE ATTACKS (RED)\n")
            f.write("-"*80 + "\n")
            top_attacks = results_df[results_df['Alert_Level'] == 'RED'].copy()
            
            # Determine threat type for each attack
            threat_types = []
            threat_confs = []
            for idx, row in top_attacks.iterrows():
                primary = row['Predicted_Class']
                if primary != 'Benign':
                    threat_types.append(primary)
                    threat_confs.append(row['Confidence'])
                else:
                    # Get secondary threat
                    prob_cols = [c for c in row.index if c.startswith('Prob_')]
                    if prob_cols:
                        probs = {col.replace('Prob_', ''): row[col] for col in prob_cols}
                        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        if len(sorted_probs) > 1:
                            threat_types.append(sorted_probs[1][0])
                            threat_confs.append(sorted_probs[1][1])
                        else:
                            threat_types.append(primary)
                            threat_confs.append(row['Confidence'])
                    else:
                        threat_types.append(primary)
                        threat_confs.append(row['Confidence'])
            
            top_attacks['Threat_Type'] = threat_types
            top_attacks['Threat_Conf'] = threat_confs
            top_attacks_sorted = top_attacks.nlargest(10, 'Threat_Conf')
            
            for rank, (idx, row) in enumerate(top_attacks_sorted.iterrows(), 1):
                f.write(f"{rank:2}. Row {idx:5} - {row['Threat_Type']:15} ({row['Threat_Conf']:.2%})\n")
            f.write("\n")
            
            # Top Source IPs (if available)
            if 'Src IP' in results_df.columns:
                f.write("TOP 5 SOURCE IPS WITH ATTACKS\n")
                f.write("-"*80 + "\n")
                src_ips = results_df[results_df['Is_Attack']]['Src IP'].value_counts().head(5)
                for rank, (ip, count) in enumerate(src_ips.items(), 1):
                    f.write(f"{rank}. {ip:20} {count} attacks\n")
                f.write("\n")
            
            # Top Destination IPs (if available)
            if 'Dst IP' in results_df.columns:
                f.write("TOP 5 DESTINATION IPS UNDER ATTACK\n")
                f.write("-"*80 + "\n")
                dst_ips = results_df[results_df['Is_Attack']]['Dst IP'].value_counts().head(5)
                for rank, (ip, count) in enumerate(dst_ips.items(), 1):
                    f.write(f"{rank}. {ip:20} {count} attacks\n")
                f.write("\n")
            
            # Top Destination Ports (if available)
            if 'Dst Port' in results_df.columns:
                f.write("TOP 5 DESTINATION PORTS UNDER ATTACK\n")
                f.write("-"*80 + "\n")
                dst_ports = results_df[results_df['Is_Attack']]['Dst Port'].value_counts().head(5)
                for rank, (port, count) in enumerate(dst_ports.items(), 1):
                    f.write(f"{rank}. Port {int(port):6} {count} attacks\n")
                f.write("\n")
            
            # Confidence Distribution
            f.write("CONFIDENCE DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Confidence:     {results_df['Confidence'].mean():.2%}\n")
            f.write(f"Min Confidence:         {results_df['Confidence'].min():.2%}\n")
            f.write(f"Max Confidence:         {results_df['Confidence'].max():.2%}\n")
            f.write(f"Median Confidence:      {results_df['Confidence'].median():.2%}\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved: {output_path}")
