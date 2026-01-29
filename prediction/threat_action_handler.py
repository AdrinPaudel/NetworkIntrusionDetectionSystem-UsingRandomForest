"""
Threat Action Handler
Triggered when attacks are detected during prediction
Handles alert response and actions (currently: notification/logging)
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ThreatActionHandler:
    """
    Handles threat/attack detection responses
    
    Alert Levels:
    - RED: confidence >= 50% (Critical)
    - YELLOW: 25% <= confidence < 50% (Warning)
    - GREEN: confidence < 25% (Low/Benign - no action)
    """
    
    # Alert level thresholds
    RED_THRESHOLD = 0.50      # >= 50% confidence
    YELLOW_THRESHOLD = 0.25   # >= 25% confidence
    
    def __init__(self, enable_logging=True, log_file=None):
        """
        Initialize threat action handler
        
        Args:
            enable_logging: Whether to log actions to file
            log_file: Path to log file (optional)
        """
        self.enable_logging = enable_logging
        self.log_file = log_file
        self.action_count = 0
        self.red_alerts = 0
        self.yellow_alerts = 0
    
    def determine_alert_level(self, confidence):
        """
        Determine alert level based on confidence
        
        Args:
            confidence: Prediction confidence (0.0 - 1.0)
        
        Returns:
            str: 'RED', 'YELLOW', or 'GREEN'
        """
        if confidence >= self.RED_THRESHOLD:
            return 'RED'
        elif confidence >= self.YELLOW_THRESHOLD:
            return 'YELLOW'
        else:
            return 'GREEN'
    
    def handle_threat(self, prediction_result):
        """
        Handle detected threat/attack
        
        Args:
            prediction_result: Dict with prediction details
                {
                    'Predicted_Class': str (attack type),
                    'Confidence': float (0.0-1.0),
                    'Is_Attack': bool,
                    'Timestamp': str (optional),
                    'Source': str (optional),
                    'Destination': str (optional)
                }
        """
        # Extract prediction details
        attack_type = prediction_result.get('Predicted_Class', 'Unknown')
        confidence = prediction_result.get('Confidence', 0.0)
        is_attack = prediction_result.get('Is_Attack', False)
        timestamp = prediction_result.get('Timestamp', datetime.now().isoformat())
        
        # Benign traffic - no action needed
        if not is_attack or attack_type == 'Benign':
            return None
        
        # Determine alert level
        alert_level = self.determine_alert_level(confidence)
        
        # If GREEN level (low confidence) - no action
        if alert_level == 'GREEN':
            return None
        
        # Execute action based on alert level
        if alert_level == 'RED':
            self._handle_critical_threat(prediction_result, alert_level)
            self.red_alerts += 1
        elif alert_level == 'YELLOW':
            self._handle_warning_threat(prediction_result, alert_level)
            self.yellow_alerts += 1
        
        self.action_count += 1
        
        return {
            'timestamp': timestamp,
            'alert_level': alert_level,
            'attack_type': attack_type,
            'confidence': confidence
        }
    
    def _handle_critical_threat(self, prediction_result, alert_level):
        """
        Handle CRITICAL threat (RED alert)
        
        Args:
            prediction_result: Prediction details
            alert_level: 'RED'
        """
        attack_type = prediction_result.get('Predicted_Class', 'Unknown')
        confidence = prediction_result.get('Confidence', 0.0)
        
        # Action message
        message = f"🔴 CRITICAL THREAT DETECTED: {attack_type} (Confidence: {confidence:.1%})"
        
        # Output to console
        print(message)
        logger.critical(message)
        
        # Log to file if enabled
        if self.enable_logging:
            self._log_action(alert_level, prediction_result)
        
        # TODO: Future actions for RED alerts
        # - Block IP address
        # - Isolate connection
        # - Send alert to SOC
        # - Trigger automated response
        # - etc.
    
    def _handle_warning_threat(self, prediction_result, alert_level):
        """
        Handle WARNING threat (YELLOW alert)
        
        Args:
            prediction_result: Prediction details
            alert_level: 'YELLOW'
        """
        attack_type = prediction_result.get('Predicted_Class', 'Unknown')
        confidence = prediction_result.get('Confidence', 0.0)
        
        # Action message
        message = f"🟡 WARNING: Suspicious activity detected - {attack_type} (Confidence: {confidence:.1%})"
        
        # Output to console
        print(message)
        logger.warning(message)
        
        # Log to file if enabled
        if self.enable_logging:
            self._log_action(alert_level, prediction_result)
        
        # TODO: Future actions for YELLOW alerts
        # - Monitor connection closely
        # - Increase logging level
        # - Alert analyst for review
        # - Prepare isolation if needed
        # - etc.
    
    def _log_action(self, alert_level, prediction_result):
        """
        Log threat action to file
        
        Args:
            alert_level: 'RED' or 'YELLOW'
            prediction_result: Prediction details
        """
        try:
            timestamp = prediction_result.get('Timestamp', datetime.now().isoformat())
            attack_type = prediction_result.get('Predicted_Class', 'Unknown')
            confidence = prediction_result.get('Confidence', 0.0)
            
            log_message = (
                f"[{timestamp}] [{alert_level}] "
                f"Attack: {attack_type} | "
                f"Confidence: {confidence:.1%} | "
                f"Source: {prediction_result.get('Source', 'N/A')} | "
                f"Destination: {prediction_result.get('Destination', 'N/A')}\n"
            )
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_message)
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
    
    def get_statistics(self):
        """
        Get threat action statistics
        
        Returns:
            dict: Statistics on detected threats
        """
        return {
            'total_actions': self.action_count,
            'red_alerts': self.red_alerts,
            'yellow_alerts': self.yellow_alerts,
            'critical_percentage': (self.red_alerts / self.action_count * 100) if self.action_count > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.action_count = 0
        self.red_alerts = 0
        self.yellow_alerts = 0


# Global instance for easy access
_threat_handler = None


def initialize_handler(enable_logging=True, log_file=None):
    """
    Initialize global threat handler
    
    Args:
        enable_logging: Whether to log actions
        log_file: Path to log file
    
    Returns:
        ThreatActionHandler instance
    """
    global _threat_handler
    _threat_handler = ThreatActionHandler(enable_logging=enable_logging, log_file=log_file)
    return _threat_handler


def get_handler():
    """Get global threat handler instance"""
    global _threat_handler
    if _threat_handler is None:
        _threat_handler = ThreatActionHandler()
    return _threat_handler


def handle_prediction(prediction_result):
    """
    Convenience function to handle threat from prediction
    
    Args:
        prediction_result: Prediction result dict
    
    Returns:
        dict: Action taken (if any)
    """
    handler = get_handler()
    return handler.handle_threat(prediction_result)
