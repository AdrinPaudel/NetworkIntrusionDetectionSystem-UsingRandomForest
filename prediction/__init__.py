"""
Prediction Pipeline - Network Intrusion Detection System
Inference module for making predictions on network traffic flows
"""

from .predictor import NetworkPredictor
from .predict_preprocess import PredictionPreprocessor
from .report_generator import ReportGenerator
from .threat_action_handler import ThreatActionHandler, initialize_handler, get_handler, handle_prediction

# Lazy imports for modules with optional dependencies
def __getattr__(name):
    if name == 'RealtimeMonitor':
        from .realtime_monitor_clean import RealtimeMonitor
        return RealtimeMonitor
    elif name == 'RealtimeSimulation':
        from .realtime_simulation import RealtimeSimulation
        return RealtimeSimulation
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'NetworkPredictor',
    'PredictionPreprocessor',
    'RealtimeMonitor',
    'RealtimeSimulation',
    'ReportGenerator',
    'ThreatActionHandler',
    'initialize_handler',
    'get_handler',
    'handle_prediction',
]

