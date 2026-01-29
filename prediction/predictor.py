"""
Prediction module for NIDS - Makes predictions on network traffic data
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.*')
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

from .threat_action_handler import handle_prediction

# Setup logging
logger = logging.getLogger(__name__)


class NetworkPredictor:
    """
    Loads trained model and makes predictions on network traffic data
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize predictor with trained model and preprocessing objects
        
        Args:
            model_dir: Path to directory containing trained model
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'trained_model'
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_list = None
        
        self._load_model_artifacts()
    
    def _load_model_artifacts(self):
        """Load model, scaler, label encoder from disk"""
        try:
            # Load model
            model_path = self.model_dir / 'random_forest_model.joblib'
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded: {self.model}")
            
            # Get feature list directly from the model (45 features)
            self.feature_list = list(self.model.feature_names_in_)
            logger.info(f"Loaded {len(self.feature_list)} features from model")
            
            # Load scaler (from data/preprocessed)
            preprocessed_dir = Path(__file__).parent.parent / 'data' / 'preprocessed'
            scaler_path = preprocessed_dir / 'scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
            
            # Load label encoder
            le_path = preprocessed_dir / 'label_encoder.joblib'
            self.label_encoder = joblib.load(le_path)
            logger.info(f"Label encoder loaded. Classes: {self.label_encoder.classes_}")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
            logger.info(f"Loaded {len(self.feature_list)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
    
    def predict_single(self, data_dict):
        """
        Predict on a single network flow entry
        
        Args:
            data_dict: Dictionary with feature names as keys and values
        
        Returns:
            dict: {
                'prediction': class label (str),
                'confidence': max probability,
                'probabilities': dict of all class probabilities,
                'is_attack': bool
            }
        """
        # Convert to dataframe
        df = pd.DataFrame([data_dict])
        
        # Validate and prepare
        df_prepared = self._prepare_data(df)
        
        # Predict
        pred_class_idx = self.model.predict(df_prepared)[0]
        pred_proba = self.model.predict_proba(df_prepared)[0]
        
        # Convert to labels
        pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]
        confidence = float(pred_proba[pred_class_idx])
        
        # Build probabilities dict
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(pred_proba)
        }
        
        is_attack = pred_class != 'Benign'
        
        result = {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'is_attack': is_attack,
            'all_probabilities': dict(sorted(probabilities.items(), 
                                            key=lambda x: x[1], reverse=True))
        }
        
        # Trigger threat action handler if attack detected
        if is_attack:
            handle_prediction({
                'Predicted_Class': pred_class,
                'Confidence': confidence,
                'Is_Attack': is_attack
            })
        
        return result
    
    def predict_batch(self, data_df):
        """
        Predict on batch of network flows (CSV data)
        
        Args:
            data_df: pandas DataFrame with network flow data
        
        Returns:
            DataFrame with predictions added
        """
        logger.debug(f"Predicting batch with shape: {data_df.shape}")
        
        # Prepare data
        df_prepared = self._prepare_data(data_df.copy())
        
        # Predict
        predictions = self.model.predict(df_prepared)
        probabilities = self.model.predict_proba(df_prepared)
        
        # Convert to labels
        pred_labels = self.label_encoder.inverse_transform(predictions)
        max_probs = probabilities.max(axis=1)
        
        # Add to dataframe
        result_df = data_df.copy()
        result_df['Predicted_Class'] = pred_labels
        result_df['Confidence'] = max_probs
        
        # Add all class probabilities
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'Prob_{class_name}'] = probabilities[:, i]
        
        # Add secondary prediction (2nd highest probability)
        secondary_classes = []
        secondary_confs = []
        for idx in range(len(result_df)):
            all_probs = [(class_name, probabilities[idx, i]) for i, class_name in enumerate(self.label_encoder.classes_)]
            all_probs.sort(key=lambda x: x[1], reverse=True)
            if len(all_probs) > 1:
                secondary_classes.append(all_probs[1][0])
                secondary_confs.append(all_probs[1][1])
            else:
                secondary_classes.append('')
                secondary_confs.append(0.0)
        
        result_df['Secondary_Class'] = secondary_classes
        result_df['Secondary_Confidence'] = secondary_confs
        
        # 3-Tier Alert System:
        # RED: Primary != Benign → Definite attack
        # YELLOW: Primary = Benign BUT Secondary != Benign + conf >= 25% → Warning for review
        # GREEN: Primary = Benign and nothing suspicious → Safe
        is_attacks = []
        alert_levels = []
        
        for idx, row in result_df.iterrows():
            primary = row['Predicted_Class']
            
            # RED: Attack only if Primary != Benign
            if primary != 'Benign':
                is_attacks.append(True)
                alert_levels.append('RED')
            else:
                # Primary is Benign - check secondary for warning
                is_attacks.append(False)  # Not an attack
                
                # Get second highest probability
                all_probs = [(class_name, row[f'Prob_{class_name}']) for class_name in self.label_encoder.classes_]
                all_probs.sort(key=lambda x: x[1], reverse=True)
                if len(all_probs) > 1:
                    secondary_class = all_probs[1][0]
                    secondary_conf = all_probs[1][1]
                    if secondary_class != 'Benign' and secondary_conf >= 0.25:
                        alert_levels.append('YELLOW')  # Warning - review recommended
                    else:
                        alert_levels.append('GREEN')  # Safe
                else:
                    alert_levels.append('GREEN')  # Safe
        
        result_df['Is_Attack'] = is_attacks
        result_df['Alert_Level'] = alert_levels
        
        # Trigger threat action handler for detected attacks
        for idx, row in result_df.iterrows():
            if row['Is_Attack']:
                handle_prediction({
                    'Predicted_Class': row['Predicted_Class'],
                    'Confidence': row['Confidence'],
                    'Is_Attack': True,
                    'Alert_Level': row['Alert_Level']
                })
        
        return result_df
    
    def _prepare_data(self, df):
        """
        Prepare user input data for prediction
        1. One-hot encode Protocol (creates 80 features total)
        2. Scale all 80 features
        3. Extract only the 45 features the model needs
        
        Args:
            df: pandas DataFrame with user data
        
        Returns:
            Scaled numpy array with 45 features ready for model
        """
        # One-hot encode Protocol if it exists and isn't already encoded
        # Create Protocol_0, Protocol_6, Protocol_17 (from training)
        if 'Protocol' in df.columns and not any('Protocol_' in col for col in df.columns):
            logger.debug("One-hot encoding Protocol column")
            df['Protocol_0'] = (df['Protocol'].astype(str) == '0').astype(int)
            df['Protocol_6'] = (df['Protocol'].astype(str) == '6').astype(int)
            df['Protocol_17'] = (df['Protocol'].astype(str) == '17').astype(int)
            # Drop original Protocol column
            df = df.drop(columns=['Protocol'])
        
        # Get feature names from scaler (80 features)
        scaler_features = list(self.scaler.feature_names_in_)
        
        # Create matrix with all 80 features for scaling
        X_for_scaling = pd.DataFrame(index=df.index)
        for feature in scaler_features:
            if feature in df.columns:
                X_for_scaling[feature] = df[feature]
            else:
                X_for_scaling[feature] = 0.0
        
        X_for_scaling = X_for_scaling.astype(float)
        
        # Scale all 80 features
        X_scaled_all = self.scaler.transform(X_for_scaling.values)
        
        # Convert back to DataFrame to select specific features
        X_scaled_df = pd.DataFrame(X_scaled_all, columns=scaler_features, index=df.index)
        
        # Now extract only the 45 features the model needs
        X_final = X_scaled_df[self.feature_list]
        logger.debug(f"Final feature matrix: {X_final.shape}")
        
        return X_final.values
    
    def get_feature_requirements(self):
        """
        Get list of features that model expects
        
        Returns:
            list: Feature names required for prediction
        """
        return self.feature_list
    
    def get_attack_types(self):
        """Get all attack types the model can classify"""
        return list(self.label_encoder.classes_)


if __name__ == "__main__":
    # Test the predictor
    predictor = NetworkPredictor()
    
    print("Features required:", len(predictor.get_feature_requirements()))
    print("Attack types:", predictor.get_attack_types())
    
    # Test single prediction with dummy data
    test_data = {feature: 0.0 for feature in predictor.get_feature_requirements()}
    test_data['Dst Port'] = 80  # Example: HTTP traffic
    
    result = predictor.predict_single(test_data)
    print("\nTest Prediction:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Is Attack: {result['is_attack']}")
