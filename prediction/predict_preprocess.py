"""
Input preprocessing and validation for prediction
- Maps user input to required features
- Validates data types and ranges
- Handles missing features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PredictionPreprocessor:
    """
    Preprocesses user input data for prediction
    - Maps user columns to model features
    - Validates and cleans data
    - Handles missing/unknown features
    """
    
    # Common aliases for features (users might call them differently)
    FEATURE_ALIASES = {
        'destination_port': 'Dst Port',
        'dest_port': 'Dst Port',
        'source_port': 'Src Port',
        'src_port': 'Src Port',
        'protocol': 'Protocol',
        'src_ip': 'Src IP',
        'dst_ip': 'Dst IP',
        'timestamp': 'Timestamp',
        'duration': 'Flow Duration',
        'flow_duration': 'Flow Duration',
        'packets_forward': 'Tot Fwd Pkts',
        'packets_backward': 'Tot Bwd Pkts',
        'bytes_forward': 'TotLen Fwd Pkts',
        'bytes_backward': 'TotLen Bwd Pkts',
        'ack_flag': 'ACK Flag Cnt',
        'rst_flag': 'RST Flag Cnt',
    }
    
    # Expected data types for features
    FEATURE_DTYPES = {
        'Dst Port': 'numeric',
        'Src Port': 'numeric',
        'Protocol': 'numeric',
        'Flow Duration': 'numeric',
        'Tot Fwd Pkts': 'numeric',
        'Tot Bwd Pkts': 'numeric',
        'TotLen Fwd Pkts': 'numeric',
        'TotLen Bwd Pkts': 'numeric',
    }
    
    def __init__(self):
        """Initialize preprocessor"""
        self.processed_count = 0
        self.error_count = 0
    
    def load_csv(self, csv_path):
        """
        Load CSV file and preprocess
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            pandas DataFrame ready for prediction
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if Label column exists (for reference, not required)
            has_label = 'Label' in df.columns
            if has_label:
                logger.info(f"Label column found - will preserve for reference")
            else:
                logger.info(f"No Label column - user-provided data or real predictions")
            
            # Map column names
            df = self._map_column_names(df)
            
            # Validate and clean
            df = self._validate_and_clean(df)
            
            logger.info(f"Preprocessed: {len(df)} valid rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def preprocess_single_entry(self, data_dict):
        """
        Preprocess a single network entry
        
        Args:
            data_dict: Dictionary with feature-value pairs
        
        Returns:
            dict: Validated and mapped data
        """
        try:
            # Map column names
            mapped_dict = {}
            for key, value in data_dict.items():
                mapped_key = self._map_column_name(key)
                mapped_dict[mapped_key] = value
            
            # Validate data types
            mapped_dict = self._validate_single_entry(mapped_dict)
            
            return mapped_dict
            
        except Exception as e:
            logger.error(f"Failed to preprocess entry: {e}")
            raise
    
    def _map_column_names(self, df):
        """
        Map user column names to model feature names
        
        Args:
            df: pandas DataFrame with user columns
        
        Returns:
            DataFrame with mapped columns
        """
        rename_map = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Check if it's in aliases
            if col_lower in self.FEATURE_ALIASES:
                rename_map[col] = self.FEATURE_ALIASES[col_lower]
            # Check if it's already a correct feature name - keep as is
            elif col in self.FEATURE_ALIASES.values():
                continue  # Keep as is
            else:
                # For raw CSV data, the column names are already correct
                # Just keep them as is - don't try to fuzzy match
                continue
        
        df = df.rename(columns=rename_map)
        return df
    
    def _map_column_name(self, col_name):
        """Map single column name"""
        col_lower = col_name.lower().strip()
        
        if col_lower in self.FEATURE_ALIASES:
            return self.FEATURE_ALIASES[col_lower]
        elif col_name in self.FEATURE_ALIASES.values():
            return col_name
        else:
            mapped = self._fuzzy_match_feature(col_lower)
            return mapped if mapped else col_name
    
    def _fuzzy_match_feature(self, col_name):
        """
        Try to fuzzy match column name to features
        
        Args:
            col_name: Column name to match
        
        Returns:
            str or None: Matched feature name
        """
        col_name_lower = col_name.lower()
        
        # Check if any feature name contains the user's column
        all_features = list(self.FEATURE_ALIASES.values())
        for feature in all_features:
            if col_name_lower in feature.lower() or feature.lower() in col_name_lower:
                return feature
        
        return None
    
    def _validate_and_clean(self, df):
        """
        Validate data types and clean values
        
        Args:
            df: pandas DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Drop rows with all NaN
        df = df.dropna(how='all')
        
        # Convert numeric columns
        for col in df.columns:
            if col in self.FEATURE_DTYPES and self.FEATURE_DTYPES[col] == 'numeric':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Fill NaN with median (not 0) for numeric columns
        # This is more statistically sound than filling with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If median is NaN, use 0 as fallback
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(median_val)
                logger.info(f"Filled NaN in '{col}' with median: {median_val}")
        
        # Remove any infinite values (replace with median or 0)
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                median_val = df[col][~np.isinf(df[col])].median()
                if pd.isna(median_val):
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
                else:
                    df[col] = df[col].replace([np.inf, -np.inf], median_val)
                logger.info(f"Replaced infinite values in '{col}'")
        
        return df
    
    def _validate_single_entry(self, data_dict):
        """Validate single entry"""
        validated = {}
        
        for key, value in data_dict.items():
            # Try to convert to numeric if expected to be numeric
            if key in self.FEATURE_DTYPES and self.FEATURE_DTYPES[key] == 'numeric':
                try:
                    validated[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {key}={value} to numeric, using 0")
                    validated[key] = 0.0
            else:
                validated[key] = value
        
        return validated
    
    def get_feature_template(self):
        """
        Get a template dict with all expected features
        
        Returns:
            dict: Empty template with all feature names
        """
        # Get list from predictor
        from .predictor import NetworkPredictor
        predictor = NetworkPredictor()
        features = predictor.get_feature_requirements()
        
        return {feature: 0.0 for feature in features}
    
    def validate_column_coverage(self, df, required_features):
        """
        Check how many required features are present in data
        
        Args:
            df: pandas DataFrame
            required_features: list of required features
        
        Returns:
            dict: Coverage statistics
        """
        present = [f for f in required_features if f in df.columns]
        missing = [f for f in required_features if f not in df.columns]
        
        coverage = len(present) / len(required_features) * 100
        
        return {
            'total_required': len(required_features),
            'present': len(present),
            'missing': len(missing),
            'coverage_percent': coverage,
            'missing_features': missing[:10]  # Show first 10
        }


if __name__ == "__main__":
    preprocessor = PredictionPreprocessor()
    
    # Test mapping
    test_data = {
        'destination_port': 443,
        'protocol': 6,
        'bytes_forward': 1024,
        'Dst Port': 80,  # Already correct
    }
    
    result = preprocessor.preprocess_single_entry(test_data)
    print("Preprocessed single entry:")
    print(result)
