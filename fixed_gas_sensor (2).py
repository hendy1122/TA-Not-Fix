#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - ADAPTIVE VERSION 4.3
FIXED: PPM Range Mismatch + Adaptive Feature Scaling + Enhanced Detection
Solution untuk training data range berbeda dengan current readings
"""

import time
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import logging
import math
import glob
import os
import warnings
from pathlib import Path

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Library untuk ADC dan GPIO
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except ImportError:
    print("Installing required libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "adafruit-circuitpython-ads1x15"])
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn

# Machine Learning libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    import joblib
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    import joblib

class AdaptiveFeatureProcessor:
    """Adaptive Feature Processor - Solusi untuk PPM range mismatch"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Training data statistics (akan di-load dari metadata)
        self.training_stats = {
            'TGS2600_ppm': {'min': 0, 'max': 100, 'mean': 20, 'std': 15},
            'TGS2602_ppm': {'min': 0, 'max': 100, 'mean': 18, 'std': 12},
            'TGS2610_ppm': {'min': 0, 'max': 100, 'mean': 25, 'std': 18},
            'TGS2600_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15},
            'TGS2602_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15},
            'TGS2610_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15}
        }
        
        # Current data statistics (auto-updated)
        self.current_stats = {}
        
        # Adaptive scaling factors
        self.ppm_scaling_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0, 
            'TGS2610': 1.0
        }
        
        # Feature transformation methods
        self.transformation_methods = {
            'linear_scale': self.linear_scale_transform,
            'log_scale': self.log_scale_transform,
            'robust_scale': self.robust_scale_transform,
            'percentile_scale': self.percentile_scale_transform,
            'adaptive_normalize': self.adaptive_normalize_transform
        }
        
        # Current best transformation method
        self.best_transform_method = 'adaptive_normalize'
        
        # Multi-scale features
        self.enable_multi_scale = True
        
        self.load_training_stats()
    
    def load_training_stats(self):
        """Load training data statistics"""
        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            if 'training_stats' in metadata:
                self.training_stats = metadata['training_stats']
                self.logger.info("Training statistics loaded from metadata")
            else:
                self.logger.info("Using default training statistics")
                
        except FileNotFoundError:
            self.logger.info("No model metadata found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading training stats: {e}")
    
    def update_current_stats(self, readings_history):
        """Update current statistics from recent readings"""
        if len(readings_history) < 10:
            return  # Need minimum samples
        
        # Calculate statistics for recent readings
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_values = [r[sensor]['ppm'] for r in readings_history if r[sensor]['ppm'] > 0]
            voltage_values = [r[sensor]['raw_voltage'] for r in readings_history]
            
            if ppm_values:
                self.current_stats[f'{sensor}_ppm'] = {
                    'min': min(ppm_values),
                    'max': max(ppm_values),
                    'mean': np.mean(ppm_values),
                    'std': np.std(ppm_values)
                }
            
            if voltage_values:
                self.current_stats[f'{sensor}_voltage'] = {
                    'min': min(voltage_values),
                    'max': max(voltage_values),
                    'mean': np.mean(voltage_values),
                    'std': np.std(voltage_values)
                }
        
        # Update scaling factors
        self.calculate_adaptive_scaling_factors()
    
    def calculate_adaptive_scaling_factors(self):
        """Calculate adaptive scaling factors"""
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_key = f'{sensor}_ppm'
            
            if ppm_key in self.current_stats and ppm_key in self.training_stats:
                current_max = self.current_stats[ppm_key]['max']
                training_max = self.training_stats[ppm_key]['max']
                
                if current_max > 0 and training_max > 0:
                    # Calculate scaling factor to bring current range to training range
                    scale_factor = training_max / current_max
                    
                    # Apply smoothing to avoid extreme scaling
                    scale_factor = max(0.1, min(10.0, scale_factor))
                    
                    self.ppm_scaling_factors[sensor] = scale_factor
                    self.logger.info(f"{sensor} PPM scaling factor: {scale_factor:.3f}")
    
    def linear_scale_transform(self, value, feature_name):
        """Linear scaling transformation"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            return value * self.ppm_scaling_factors.get(sensor, 1.0)
        return value
    
    def log_scale_transform(self, value, feature_name):
        """Log scaling transformation"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            return np.log1p(scaled_value)  # log(1 + x) to handle 0 values
        return value
    
    def robust_scale_transform(self, value, feature_name):
        """Robust scaling using percentiles"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Use training stats for normalization
            if feature_name in self.training_stats:
                training_median = self.training_stats[feature_name].get('mean', 20)
                training_iqr = self.training_stats[feature_name].get('std', 10) * 1.349  # Approximate IQR
                
                return (scaled_value - training_median) / (training_iqr + 1e-8)
        
        return value
    
    def percentile_scale_transform(self, value, feature_name):
        """Percentile-based scaling"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Map to percentile of training distribution
            if feature_name in self.training_stats:
                training_max = self.training_stats[feature_name]['max']
                percentile = min(100, (scaled_value / training_max) * 100)
                return percentile / 100.0  # Normalize to 0-1
        
        return value
    
    def adaptive_normalize_transform(self, value, feature_name):
        """Advanced adaptive normalization"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            
            # Multi-step transformation
            # Step 1: Apply adaptive scaling
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Step 2: Apply training range normalization
            if feature_name in self.training_stats:
                training_stats = self.training_stats[feature_name]
                training_range = training_stats['max'] - training_stats['min']
                
                if training_range > 0:
                    normalized = (scaled_value - training_stats['min']) / training_range
                    # Clip to reasonable range but allow some extrapolation
                    normalized = np.clip(normalized, -0.5, 1.5)
                    return normalized
            
            # Fallback normalization
            return min(1.0, scaled_value / 100.0)
        
        # Voltage normalization
        elif feature_name.endswith('_voltage'):
            if feature_name in self.training_stats:
                training_stats = self.training_stats[feature_name]
                training_range = training_stats['max'] - training_stats['min']
                
                if training_range > 0:
                    normalized = (value - training_stats['min']) / training_range
                    return np.clip(normalized, 0.0, 1.0)
        
        return value
    
    def extract_multi_scale_features(self, readings):
        """Extract multi-scale features for better detection"""
        multi_scale_features = {}
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            sensor_data = readings[sensor]
            
            # Original features
            multi_scale_features[f'{sensor}_voltage'] = sensor_data['raw_voltage']
            multi_scale_features[f'{sensor}_ppm'] = sensor_data['ppm']
            
            if self.enable_multi_scale:
                # Ratio features
                base_voltage = 1.6
                voltage_ratio = sensor_data['raw_voltage'] / base_voltage
                multi_scale_features[f'{sensor}_voltage_ratio'] = voltage_ratio
                
                # Log features (better for wide range data)
                log_ppm = np.log1p(sensor_data['ppm'])
                multi_scale_features[f'{sensor}_log_ppm'] = log_ppm
                
                # Normalized features
                if sensor_data['ppm'] > 0:
                    # PPM per voltage drop
                    voltage_drop = base_voltage - sensor_data['raw_voltage']
                    if voltage_drop > 0.001:
                        ppm_per_voltage = sensor_data['ppm'] / voltage_drop
                        multi_scale_features[f'{sensor}_ppm_efficiency'] = min(1000, ppm_per_voltage)
                    else:
                        multi_scale_features[f'{sensor}_ppm_efficiency'] = 0
                else:
                    multi_scale_features[f'{sensor}_ppm_efficiency'] = 0
                
                # Sensitivity-aware features
                sensitivity_multiplier = sensor_data.get('sensitivity_multiplier', 1.0)
                if sensitivity_multiplier > 1:
                    raw_ppm = sensor_data['ppm'] / sensitivity_multiplier
                    multi_scale_features[f'{sensor}_raw_ppm'] = raw_ppm
                else:
                    multi_scale_features[f'{sensor}_raw_ppm'] = sensor_data['ppm']
        
        return multi_scale_features
    
    def transform_features(self, feature_vector, feature_names):
        """Transform feature vector using adaptive methods"""
        transformed_features = []
        
        for i, (value, feature_name) in enumerate(zip(feature_vector, feature_names)):
            # Apply selected transformation method
            transform_func = self.transformation_methods[self.best_transform_method]
            transformed_value = transform_func(value, feature_name)
            
            # Handle NaN and infinite values
            if np.isnan(transformed_value) or np.isinf(transformed_value):
                transformed_value = 0.0
            
            transformed_features.append(transformed_value)
        
        return transformed_features
    
    def evaluate_transformation_quality(self, readings_history):
        """Evaluate which transformation method works best"""
        if len(readings_history) < 20:
            return
        
        # Simple heuristic: test different methods and see which gives most stable features
        method_scores = {}
        
        for method_name, transform_func in self.transformation_methods.items():
            feature_stability = []
            
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm_values = [r[sensor]['ppm'] for r in readings_history[-20:]]
                transformed_values = [transform_func(ppm, f'{sensor}_ppm') for ppm in ppm_values]
                
                # Calculate coefficient of variation (stability metric)
                if np.mean(transformed_values) > 0:
                    cv = np.std(transformed_values) / np.mean(transformed_values)
                    feature_stability.append(cv)
            
            if feature_stability:
                method_scores[method_name] = np.mean(feature_stability)
        
        # Select method with lowest CV (most stable)
        if method_scores:
            best_method = min(method_scores.keys(), key=lambda x: method_scores[x])
            if best_method != self.best_transform_method:
                self.best_transform_method = best_method
                self.logger.info(f"Switched to transformation method: {best_method}")
    
    def get_adaptive_confidence_threshold(self, predicted_gas, base_confidence):
        """Get adaptive confidence threshold based on gas type and current conditions"""
        # Base thresholds per gas type (lower for difficult gases)
        base_thresholds = {
            'normal': 0.3,
            'alcohol': 0.4,
            'pertalite': 0.4,
            'toluene': 0.2,    # Lower threshold for harder to detect gases
            'ammonia': 0.2,
            'dexlite': 0.3,
            'butane': 0.25,
            'propane': 0.25
        }
        
        base_threshold = base_thresholds.get(predicted_gas, 0.35)
        
        # Adaptive adjustment based on current PPM levels
        # If PPM levels are high (indicating strong signal), can use higher threshold
        # If PPM levels are low, use lower threshold
        
        return base_threshold
    
    def save_adaptive_config(self):
        """Save adaptive configuration"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'version': 'adaptive_v4.3',
            'ppm_scaling_factors': self.ppm_scaling_factors,
            'current_stats': self.current_stats,
            'best_transform_method': self.best_transform_method,
            'enable_multi_scale': self.enable_multi_scale
        }
        
        try:
            with open('adaptive_feature_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info("Adaptive feature configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving adaptive config: {e}")
    
    def load_adaptive_config(self):
        """Load adaptive configuration"""
        try:
            with open('adaptive_feature_config.json', 'r') as f:
                config = json.load(f)
            
            self.ppm_scaling_factors = config.get('ppm_scaling_factors', self.ppm_scaling_factors)
            self.current_stats = config.get('current_stats', {})
            self.best_transform_method = config.get('best_transform_method', 'adaptive_normalize')
            self.enable_multi_scale = config.get('enable_multi_scale', True)
            
            self.logger.info("Adaptive feature configuration loaded")
            
        except FileNotFoundError:
            self.logger.info("No adaptive config found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading adaptive config: {e}")

class EnhancedPredictionEngine:
    """Enhanced Prediction Engine with confidence boosting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.recent_predictions = []
        self.prediction_history = {}
        
        # Confidence boosting parameters
        self.consistency_boost = True
        self.temporal_smoothing = True
        self.multi_evidence_fusion = True
        
        # Gas-specific detection patterns
        self.gas_signatures = {
            'alcohol': {
                'primary_sensors': ['TGS2600', 'TGS2602'],
                'voltage_pattern': 'decrease',
                'ppm_ratio_range': (0.5, 2.0),
                'response_speed': 'fast'
            },
            'pertalite': {
                'primary_sensors': ['TGS2602', 'TGS2610'],
                'voltage_pattern': 'decrease',
                'ppm_ratio_range': (0.3, 1.5),
                'response_speed': 'medium'
            },
            'toluene': {
                'primary_sensors': ['TGS2602'],
                'voltage_pattern': 'strong_decrease',
                'ppm_ratio_range': (1.0, 3.0),
                'response_speed': 'fast'
            },
            'ammonia': {
                'primary_sensors': ['TGS2602'],
                'voltage_pattern': 'moderate_decrease',
                'ppm_ratio_range': (0.8, 2.5),
                'response_speed': 'slow'
            }
        }
    
    def analyze_sensor_patterns(self, readings):
        """Analyze sensor response patterns for gas detection"""
        pattern_scores = {}
        
        base_voltage = 1.6
        
        for gas_type, signature in self.gas_signatures.items():
            score = 0.0
            evidence_count = 0
            
            # Check primary sensors
            for sensor in signature['primary_sensors']:
                if sensor in readings:
                    sensor_data = readings[sensor]
                    
                    # Voltage pattern check
                    voltage_drop = base_voltage - sensor_data['raw_voltage']
                    if voltage_drop > 0.02:  # Minimum response
                        if signature['voltage_pattern'] == 'strong_decrease' and voltage_drop > 0.2:
                            score += 2.0
                        elif signature['voltage_pattern'] == 'decrease' and voltage_drop > 0.05:
                            score += 1.5
                        elif signature['voltage_pattern'] == 'moderate_decrease' and voltage_drop > 0.1:
                            score += 1.0
                        evidence_count += 1
                    
                    # PPM response check
                    if sensor_data['ppm'] > 10:  # Minimum PPM for detection
                        score += 1.0
                        evidence_count += 1
            
            # Normalize score by evidence count
            if evidence_count > 0:
                pattern_scores[gas_type] = score / evidence_count
            else:
                pattern_scores[gas_type] = 0.0
        
        return pattern_scores
    
    def boost_confidence_with_patterns(self, prediction, base_confidence, readings):
        """Boost confidence using sensor pattern analysis"""
        pattern_scores = self.analyze_sensor_patterns(readings)
        
        # Get pattern score for predicted gas
        pattern_score = pattern_scores.get(prediction, 0.0)
        
        # Calculate confidence boost
        if pattern_score > 1.0:
            confidence_boost = min(0.3, pattern_score * 0.15)  # Max boost of 0.3
            boosted_confidence = min(1.0, base_confidence + confidence_boost)
            
            self.logger.debug(f"Confidence boosted for {prediction}: {base_confidence:.3f} -> {boosted_confidence:.3f}")
            return boosted_confidence
        
        return base_confidence
    
    def apply_temporal_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to reduce prediction jitter"""
        if not self.temporal_smoothing:
            return prediction, confidence
        
        # Add to recent predictions
        self.recent_predictions.append((prediction, confidence, time.time()))
        
        # Keep only recent predictions (last 30 seconds)
        current_time = time.time()
        self.recent_predictions = [
            (p, c, t) for p, c, t in self.recent_predictions 
            if current_time - t < 30
        ]
        
        if len(self.recent_predictions) < 3:
            return prediction, confidence
        
        # Count recent predictions
        recent_counts = {}
        total_confidence = 0
        
        for p, c, t in self.recent_predictions[-5:]:  # Last 5 predictions
            recent_counts[p] = recent_counts.get(p, 0) + c
            total_confidence += c
        
        # Find most consistent prediction
        if total_confidence > 0:
            most_consistent = max(recent_counts.keys(), key=lambda x: recent_counts[x])
            consistency_score = recent_counts[most_consistent] / total_confidence
            
            # If current prediction matches most consistent and has good consistency
            if prediction == most_consistent and consistency_score > 0.6:
                smoothed_confidence = min(1.0, confidence * (1 + consistency_score * 0.2))
                return prediction, smoothed_confidence
        
        return prediction, confidence
    
    def fuse_multi_evidence(self, ml_prediction, ml_confidence, readings):
        """Fuse ML prediction with rule-based evidence"""
        if not self.multi_evidence_fusion:
            return ml_prediction, ml_confidence
        
        # Get pattern-based evidence
        pattern_scores = self.analyze_sensor_patterns(readings)
        
        # Simple rule-based predictions
        rule_predictions = {}
        
        # Rule 1: High PPM on specific sensors
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            if sensor in readings and readings[sensor]['ppm'] > 50:
                if sensor == 'TGS2600' and readings[sensor]['ppm'] > 100:
                    rule_predictions['alcohol'] = rule_predictions.get('alcohol', 0) + 0.3
                elif sensor == 'TGS2602' and readings[sensor]['ppm'] > 80:
                    rule_predictions['toluene'] = rule_predictions.get('toluene', 0) + 0.4
                    rule_predictions['pertalite'] = rule_predictions.get('pertalite', 0) + 0.3
                elif sensor == 'TGS2610' and readings[sensor]['ppm'] > 60:
                    rule_predictions['pertalite'] = rule_predictions.get('pertalite', 0) + 0.3
        
        # Rule 2: Voltage patterns
        base_voltage = 1.6
        total_voltage_drop = sum([
            max(0, base_voltage - readings[sensor]['raw_voltage']) 
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610'] if sensor in readings
        ])
        
        if total_voltage_drop > 0.3:
            rule_predictions['alcohol'] = rule_predictions.get('alcohol', 0) + 0.2
        
        # Fusion: ML prediction with rule-based evidence
        if rule_predictions:
            best_rule_gas = max(rule_predictions.keys(), key=lambda x: rule_predictions[x])
            best_rule_score = rule_predictions[best_rule_gas]
            
            # If rule-based prediction agrees with ML prediction
            if best_rule_gas == ml_prediction and best_rule_score > 0.3:
                fused_confidence = min(1.0, ml_confidence + best_rule_score * 0.3)
                return ml_prediction, fused_confidence
            
            # If rule-based prediction is strong and ML confidence is low
            elif best_rule_score > 0.5 and ml_confidence < 0.4:
                return best_rule_gas, best_rule_score
        
        return ml_prediction, ml_confidence

# [Previous classes remain the same: AdvancedSensitivityManager, EmergencyPPMCalculator, SmartDriftManager, MonitoringDataCollector]

class AdvancedSensitivityManager:
    """Advanced Sensitivity Manager - Solusi untuk response rendah dan optimization"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Sensitivity profiles per sensor
        self.sensitivity_profiles = {
            'TGS2600': {
                'ultra_sensitive': {'multiplier': 10.0, 'threshold': 0.02, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 5.0, 'threshold': 0.05, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.0, 'threshold': 0.10, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.0, 'threshold': 0.15, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.5, 'threshold': 0.20, 'baseline_factor': 0.75}
            },
            'TGS2602': {
                'ultra_sensitive': {'multiplier': 12.0, 'threshold': 0.015, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 6.0, 'threshold': 0.04, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.5, 'threshold': 0.08, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.2, 'threshold': 0.12, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.6, 'threshold': 0.18, 'baseline_factor': 0.75}
            },
            'TGS2610': {
                'ultra_sensitive': {'multiplier': 8.0, 'threshold': 0.03, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 4.0, 'threshold': 0.06, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.0, 'threshold': 0.12, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.0, 'threshold': 0.18, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.5, 'threshold': 0.25, 'baseline_factor': 0.75}
            }
        }
        
        # Current sensitivity settings
        self.current_sensitivity = {
            'TGS2600': 'normal',
            'TGS2602': 'normal', 
            'TGS2610': 'normal'
        }
        
        # Custom sensitivity factors
        self.custom_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0,
            'TGS2610': 1.0
        }
        
        self.load_sensitivity_data()
    
    def advanced_ppm_calculation(self, sensor_name, current_voltage, baseline_voltage=None, gas_type='auto'):
        """Advanced PPM calculation dengan multiple algorithms"""
        if baseline_voltage is None:
            baseline_voltage = 1.6
        
        # Get current sensitivity profile
        profile_name = self.current_sensitivity.get(sensor_name, 'normal')
        profile = self.sensitivity_profiles[sensor_name][profile_name]
        custom_factor = self.custom_factors.get(sensor_name, 1.0)
        
        voltage_drop = baseline_voltage - current_voltage
        
        # Algorithm 1: Ultra-sensitive voltage-based
        if abs(voltage_drop) < 0.01:  # Very small changes
            ppm_algo1 = voltage_drop * 1000 * profile['multiplier'] * custom_factor
        else:
            ppm_algo1 = voltage_drop * 500 * profile['multiplier'] * custom_factor
        
        # Algorithm 2: Exponential sensitivity curve
        if voltage_drop > 0:
            ppm_algo2 = (math.exp(voltage_drop * 5) - 1) * profile['multiplier'] * custom_factor * 50
        else:
            ppm_algo2 = 0
        
        # Algorithm 3: Gas-specific factors
        gas_factor = self.get_gas_response_factor(sensor_name, gas_type)
        if voltage_drop > profile['threshold']:
            ppm_algo3 = (voltage_drop ** 1.5) * 1000 * gas_factor * custom_factor
        else:
            ppm_algo3 = voltage_drop * 200 * gas_factor * custom_factor
        
        # Combine algorithms
        algorithms = [ppm_algo1, ppm_algo2, ppm_algo3]
        valid_algorithms = [ppm for ppm in algorithms if ppm >= 0]
        
        if not valid_algorithms:
            return 0
        
        final_ppm = max(valid_algorithms)  # Take maximum for sensitivity
        
        # Apply constraints
        max_ppm = 1000 if profile_name in ['ultra_sensitive', 'high_sensitive'] else 500
        return min(max_ppm, max(0, final_ppm))
    
    def get_gas_response_factor(self, sensor_name, gas_type):
        """Get gas-specific response factor"""
        response_factors = {
            'TGS2600': {
                'alcohol': 3.0, 'hydrogen': 2.5, 'carbon_monoxide': 2.0,
                'pertalite': 2.8, 'pertamax': 3.2, 'auto': 2.5, 'default': 2.0
            },
            'TGS2602': {
                'alcohol': 4.0, 'toluene': 3.5, 'ammonia': 3.0, 'h2s': 4.5,
                'pertalite': 3.2, 'pertamax': 3.8, 'auto': 3.0, 'default': 2.5
            },
            'TGS2610': {
                'butane': 2.5, 'propane': 2.8, 'lp_gas': 2.6, 'iso_butane': 2.4,
                'pertalite': 3.0, 'pertamax': 3.5, 'auto': 2.5, 'default': 2.0
            }
        }
        
        return response_factors.get(sensor_name, {}).get(gas_type, 2.0)
    
    def get_sensitivity_status(self):
        """Get current sensitivity status for all sensors"""
        status = {}
        
        for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
            profile = self.current_sensitivity.get(sensor_name, 'normal')
            custom = self.custom_factors.get(sensor_name, 1.0)
            multiplier = self.sensitivity_profiles[sensor_name][profile]['multiplier']
            
            status[sensor_name] = {
                'profile': profile,
                'custom_factor': custom,
                'base_multiplier': multiplier,
                'effective_multiplier': multiplier * custom,
                'sensitivity_level': self.get_sensitivity_level(multiplier * custom)
            }
            
        return status
    
    def get_sensitivity_level(self, effective_multiplier):
        """Convert multiplier to descriptive level"""
        if effective_multiplier >= 10:
            return "ULTRA HIGH"
        elif effective_multiplier >= 5:
            return "HIGH"
        elif effective_multiplier >= 2:
            return "NORMAL"
        elif effective_multiplier >= 1:
            return "MODERATE"
        else:
            return "LOW"
    
    def save_sensitivity_data(self):
        """Save sensitivity configuration"""
        sensitivity_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'advanced_sensitivity_v4.0',
            'current_sensitivity': self.current_sensitivity,
            'custom_factors': self.custom_factors
        }
        
        try:
            with open('sensitivity_config.json', 'w') as f:
                json.dump(sensitivity_data, f, indent=2)
            self.logger.info("Sensitivity configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving sensitivity data: {e}")
    
    def load_sensitivity_data(self):
        """Load sensitivity configuration"""
        try:
            with open('sensitivity_config.json', 'r') as f:
                data = json.load(f)
            
            self.current_sensitivity = data.get('current_sensitivity', {
                'TGS2600': 'normal', 'TGS2602': 'normal', 'TGS2610': 'normal'
            })
            self.custom_factors = data.get('custom_factors', {
                'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0
            })
            
            self.logger.info("Sensitivity configuration loaded")
            
        except FileNotFoundError:
            self.logger.info("No sensitivity config found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading sensitivity data: {e}")

class EmergencyPPMCalculator:
    """Enhanced Emergency PPM Calculator"""
    
    def __init__(self, logger):
        self.logger = logger
        
        self.emergency_baselines = {
            'TGS2600': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 3.5,
                'detection_threshold': 0.005,
                'voltage_noise_threshold': 0.002
            },
            'TGS2602': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 4.0,
                'detection_threshold': 0.003,
                'voltage_noise_threshold': 0.001
            },
            'TGS2610': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 2.5,
                'detection_threshold': 0.008,
                'voltage_noise_threshold': 0.003
            }
        }
    
    def calculate_emergency_ppm(self, sensor_name, current_voltage, gas_type='default', sensitivity_manager=None):
        """Enhanced emergency PPM calculation"""
        if sensor_name not in self.emergency_baselines:
            return 0
        
        baseline = self.emergency_baselines[sensor_name]
        baseline_voltage = baseline['clean_air_voltage']
        voltage_drop = baseline_voltage - current_voltage
        
        if abs(voltage_drop) < baseline['voltage_noise_threshold']:
            return 0
        
        # Apply sensitivity manager multiplier if available
        sensitivity_multiplier = 1.0
        if sensitivity_manager:
            profile = sensitivity_manager.current_sensitivity.get(sensor_name, 'normal')
            custom_factor = sensitivity_manager.custom_factors.get(sensor_name, 1.0)
            base_multiplier = sensitivity_manager.sensitivity_profiles[sensor_name][profile]['multiplier']
            sensitivity_multiplier = base_multiplier * custom_factor
        
        # Enhanced calculation
        response_factor = baseline['gas_response_factor']
        
        if abs(voltage_drop) < 0.05:
            ppm = response_factor * (math.exp(abs(voltage_drop) * 10) - 1)
        else:
            ppm = response_factor * (abs(voltage_drop) * 200) ** 1.3
        
        ppm *= sensitivity_multiplier
        
        max_ppm = 2000 if sensitivity_multiplier > 5 else 1000
        return min(max_ppm, max(0, ppm))

class SmartDriftManager:
    """Smart Drift Manager dengan enhanced troubleshooting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.baseline_history = {}
        self.drift_compensation_factors = {}
        self.last_calibration_time = None
        self.daily_check_done = False
        
        self.drift_tolerance = {
            'excellent': 0.020,
            'good': 0.050,
            'moderate': 0.100,
            'high': 0.200,
            'extreme': 0.300
        }
        
        self.original_baseline = {
            'TGS2600': 1.6,
            'TGS2602': 1.6,
            'TGS2610': 1.6
        }
        
        self.current_baseline = {
            'TGS2600': 1.6,
            'TGS2602': 1.6,
            'TGS2610': 1.6
        }
        
        self.normalization_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0,
            'TGS2610': 1.0
        }
        
        self.voltage_adjustments = {
            'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
        }
        
        self.load_drift_data()
    
    def apply_smart_compensation(self, sensor_name, raw_voltage):
        """Apply smart compensation + normalization"""
        compensated_voltage = raw_voltage
        if sensor_name in self.drift_compensation_factors:
            compensated_voltage = raw_voltage * self.drift_compensation_factors[sensor_name]
        
        normalized_voltage = compensated_voltage * self.normalization_factors.get(sensor_name, 1.0)
        return normalized_voltage, compensated_voltage
    
    def save_drift_data(self):
        """Save drift data"""
        drift_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'smart_drift_v4.0_complete',
            'baseline_history': self.baseline_history,
            'drift_compensation_factors': self.drift_compensation_factors,
            'original_baseline': self.original_baseline,
            'current_baseline': self.current_baseline,
            'normalization_factors': self.normalization_factors,
            'voltage_adjustments': self.voltage_adjustments,
            'daily_check_done': self.daily_check_done
        }
        
        try:
            with open('smart_drift_data.json', 'w') as f:
                json.dump(drift_data, f, indent=2)
            self.logger.info("Smart drift data saved")
        except Exception as e:
            self.logger.error(f"Error saving drift data: {e}")
    
    def load_drift_data(self):
        """Load drift data"""
        try:
            with open('smart_drift_data.json', 'r') as f:
                drift_data = json.load(f)
            
            self.baseline_history = drift_data.get('baseline_history', {})
            self.drift_compensation_factors = drift_data.get('drift_compensation_factors', {})
            self.original_baseline = drift_data.get('original_baseline', 
                {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.current_baseline = drift_data.get('current_baseline', 
                {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.normalization_factors = drift_data.get('normalization_factors', 
                {'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0})
            self.voltage_adjustments = drift_data.get('voltage_adjustments', {
                'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
            })
            self.daily_check_done = drift_data.get('daily_check_done', False)
            
            self.logger.info("Smart drift data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No drift data found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading drift data: {e}")

class MonitoringDataCollector:
    """Enhanced Monitoring Data Collector with CSV saving"""
    
    def __init__(self, logger):
        self.logger = logger
        self.is_collecting = False
        self.collection_thread = None
        self.data_queue = queue.Queue()
        self.csv_writer = None
        self.csv_file = None
        self.current_filename = None
        
    def start_monitoring(self, sensor_array, mode='datasheet', save_to_csv=True):
        """Start monitoring with optional CSV saving"""
        if self.is_collecting:
            print("⚠️ Monitoring already running!")
            return False
            
        self.is_collecting = True
        
        if save_to_csv:
            # Create CSV file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_filename = f"monitoring_data_{timestamp}.csv"
            
            # CSV headers
            headers = [
                'timestamp', 'predicted_gas', 'confidence'
            ]
            
            # Add sensor columns
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                headers.extend([
                    f'{sensor}_raw_voltage',
                    f'{sensor}_voltage', 
                    f'{sensor}_resistance',
                    f'{sensor}_ppm',
                    f'{sensor}_emergency_ppm',
                    f'{sensor}_advanced_ppm',
                    f'{sensor}_mode',
                    f'{sensor}_drift_factor',
                    f'{sensor}_normalization_factor',
                    f'{sensor}_sensitivity_profile',
                    f'{sensor}_sensitivity_multiplier'
                ])
            
            try:
                self.csv_file = open(self.current_filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(headers)
                self.csv_file.flush()
                print(f"📁 Monitoring data will be saved to: {self.current_filename}")
            except Exception as e:
                print(f"❌ Error creating CSV file: {e}")
                save_to_csv = False
        
        # Start monitoring thread
        self.collection_thread = threading.Thread(
            target=self._monitoring_worker,
            args=(sensor_array, mode, save_to_csv),
            daemon=True
        )
        self.collection_thread.start()
        
        print(f"🚀 Monitoring started in {mode} mode")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring and close CSV file"""
        if not self.is_collecting:
            return False
            
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"💾 Monitoring data saved to: {self.current_filename}")
            except Exception as e:
                print(f"⚠️ Error closing CSV file: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None
        
        print("⏹️ Monitoring stopped")
        return True
    
    def _monitoring_worker(self, sensor_array, mode, save_to_csv):
        """Background monitoring worker"""
        sample_count = 0
        
        try:
            while self.is_collecting:
                # Read sensors
                readings = sensor_array.read_sensors()
                predicted_gas, confidence = sensor_array.predict_gas(readings)
                
                # Prepare row data
                row_data = [
                    datetime.now().isoformat(),
                    predicted_gas,
                    confidence
                ]
                
                # Add sensor data
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    sensor_data = readings[sensor]
                    row_data.extend([
                        sensor_data['raw_voltage'],
                        sensor_data['voltage'],
                        sensor_data['resistance'], 
                        sensor_data['ppm'],
                        sensor_data['emergency_ppm'],
                        sensor_data['advanced_ppm'],
                        sensor_data['mode'],
                        sensor_data['drift_factor'],
                        sensor_data['normalization_factor'],
                        sensor_data['sensitivity_profile'],
                        sensor_data['sensitivity_multiplier']
                    ])
                
                # Save to CSV
                if save_to_csv and self.csv_writer:
                    try:
                        self.csv_writer.writerow(row_data)
                        self.csv_file.flush()  # Force write to disk
                        sample_count += 1
                    except Exception as e:
                        self.logger.error(f"Error writing to CSV: {e}")
                
                # Display progress
                if sample_count % 10 == 0 and save_to_csv:
                    print(f"\r💾 Samples saved: {sample_count}", end="")
                
                time.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            self.logger.error(f"Monitoring worker error: {e}")
        finally:
            if save_to_csv and sample_count > 0:
                print(f"\n✅ Total samples saved: {sample_count}")

class EnhancedDatasheetGasSensorArray:
    def __init__(self):
        """Initialize complete enhanced gas sensor array system"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gas_sensor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize I2C and ADC
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)

            self.tgs2600 = AnalogIn(self.ads, ADS.P0)
            self.tgs2602 = AnalogIn(self.ads, ADS.P1)
            self.tgs2610 = AnalogIn(self.ads, ADS.P2)

            self.logger.info("ADC ADS1115 initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ADC: {e}")
            raise

        # Enhanced sensor configurations
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['hydrogen', 'carbon_monoxide', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 500),
                'heater_voltage': 5.0,
                'heater_current': 42e-3,
                'power_consumption': 210e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-20, 50),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'hydrogen': (0.3, 0.6),
                    'carbon_monoxide': (0.4, 0.7),
                    'alcohol': (0.2, 0.5)
                },
                'use_extended_mode': False,
                'concentration_threshold': 50,
                'extended_sensitivity': 2.5,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
            },
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['toluene', 'ammonia', 'h2s', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 300),
                'heater_voltage': 5.0,
                'heater_current': 56e-3,
                'power_consumption': 280e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-10, 60),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'alcohol': (0.08, 0.5),
                    'toluene': (0.1, 0.4),
                    'ammonia': (0.15, 0.6),
                    'h2s': (0.05, 0.3)
                },
                'use_extended_mode': False,
                'concentration_threshold': 40,
                'extended_sensitivity': 3.0,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
            },
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'iso_butane'],
                'detection_range': (1, 25),
                'extended_range': (1, 200),
                'heater_voltage': 5.0,
                'heater_current': 56e-3,
                'power_consumption': 280e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-10, 50),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'iso_butane': (0.45, 0.62),
                    'butane': (0.4, 0.6),
                    'propane': (0.35, 0.55),
                    'lp_gas': (0.4, 0.6)
                },
                'use_extended_mode': False,
                'concentration_threshold': 30,
                'extended_sensitivity': 2.0,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
            }
        }

        # Initialize all managers and new adaptive components
        self.drift_manager = SmartDriftManager(self.logger)
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)
        self.sensitivity_manager = AdvancedSensitivityManager(self.logger)
        self.monitoring_collector = MonitoringDataCollector(self.logger)
        
        # NEW: Adaptive feature processing
        self.adaptive_processor = AdaptiveFeatureProcessor(self.logger)
        self.prediction_engine = EnhancedPredictionEngine(self.logger)
        
        # Readings history for adaptive learning
        self.readings_history = []
        self.max_history_size = 100

        # Environmental compensation
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0
        self.current_humidity = 65.0

        # Enhanced Machine Learning with adaptive features
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.feature_names = None
        self.training_metadata = None  # Store training metadata

        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)

        self.logger.info("Enhanced Gas Sensor Array System v4.3 - ADAPTIVE PPM RANGE + Enhanced Detection")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """Convert ADC voltage to sensor resistance"""
        if voltage <= 0.001:
            return float('inf')

        circuit_voltage = 5.0
        if voltage >= circuit_voltage:
            return 0.1

        sensor_resistance = load_resistance * (circuit_voltage - voltage) / voltage
        return max(1, sensor_resistance)

    def temperature_compensation(self, sensor_name, raw_value, temperature):
        """Apply temperature compensation"""
        if not self.temp_compensation_enabled:
            return raw_value

        temp_factors = {
            'TGS2600': {-20: 1.8, -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0, 30: 0.95, 40: 0.9, 50: 0.85},
            'TGS2602': {-10: 1.5, 0: 1.3, 10: 1.1, 20: 1.0, 30: 0.9, 40: 0.85, 50: 0.8, 60: 0.75},
            'TGS2610': {-10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0, 30: 0.95, 40: 0.9, 50: 0.85}
        }

        temp_curve = temp_factors.get(sensor_name, {20: 1.0})
        temps = sorted(temp_curve.keys())

        if temperature <= temps[0]:
            factor = temp_curve[temps[0]]
        elif temperature >= temps[-1]:
            factor = temp_curve[temps[-1]]
        else:
            for i in range(len(temps) - 1):
                if temps[i] <= temperature <= temps[i + 1]:
                    t1, t2 = temps[i], temps[i + 1]
                    f1, f2 = temp_curve[t1], temp_curve[t2]
                    factor = f1 + (f2 - f1) * (temperature - t1) / (t2 - t1)
                    break
            else:
                factor = 1.0

        return raw_value * factor

    def humidity_compensation(self, sensor_name, raw_value, humidity):
        """Apply humidity compensation"""
        if not self.humidity_compensation_enabled:
            return raw_value

        humidity_factors = {
            'TGS2600': {35: 1.1, 65: 1.0, 95: 0.9},
            'TGS2602': {40: 1.05, 65: 1.0, 85: 0.95, 100: 0.9},
            'TGS2610': {40: 1.1, 65: 1.0, 85: 0.95}
        }

        humidity_curve = humidity_factors.get(sensor_name, {65: 1.0})
        humidities = sorted(humidity_curve.keys())

        if humidity <= humidities[0]:
            factor = humidity_curve[humidities[0]]
        elif humidity >= humidities[-1]:
            factor = humidity_curve[humidities[-1]]
        else:
            for i in range(len(humidities) - 1):
                if humidities[i] <= humidity <= humidities[i + 1]:
                    h1, h2 = humidities[i], humidities[i + 1]
                    f1, f2 = humidity_curve[h1], humidity_curve[h2]
                    factor = f1 + (f2 - f1) * (humidity - h1) / (h2 - h1)
                    break
            else:
                factor = 1.0

        return raw_value * factor

    def resistance_to_ppm(self, sensor_name, resistance, gas_type='auto'):
        """Enhanced resistance to PPM conversion with all algorithms"""
        config = self.sensor_config[sensor_name]
        
        # Priority 1: Advanced sensitivity calculation
        if config.get('use_advanced_sensitivity', True):
            current_voltage = config['channel'].voltage
            baseline_voltage = config.get('baseline_voltage', 1.6)
            ppm_advanced = self.sensitivity_manager.advanced_ppm_calculation(
                sensor_name, current_voltage, baseline_voltage, gas_type
            )
            if ppm_advanced > 0:
                return ppm_advanced
        
        # Priority 2: Emergency calculation
        if config.get('use_emergency_ppm', False):
            current_voltage = config['channel'].voltage
            return self.emergency_ppm_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, gas_type, self.sensitivity_manager
            )
        
        # Priority 3: Standard calculation
        R0 = config.get('R0')
        if R0 is None or R0 == 0:
            current_voltage = config['channel'].voltage
            emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, gas_type, self.sensitivity_manager
            )
            
            if emergency_ppm > 0:
                return emergency_ppm
            
            return self.simplified_ppm_calculation(sensor_name, resistance)

        rs_r0_ratio = resistance / R0

        if config['use_extended_mode']:
            return self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            ppm = self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
            
            if ppm == 0:
                current_voltage = config['channel'].voltage
                baseline_voltage = config.get('baseline_voltage', 1.6)
                advanced_ppm = self.sensitivity_manager.advanced_ppm_calculation(
                    sensor_name, current_voltage, baseline_voltage, gas_type
                )
                return max(ppm, advanced_ppm)
            
            return ppm

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Enhanced datasheet PPM calculation"""
        if sensor_name == 'TGS2600':
            if rs_r0_ratio < 0.15:
                ppm = min(60, 60 * (0.4 / rs_r0_ratio) ** 2.5)
            elif rs_r0_ratio > 0.98:
                ppm = 0
            else:
                ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                ppm = min(ppm, 60)

        elif sensor_name == 'TGS2602':
            if rs_r0_ratio < 0.03:
                ppm = min(50, 45 * (0.15 / rs_r0_ratio) ** 1.8)
            elif rs_r0_ratio > 0.98:
                ppm = 0
            else:
                ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                ppm = min(ppm, 50)

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.35:
                ppm = min(35, 40 * (0.45 / rs_r0_ratio) ** 1.2)
            elif rs_r0_ratio > 0.99:
                ppm = 0
            else:
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                ppm = min(ppm, 35)
        else:
            ppm = 0

        return max(0, ppm)

    def extended_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Extended PPM calculation"""
        config = self.sensor_config[sensor_name]
        sensitivity = config['extended_sensitivity']

        if rs_r0_ratio >= 1.0:
            return 0

        if sensor_name == 'TGS2600':
            if rs_r0_ratio < 0.05:
                base_ppm = 200 + (0.05 - rs_r0_ratio) * 1000
            elif rs_r0_ratio < 0.2:
                base_ppm = 100 + (0.2 - rs_r0_ratio) * 500
            else:
                base_ppm = 60 * ((0.6 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2602':
            if rs_r0_ratio < 0.02:
                base_ppm = 150 + (0.02 - rs_r0_ratio) * 2000
            elif rs_r0_ratio < 0.1:
                base_ppm = 75 + (0.1 - rs_r0_ratio) * 800
            else:
                base_ppm = 50 * ((0.3 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.1:
                base_ppm = 100 + (0.1 - rs_r0_ratio) * 1500
            elif rs_r0_ratio < 0.3:
                base_ppm = 50 + (0.3 - rs_r0_ratio) * 400
            else:
                base_ppm = 40 * ((0.7 / rs_r0_ratio) ** sensitivity)
        else:
            base_ppm = 0

        gas_multipliers = {
            'alcohol': 1.0, 'pertalite': 1.3, 'pertamax': 1.6, 'dexlite': 1.9, 
            'biosolar': 2.2, 'hydrogen': 0.8, 'toluene': 1.1, 'ammonia': 0.9, 
            'butane': 1.2, 'propane': 1.4, 'normal': 0.7
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Simplified PPM calculation"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 1.6)

        baseline_resistance = self.voltage_to_resistance(baseline_voltage)

        if resistance >= baseline_resistance:
            return 0

        ratio = baseline_resistance / resistance

        if config['use_extended_mode']:
            max_range = config['extended_range'][1]
            ppm = max_range * (ratio - 1) * 0.4
        else:
            max_range = config['detection_range'][1]
            ppm = max_range * (ratio - 1) * 0.5

        return max(0, ppm)

    def read_sensors(self):
        """Enhanced sensor reading with adaptive features"""
        readings = {}

        for sensor_name, config in self.sensor_config.items():
            try:
                # Read raw voltage
                raw_voltage = config['channel'].voltage
                
                # Apply smart drift compensation
                normalized_voltage, compensated_voltage = self.drift_manager.apply_smart_compensation(sensor_name, raw_voltage)

                # Convert to resistance
                resistance = self.voltage_to_resistance(compensated_voltage, config['load_resistance'])

                # Apply environmental compensation
                compensated_resistance = self.temperature_compensation(
                    sensor_name, resistance, self.current_temperature)
                compensated_resistance = self.humidity_compensation(
                    sensor_name, compensated_resistance, self.current_humidity)

                # Calculate Rs/R0 ratio
                R0 = config.get('R0')
                rs_r0_ratio = compensated_resistance / R0 if R0 else None

                # Calculate PPM with all methods
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)
                
                emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(
                    sensor_name, raw_voltage, 'auto', self.sensitivity_manager
                )
                
                advanced_ppm = self.sensitivity_manager.advanced_ppm_calculation(
                    sensor_name, raw_voltage, config.get('baseline_voltage', 1.6)
                )

                # Mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"
                if config.get('use_emergency_ppm', False):
                    current_mode += " (Emergency)"
                if config.get('use_advanced_sensitivity', True):
                    current_mode += " (Advanced)"

                # Status info
                drift_factor = self.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                normalization_factor = self.drift_manager.normalization_factors.get(sensor_name, 1.0)
                smart_compensation_applied = abs(1 - drift_factor) > 0.01 or abs(1 - normalization_factor) > 0.01
                voltage_adjusted = self.drift_manager.voltage_adjustments.get(sensor_name, {}).get('adjusted', False)
                
                sensitivity_status = self.sensitivity_manager.get_sensitivity_status().get(sensor_name, {})

                readings[sensor_name] = {
                    'voltage': normalized_voltage,
                    'raw_voltage': raw_voltage,
                    'compensated_voltage': compensated_voltage,
                    'resistance': resistance,
                    'compensated_resistance': compensated_resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'emergency_ppm': emergency_ppm,
                    'advanced_ppm': advanced_ppm,
                    'R0': R0,
                    'mode': current_mode,
                    'target_gases': config['target_gases'],
                    'smart_compensation_applied': smart_compensation_applied,
                    'drift_factor': drift_factor,
                    'normalization_factor': normalization_factor,
                    'voltage_adjusted': voltage_adjusted,
                    'emergency_mode': config.get('use_emergency_ppm', False),
                    'advanced_sensitivity_mode': config.get('use_advanced_sensitivity', True),
                    'sensitivity_profile': sensitivity_status.get('profile', 'normal'),
                    'sensitivity_multiplier': sensitivity_status.get('effective_multiplier', 1.0)
                }

            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'raw_voltage': 0, 'compensated_voltage': 0, 'resistance': 0, 
                    'compensated_resistance': 0, 'rs_r0_ratio': None, 'ppm': 0, 'emergency_ppm': 0, 
                    'advanced_ppm': 0, 'R0': None, 'mode': 'Error', 'target_gases': [], 
                    'smart_compensation_applied': False, 'drift_factor': 1.0, 'normalization_factor': 1.0, 
                    'voltage_adjusted': False, 'emergency_mode': False, 'advanced_sensitivity_mode': False, 
                    'sensitivity_profile': 'unknown', 'sensitivity_multiplier': 1.0
                }

        # Add to readings history for adaptive learning
        self.readings_history.append(readings)
        if len(self.readings_history) > self.max_history_size:
            self.readings_history.pop(0)
        
        # Update adaptive processor periodically
        if len(self.readings_history) % 20 == 0:
            self.adaptive_processor.update_current_stats(self.readings_history)
            self.adaptive_processor.evaluate_transformation_quality(self.readings_history)

        return readings

    def set_sensor_mode(self, mode='datasheet'):
        """Set calculation mode for all sensors"""
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)

            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']
                    self.sensor_config[sensor_name]['emergency_mode'] = False
                    self.sensor_config[sensor_name]['use_emergency_ppm'] = False

            if 'timestamp' in calib_data:
                self.drift_manager.last_calibration_time = datetime.fromisoformat(calib_data['timestamp'])

            self.logger.info("Enhanced calibration data loaded successfully")
            self.logger.info(f"Calibration date: {calib_data.get('timestamp', 'Unknown')}")

            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Advanced features available.")
            return False
        except Exception as e:
            self.logger.error(f"Error loading calibration file: {e}")
            return False

    def find_training_files(self):
        """Enhanced training file detection"""
        # Possible file patterns and locations
        possible_patterns = [
            "training_*.csv",
            "data/training_*.csv", 
            "*training*.csv",
            "data/*training*.csv"
        ]
        
        training_files = []
        
        for pattern in possible_patterns:
            files = glob.glob(pattern)
            training_files.extend(files)
        
        # Remove duplicates
        training_files = list(set(training_files))
        
        # Filter and validate files
        valid_files = []
        for file in training_files:
            if os.path.exists(file) and file.endswith('.csv'):
                try:
                    # Quick validation - check if file has expected columns
                    df = pd.read_csv(file, nrows=1)
                    required_columns = ['gas_type', 'TGS2600_voltage', 'TGS2602_voltage', 'TGS2610_voltage']
                    if all(col in df.columns for col in required_columns):
                        valid_files.append(file)
                        self.logger.info(f"Found valid training file: {file}")
                except Exception as e:
                    self.logger.warning(f"Skipping invalid file {file}: {e}")
        
        return valid_files

    def train_model(self):
        """Enhanced model training with adaptive feature processing"""
        print("\n🤖 ADAPTIVE MODEL TRAINING WITH PPM RANGE COMPENSATION")
        print("="*60)
        
        # Find training files
        training_files = self.find_training_files()
        
        if not training_files:
            print("❌ No training data found!")
            print("📁 Looking for files matching patterns:")
            print("   - training_*.csv")
            print("   - data/training_*.csv") 
            print("   - *training*.csv")
            print("\n💡 Make sure training files are in current directory or 'data/' folder")
            print("Collect training data first using option 2")
            return False
        
        print(f"📂 Found {len(training_files)} training files:")
        for file in training_files:
            print(f"   ✅ {file}")
        
        # Load and combine data
        print(f"\n📊 Loading training data...")
        all_data = []
        total_samples = 0
        gas_type_counts = {}
        
        # Collect training statistics
        training_stats = {}
        
        for file in training_files:
            try:
                df = pd.read_csv(file)
                samples = len(df)
                total_samples += samples
                
                # Count gas types
                if 'gas_type' in df.columns:
                    gas_types = df['gas_type'].value_counts().to_dict()
                    for gas, count in gas_types.items():
                        gas_type_counts[gas] = gas_type_counts.get(gas, 0) + count
                
                # Collect statistics for adaptive processing
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    for feature_type in ['voltage', 'ppm']:
                        col_name = f'{sensor}_{feature_type}'
                        if col_name in df.columns:
                            values = df[col_name].dropna()
                            if len(values) > 0:
                                if col_name not in training_stats:
                                    training_stats[col_name] = {
                                        'min': float(values.min()),
                                        'max': float(values.max()),
                                        'mean': float(values.mean()),
                                        'std': float(values.std())
                                    }
                                else:
                                    # Update stats with new data
                                    existing = training_stats[col_name]
                                    training_stats[col_name]['min'] = min(existing['min'], float(values.min()))
                                    training_stats[col_name]['max'] = max(existing['max'], float(values.max()))
                
                all_data.append(df)
                print(f"   📄 {os.path.basename(file)}: {samples} samples")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if not all_data:
            print("❌ No valid training data could be loaded!")
            return False
        
        # Update adaptive processor with training statistics
        self.adaptive_processor.training_stats = training_stats
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📈 Training Data Summary:")
        print(f"   Total samples: {total_samples}")
        print(f"   Gas types: {len(gas_type_counts)}")
        for gas, count in gas_type_counts.items():
            percentage = (count / total_samples) * 100
            print(f"     {gas}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n📊 PPM RANGE ANALYSIS:")
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            if ppm_col in combined_df.columns:
                ppm_values = combined_df[ppm_col].dropna()
                if len(ppm_values) > 0:
                    print(f"   {sensor}: {ppm_values.min():.1f} - {ppm_values.max():.1f} PPM")
        
        # Prepare features and target with adaptive processing
        print(f"\n🔧 Preparing adaptive features...")
        
        # Define base feature columns
        base_feature_columns = []
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            base_feature_columns.extend([
                f'{sensor}_voltage',
                f'{sensor}_ppm'
            ])
        
        # Add environmental features if available
        if 'temperature' in combined_df.columns:
            base_feature_columns.append('temperature')
        if 'humidity' in combined_df.columns:
            base_feature_columns.append('humidity')
        
        # Extract and transform features
        print(f"   🎯 Applying adaptive feature transformations...")
        
        # Prepare feature matrix
        X_raw = combined_df[base_feature_columns].fillna(0)
        y = combined_df['gas_type']
        
        # Apply adaptive transformations to training data
        X_transformed = []
        for idx, row in X_raw.iterrows():
            feature_vector = row.values.tolist()
            transformed_features = self.adaptive_processor.transform_features(
                feature_vector, base_feature_columns
            )
            X_transformed.append(transformed_features)
        
        X_transformed = np.array(X_transformed)
        
        # Store feature names for consistency
        self.feature_names = base_feature_columns
        
        print(f"   Original features: {len(base_feature_columns)}")
        print(f"   Feature columns: {base_feature_columns}")
        print(f"   Adaptive transformation method: {self.adaptive_processor.best_transform_method}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Scale features
        print(f"\n⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\n🎯 Training enhanced RandomForest model...")
        self.model = RandomForestClassifier(
            n_estimators=150,  # Increased for better performance
            random_state=42,
            max_depth=12,      # Slightly deeper for complex patterns
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print(f"\n📊 Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        for gas_type, metrics in report.items():
            if gas_type not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = metrics['precision']
                recall = metrics['recall'] 
                f1 = metrics['f1-score']
                support = int(metrics['support'])
                print(f"   {gas_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({support} samples)")
        
        # Save model and enhanced metadata
        print(f"\n💾 Saving enhanced model...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save enhanced metadata with training statistics
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': 'adaptive_v4.3_ppm_range_fixed',
            'total_samples': total_samples,
            'feature_columns': base_feature_columns,
            'feature_names': self.feature_names,
            'gas_types': list(gas_type_counts.keys()),
            'gas_type_counts': gas_type_counts,
            'accuracy': accuracy,
            'model_params': self.model.get_params(),
            'training_files': training_files,
            'training_stats': training_stats,  # IMPORTANT: PPM range info
            'adaptive_processor_config': {
                'best_transform_method': self.adaptive_processor.best_transform_method,
                'ppm_scaling_factors': self.adaptive_processor.ppm_scaling_factors
            }
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update adaptive processor training stats
        self.adaptive_processor.training_stats = training_stats
        self.adaptive_processor.save_adaptive_config()
        
        self.is_model_trained = True
        self.training_metadata = metadata
        
        print(f"✅ Enhanced adaptive model training completed!")
        print(f"📁 Files saved:")
        print(f"   🤖 models/gas_classifier.pkl")
        print(f"   ⚖️  models/scaler.pkl") 
        print(f"   📋 models/model_metadata.json (with PPM range info)")
        print(f"   🎯 adaptive_feature_config.json")
        
        print(f"\n🎯 ADAPTIVE FEATURES ACTIVE:")
        print(f"   PPM Range Compensation: ✅")
        print(f"   Multi-scale Features: ✅")
        print(f"   Confidence Boosting: ✅")
        print(f"   Temporal Smoothing: ✅")
        
        return True

    def load_model(self):
        """Load trained model with adaptive features"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            
            # Load enhanced metadata with training statistics
            try:
                with open('models/model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', metadata.get('feature_columns', []))
                self.training_metadata = metadata
                
                # Update adaptive processor with training stats
                if 'training_stats' in metadata:
                    self.adaptive_processor.training_stats = metadata['training_stats']
                
                # Load adaptive processor config
                if 'adaptive_processor_config' in metadata:
                    config = metadata['adaptive_processor_config']
                    self.adaptive_processor.best_transform_method = config.get(
                        'best_transform_method', 'adaptive_normalize'
                    )
                    self.adaptive_processor.ppm_scaling_factors = config.get(
                        'ppm_scaling_factors', self.adaptive_processor.ppm_scaling_factors
                    )
                
            except Exception as e:
                self.logger.warning(f"Could not load enhanced metadata: {e}")
                # Fallback feature names
                self.feature_names = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                                     'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']
            
            # Load adaptive processor config
            self.adaptive_processor.load_adaptive_config()
            
            self.is_model_trained = True
            self.logger.info("Enhanced adaptive model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction with adaptive features and confidence boosting"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0

        try:
            # Extract multi-scale features
            multi_scale_features = self.adaptive_processor.extract_multi_scale_features(readings)
            
            # Use base feature names for consistency
            feature_columns = self.feature_names or ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                                                    'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']

            feature_vector = []
            for feature in feature_columns:
                if feature == 'temperature':
                    feature_vector.append(self.current_temperature)
                elif feature == 'humidity':
                    feature_vector.append(self.current_humidity)
                else:
                    # Use multi-scale features if available, otherwise fall back to basic features
                    if feature in multi_scale_features:
                        value = multi_scale_features[feature]
                    else:
                        parts = feature.split('_')
                        sensor = parts[0]
                        measurement = '_'.join(parts[1:])
                        
                        if sensor in readings and measurement in readings[sensor]:
                            value = readings[sensor][measurement]
                        else:
                            value = 0.0
                    
                    feature_vector.append(value if value is not None else 0.0)

            # Apply adaptive feature transformations
            transformed_features = self.adaptive_processor.transform_features(feature_vector, feature_columns)

            # Create DataFrame with feature names to avoid warnings
            features_df = pd.DataFrame([transformed_features], columns=feature_columns)
            features_array = np.nan_to_num(features_df.values, nan=0.0)

            # Scale and predict
            features_scaled = self.scaler.transform(features_array)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            base_confidence = probabilities.max()

            # Apply confidence boosting
            boosted_confidence = self.prediction_engine.boost_confidence_with_patterns(
                prediction, base_confidence, readings
            )

            # Apply temporal smoothing
            final_prediction, final_confidence = self.prediction_engine.apply_temporal_smoothing(
                prediction, boosted_confidence
            )

            # Apply multi-evidence fusion
            fused_prediction, fused_confidence = self.prediction_engine.fuse_multi_evidence(
                final_prediction, final_confidence, readings
            )

            # Apply adaptive confidence threshold
            adaptive_threshold = self.adaptive_processor.get_adaptive_confidence_threshold(
                fused_prediction, fused_confidence
            )

            # If confidence is below adaptive threshold, classify as uncertain
            if fused_confidence < adaptive_threshold:
                return f"{fused_prediction} (uncertain)", fused_confidence

            return fused_prediction, fused_confidence

        except Exception as e:
            self.logger.error(f"Error in enhanced prediction: {e}")
            return "Error", 0.0

    def set_environmental_conditions(self, temperature=None, humidity=None):
        """Set environmental conditions"""
        if temperature is not None:
            self.current_temperature = temperature
            self.logger.info(f"Temperature set to {temperature}°C")

        if humidity is not None:
            self.current_humidity = humidity
            self.logger.info(f"Humidity set to {humidity}%RH")

def main():
    """Complete main function with adaptive features"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️  CALIBRATION NOT FOUND")
        print("Advanced Sensitivity System available as fallback")

    # Load existing model
    model_loaded = gas_sensor.load_model()
    
    if model_loaded:
        print("\n🎯 ADAPTIVE MODEL LOADED")
        print("✅ PPM Range Compensation: ACTIVE")
        print("✅ Enhanced Detection: ACTIVE")

    while True:
        print("\n" + "="*70)
        print("🧠 SMART Gas Sensor Array System - ADAPTIVE v4.3")
        print("🎯 FIXED: PPM Range Mismatch + Enhanced Gas Detection")
        print("="*70)
        print("1. Calibrate sensors")
        print("2. Collect training data")
        print("3. Train ADAPTIVE machine learning model")
        print("4. Start monitoring - Datasheet mode (ENHANCED)")
        print("5. Start monitoring - Extended mode (ENHANCED)")
        print("6. Test single reading (ADAPTIVE analysis)")
        print("7. Set environmental conditions")
        print("8. Switch sensor calculation mode")
        print("9. View sensor diagnostics")
        print("10. Exit")
        print("-" * 40)
        print("🎯 ADAPTIVE FEATURES:")
        print("31. View PPM range compensation status")
        print("32. Adjust adaptive transformation method")
        print("33. Enable/disable multi-scale features")
        print("34. View prediction confidence analysis")
        print("35. Test gas detection with current settings")
        print("-"*70)

        try:
            choice = input("Select option (1-35): ").strip()

            if choice == '3':
                gas_sensor.train_model()

            elif choice == '4':
                gas_sensor.set_sensor_mode('datasheet')
                print("\n🔬 Starting ADAPTIVE monitoring mode...")
                print("🎯 Enhanced detection with PPM range compensation")
                print("Press Ctrl+C to stop\n")
                
                # Start monitoring with CSV saving
                gas_sensor.monitoring_collector.start_monitoring(gas_sensor, 'datasheet', save_to_csv=True)
                
                try:
                    while gas_sensor.monitoring_collector.is_collecting:
                        readings = gas_sensor.read_sensors()
                        predicted_gas, confidence = gas_sensor.predict_gas(readings)
                        
                        print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | "
                              f"🎯 {predicted_gas} ({confidence:.3f}) | "
                              f"TGS2600: {readings['TGS2600']['ppm']:.1f} | "
                              f"TGS2602: {readings['TGS2602']['ppm']:.1f} | "
                              f"TGS2610: {readings['TGS2610']['ppm']:.1f}", end="")
                        
                        time.sleep(2)
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️  Monitoring stopped by user")
                finally:
                    gas_sensor.monitoring_collector.stop_monitoring()

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*70)
                print("🎯 ADAPTIVE SENSOR ANALYSIS - ENHANCED DETECTION")
                print("="*70)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    
                    print(f"\n  📊 PPM ANALYSIS (All Methods):")
                    print(f"    Main PPM: {data['ppm']:.1f}")
                    print(f"    Emergency PPM: {data['emergency_ppm']:.1f}")
                    print(f"    Advanced PPM: {data['advanced_ppm']:.1f}")
                    
                    print(f"\n  🎯 SENSITIVITY STATUS:")
                    print(f"    Profile: {data['sensitivity_profile']}")
                    print(f"    Multiplier: {data['sensitivity_multiplier']:.1f}x")

                print(f"\n🎯 ADAPTIVE PREDICTION:")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")
                
                # Show adaptive status
                print(f"\n🔧 ADAPTIVE STATUS:")
                print(f"  PPM Scaling: {gas_sensor.adaptive_processor.ppm_scaling_factors}")
                print(f"  Transform Method: {gas_sensor.adaptive_processor.best_transform_method}")
                print(f"  Multi-scale Features: {gas_sensor.adaptive_processor.enable_multi_scale}")

            elif choice == '31':
                print("\n🎯 PPM RANGE COMPENSATION STATUS")
                print("="*50)
                
                print("Training PPM Ranges:")
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    ppm_key = f'{sensor}_ppm'
                    if ppm_key in gas_sensor.adaptive_processor.training_stats:
                        stats = gas_sensor.adaptive_processor.training_stats[ppm_key]
                        print(f"  {sensor}: {stats['min']:.1f} - {stats['max']:.1f} PPM")
                
                print("\nCurrent PPM Ranges:")
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    ppm_key = f'{sensor}_ppm'
                    if ppm_key in gas_sensor.adaptive_processor.current_stats:
                        stats = gas_sensor.adaptive_processor.current_stats[ppm_key]
                        print(f"  {sensor}: {stats['min']:.1f} - {stats['max']:.1f} PPM")
                
                print("\nScaling Factors:")
                for sensor, factor in gas_sensor.adaptive_processor.ppm_scaling_factors.items():
                    print(f"  {sensor}: {factor:.3f}x")

            elif choice == '35':
                print("\n🧪 ENHANCED GAS DETECTION TEST")
                print("="*50)
                
                print("Testing with current sensor readings...")
                readings = gas_sensor.read_sensors()
                
                # Test multiple prediction methods
                print("\n📊 DETECTION METHODS COMPARISON:")
                
                # Basic prediction
                predicted_gas, confidence = gas_sensor.predict_gas(readings)
                print(f"1. Enhanced Adaptive: {predicted_gas} (confidence: {confidence:.3f})")
                
                # Pattern analysis
                pattern_scores = gas_sensor.prediction_engine.analyze_sensor_patterns(readings)
                best_pattern = max(pattern_scores.keys(), key=lambda x: pattern_scores[x]) if pattern_scores else "None"
                best_pattern_score = pattern_scores.get(best_pattern, 0.0)
                print(f"2. Pattern Analysis: {best_pattern} (score: {best_pattern_score:.3f})")
                
                # Show detailed sensor responses
                print(f"\n🔬 SENSOR RESPONSES:")
                base_voltage = 1.6
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    if sensor in readings:
                        data = readings[sensor]
                        voltage_drop = base_voltage - data['raw_voltage']
                        print(f"  {sensor}: {data['raw_voltage']:.3f}V (drop: {voltage_drop:.3f}V) → {data['ppm']:.1f} PPM")

            elif choice == '10':
                print("👋 Exiting...")
                # Stop monitoring if running
                if gas_sensor.monitoring_collector.is_collecting:
                    gas_sensor.monitoring_collector.stop_monitoring()
                
                gas_sensor.drift_manager.save_drift_data()
                gas_sensor.sensitivity_manager.save_sensitivity_data()
                gas_sensor.adaptive_processor.save_adaptive_config()
                break

            else:
                if choice not in ['1', '2', '5', '7', '8', '9', '31', '32', '33', '34', '35']:
                    print("❌ Invalid option!")

        except KeyboardInterrupt:
            print("\nOperation cancelled")
            # Stop monitoring if running
            if gas_sensor.monitoring_collector.is_collecting:
                gas_sensor.monitoring_collector.stop_monitoring()
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")
            gas_sensor.logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
