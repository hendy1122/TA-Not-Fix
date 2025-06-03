#!/usr/bin/env python3
"""
Enhanced Gas Detection System for USV Air Pollution Monitoring
Raspberry Pi 4 + ADS1115 ADC + TGS Gas Sensors Array
with Adaptive Rule-Based Classification
Author: [Your Name]
Date: 2025
"""

import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Uncomment these lines when running on Raspberry Pi
# import Adafruit_ADS1x15

class AdaptiveGasClassifier:
    """Enhanced classifier that adapts to training data"""
    
    def __init__(self):
        self.thresholds = None
        self.decision_tree = None
        self.feature_names = ['TGS_2600', 'TGS_2602', 'TGS_2610']
        self.is_trained = False
    
    def analyze_training_data(self, training_data):
        """Analyze training data to understand patterns"""
        
        print("\n=== ANALISIS DATA TRAINING ===")
        
        # Prepare data
        df = training_data.copy()
        
        # Rename columns to match our system
        if 'TGS 2600' in df.columns:
            df = df.rename(columns={
                'TGS 2600': 'TGS_2600',
                'TGS 2602': 'TGS_2602', 
                'TGS 2610': 'TGS_2610'
            })
        
        print(f"Total samples: {len(df)}")
        print(f"Gas types: {df['label'].unique()}")
        print(f"Samples per class:")
        print(df['label'].value_counts())
        
        # Statistics per class
        print(f"\nStatistik per kelas:")
        stats = df.groupby('label')[self.feature_names].agg(['mean', 'std', 'min', 'max'])
        print(stats.round(1))
        
        return df, stats
    
    def calculate_adaptive_thresholds(self, training_data):
        """Calculate thresholds automatically from training data"""
        
        df, _ = self.analyze_training_data(training_data)
        thresholds = {}
        
        print(f"\n=== THRESHOLD ADAPTIF ===")
        
        for gas_type in df['label'].unique():
            class_data = df[df['label'] == gas_type]
            thresholds[gas_type] = {}
            
            print(f"\nGas: {gas_type}")
            
            for sensor in self.feature_names:
                sensor_data = class_data[sensor]
                
                # Calculate various threshold metrics
                thresholds[gas_type][sensor] = {
                    'mean': sensor_data.mean(),
                    'std': sensor_data.std(),
                    'min_threshold': sensor_data.quantile(0.15),  # 15th percentile
                    'max_threshold': sensor_data.quantile(0.85),  # 85th percentile
                    'strict_min': sensor_data.quantile(0.05),    # 5th percentile
                    'strict_max': sensor_data.quantile(0.95),    # 95th percentile
                    'absolute_min': sensor_data.min(),
                    'absolute_max': sensor_data.max()
                }
                
                print(f"  {sensor}: {sensor_data.mean():.0f} ± {sensor_data.std():.0f} "
                      f"[{sensor_data.quantile(0.15):.0f}-{sensor_data.quantile(0.85):.0f}]")
        
        return thresholds
    
    def fit(self, training_data):
        """Train the adaptive classifier"""
        
        print("=== TRAINING ADAPTIVE CLASSIFIER ===")
        
        # Calculate adaptive thresholds
        self.thresholds = self.calculate_adaptive_thresholds(training_data)
        
        # Train decision tree as backup
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data for sklearn
            df, _ = self.analyze_training_data(training_data)
            X = df[self.feature_names].values
            y = df['label'].values
            
            # Train decision tree
            self.decision_tree = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.decision_tree.fit(X, y)
            
            # Store label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            
            print("✓ Decision tree backup trained successfully")
            
        except ImportError:
            print("! sklearn not available - using rule-based only")
            self.decision_tree = None
        
        self.is_trained = True
        print("✓ Adaptive classifier training completed!")
    
    def adaptive_rule_classification(self, features):
        """Enhanced rule-based classification using adaptive thresholds"""
        
        if not self.is_trained:
            raise ValueError("Classifier not trained! Call fit() first.")
        
        tgs2600, tgs2602, tgs2610 = features
        class_scores = {}
        
        # Calculate match score for each gas type
        for gas_type, gas_thresholds in self.thresholds.items():
            total_score = 0
            sensor_scores = []
            
            for sensor_val, sensor_name in zip(features, self.feature_names):
                thresh = gas_thresholds[sensor_name]
                
                # Multi-tier scoring system
                if thresh['min_threshold'] <= sensor_val <= thresh['max_threshold']:
                    # Perfect match - within main range
                    score = 1.0
                elif thresh['strict_min'] <= sensor_val <= thresh['strict_max']:
                    # Good match - within extended range
                    score = 0.8
                elif thresh['absolute_min'] <= sensor_val <= thresh['absolute_max']:
                    # Acceptable match - within absolute bounds
                    score = 0.5
                else:
                    # Out of bounds - calculate penalty
                    if sensor_val < thresh['absolute_min']:
                        # Below minimum
                        distance = (thresh['absolute_min'] - sensor_val) / thresh['absolute_min']
                        score = max(0, 0.3 - distance)
                    else:
                        # Above maximum  
                        distance = (sensor_val - thresh['absolute_max']) / thresh['absolute_max']
                        score = max(0, 0.3 - distance)
                
                sensor_scores.append(score)
                total_score += score
            
            # Average score with bonus for consistent high scores
            avg_score = total_score / len(self.feature_names)
            
            # Bonus for having multiple high-scoring sensors
            high_score_count = sum(1 for s in sensor_scores if s >= 0.8)
            if high_score_count >= 2:
                avg_score += 0.1  # 10% bonus
            
            class_scores[gas_type] = avg_score
        
        # Find best match
        if not class_scores:
            return "unknown", 0.0
        
        predicted_gas = max(class_scores, key=class_scores.get)
        confidence = class_scores[predicted_gas]
        
        # Apply minimum confidence threshold
        if confidence < 0.4:
            return "unknown", confidence
        
        return predicted_gas, confidence
    
    def tree_classification(self, features):
        """Decision tree classification (if available)"""
        
        if self.decision_tree is None:
            return "unknown", 0.0
        
        try:
            prediction = self.decision_tree.predict([features])[0]
            probabilities = self.decision_tree.predict_proba([features])[0]
            confidence = max(probabilities)
            return prediction, confidence
        except:
            return "unknown", 0.0
    
    def predict(self, features, method='hybrid'):
        """
        Main prediction method
        method: 'adaptive', 'tree', 'hybrid', or 'original'
        """
        
        if method == 'adaptive':
            return self.adaptive_rule_classification(features)
        
        elif method == 'tree':
            return self.tree_classification(features)
        
        elif method == 'hybrid':
            # Combine adaptive rules with tree
            adaptive_pred, adaptive_conf = self.adaptive_rule_classification(features)
            tree_pred, tree_conf = self.tree_classification(features)
            
            # Priority logic
            if adaptive_conf >= 0.8:
                return adaptive_pred, adaptive_conf
            elif tree_conf >= 0.8:
                return tree_pred, tree_conf
            elif adaptive_pred == tree_pred and adaptive_conf > 0.5:
                # Both agree and reasonable confidence
                return adaptive_pred, (adaptive_conf + tree_conf) / 2
            elif adaptive_conf > tree_conf:
                return adaptive_pred, adaptive_conf * 0.9  # Small penalty for disagreement
            else:
                return tree_pred, tree_conf * 0.9
        
        elif method == 'original':
            # Use original rule-based method for comparison
            return self.original_rule_classification(features)
        
        else:
            raise ValueError("Method must be 'adaptive', 'tree', 'hybrid', or 'original'")
    
    def original_rule_classification(self, features):
        """Original rule-based classification from main system"""
        tgs2600, tgs2602, tgs2610 = features
        
        if tgs2600 < 200 and tgs2602 < 60 and tgs2610 < 300:
            return "normal", 0.85
        elif tgs2600 > 3000 and tgs2602 > 120 and tgs2610 > 2000:
            return "alkohol", 0.80
        elif tgs2600 > 1000 and tgs2600 < 2000 and tgs2602 > 400 and tgs2610 > 1500:
            return "pertalite", 0.75
        elif tgs2600 > 500 and tgs2600 < 1800 and tgs2602 > 140 and tgs2610 > 1800:
            return "pertamax", 0.75
        else:
            return "unknown", 0.60
    
    def print_current_rules(self):
        """Display current adaptive rules"""
        
        if not self.is_trained:
            print("Classifier not trained yet!")
            return
        
        print("\n=== CURRENT ADAPTIVE RULES ===")
        
        for gas_type, gas_thresholds in self.thresholds.items():
            print(f"\n{gas_type.upper()}:")
            
            conditions = []
            for sensor in self.feature_names:
                thresh = gas_thresholds[sensor]
                min_val = thresh['min_threshold']
                max_val = thresh['max_threshold']
                conditions.append(f"{min_val:.0f} ≤ {sensor} ≤ {max_val:.0f}")
            
            print(f"  Main range: {' AND '.join(conditions)}")
            
            # Extended range
            ext_conditions = []
            for sensor in self.feature_names:
                thresh = gas_thresholds[sensor]
                min_val = thresh['strict_min']
                max_val = thresh['strict_max']
                ext_conditions.append(f"{min_val:.0f} ≤ {sensor} ≤ {max_val:.0f}")
            
            print(f"  Extended: {' AND '.join(ext_conditions)}")


class EnhancedGasDetectionSystem:
    """Enhanced Gas Detection System with Adaptive Classification"""
    
    def __init__(self):
        """Initialize the enhanced gas detection system"""
        
        # Initialize ADC (uncomment when running on Raspberry Pi)
        # self.adc = Adafruit_ADS1x15.ADS1115()
        self.adc = None  # For testing without hardware
        
        # ADC Configuration
        self.GAIN = 1
        self.SAMPLES_PER_SECOND = 128
        
        # Sensor channels
        self.TGS_2600_CHANNEL = 0
        self.TGS_2602_CHANNEL = 1
        self.TGS_2610_CHANNEL = 2
        
        # Voltage reference
        self.VREF = 3.3
        
        # Initialize adaptive classifier
        self.adaptive_classifier = AdaptiveGasClassifier()
        
        # Legacy models (for comparison)
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        # Data storage
        self.csv_filename = f"gas_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print("Enhanced Gas Detection System Initialized")
        print(f"Data will be saved to: {self.csv_filename}")
    
    def load_model_and_data(self):
        """Load and train adaptive classifier from training data"""
        try:
            # Load training data
            training_data = pd.read_csv('merged.csv')
            print(f"Training data loaded: {len(training_data)} samples")
            print(f"Gas types: {training_data['label'].unique()}")
            
            # Train adaptive classifier
            self.adaptive_classifier.fit(training_data)
            
            # Also create legacy model for comparison
            self.create_legacy_model(training_data)
            
            print("✓ All models loaded and trained successfully!")
            
        except FileNotFoundError:
            print("Warning: merged.csv not found!")
            print("Creating default classifier...")
            self.create_default_classifier()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.create_default_classifier()
    
    def create_legacy_model(self, training_data):
        """Create legacy RandomForest model for comparison"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            # Prepare data
            feature_cols = ['TGS 2600', 'TGS 2602', 'TGS 2610']
            if not all(col in training_data.columns for col in feature_cols):
                print("Warning: Expected columns not found in training data")
                return
                
            X = training_data[feature_cols].values
            y = training_data['label'].values
            
            # Encode and scale
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y_encoded)
            
            print("✓ Legacy RandomForest model trained")
            
        except ImportError:
            print("! sklearn not available for legacy model")
    
    def create_default_classifier(self):
        """Create default classifier when no training data available"""
        from sklearn.preprocessing import LabelEncoder
        
        # Set up default label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['normal', 'alkohol', 'pertalite', 'pertamax'])
        
        print("✓ Default classifier created")
    
    def read_sensor_voltage(self, channel):
        """Read voltage from ADC channel"""
        if self.adc is None:
            # Simulate sensor readings for testing
            import random
            return random.uniform(0.1, 3.0)
        
        try:
            adc_value = self.adc.read_adc(channel, gain=self.GAIN, data_rate=self.SAMPLES_PER_SECOND)
            voltage = (adc_value * 4.096) / 32767.0
            return max(0, voltage)
        except Exception as e:
            print(f"Error reading sensor {channel}: {e}")
            return 0.0
    
    def voltage_to_ppm(self, voltage, sensor_type):
        """Convert voltage to PPM - enhanced version"""
        
        # Circuit parameters
        vc = 3.3  # Supply voltage
        rl = 10000  # Load resistance (10kΩ)
        
        # Calculate sensor resistance
        if voltage <= 0 or voltage >= vc:
            return 0
        
        rs = ((vc / voltage) - 1) * rl
        
        if sensor_type == "TGS2600":
            ro = 47000
            rs_ro_ratio = rs / ro
            
            if rs_ro_ratio >= 1.0:
                ppm = 0
            elif rs_ro_ratio >= 0.5:
                ppm = 1 + (1.0 - rs_ro_ratio) * 20
            elif rs_ro_ratio >= 0.3:
                ppm = 10 + (0.5 - rs_ro_ratio) * 100
            else:
                ppm = 30 + (0.3 - rs_ro_ratio) * 200
                
        elif sensor_type == "TGS2602":
            ro = 55000
            rs_ro_ratio = rs / ro
            
            if rs_ro_ratio >= 1.0:
                ppm = 0
            elif rs_ro_ratio >= 0.3:
                ppm = 1 + (1.0 - rs_ro_ratio) * 14
            elif rs_ro_ratio >= 0.08:
                ppm = 10 + (0.3 - rs_ro_ratio) * 91
            else:
                ppm = 30 + (0.08 - rs_ro_ratio) * 250
                
        elif sensor_type == "TGS2610":
            ro = 1800
            rs_ro_ratio = rs / ro
            
            if rs_ro_ratio >= 10:
                ppm = 100
            elif rs_ro_ratio >= 1.0:
                ppm = 100 + (10 - rs_ro_ratio) * 100
            elif rs_ro_ratio >= 0.1:
                ppm = 1000 + (1.0 - rs_ro_ratio) * 10000
            else:
                ppm = 10000 + (0.1 - rs_ro_ratio) * 50000
            
            # Convert to % LEL for gas sensors
            ppm_to_lel = ppm / 21000 * 100
            return max(0, min(25, ppm_to_lel))
        
        else:
            # Default conversion
            ro = 20000
            rs_ro_ratio = rs / ro
            ppm = max(0, (1.0 - rs_ro_ratio) * 100)
        
        return max(0, ppm)
    
    def read_all_sensors(self):
        """Read all sensors and return data"""
        try:
            # Read voltages
            v1 = self.read_sensor_voltage(self.TGS_2600_CHANNEL)
            v2 = self.read_sensor_voltage(self.TGS_2602_CHANNEL)
            v3 = self.read_sensor_voltage(self.TGS_2610_CHANNEL)
            
            # Convert to PPM
            ppm1 = self.voltage_to_ppm(v1, "TGS2600")
            ppm2 = self.voltage_to_ppm(v2, "TGS2602")
            ppm3 = self.voltage_to_ppm(v3, "TGS2610")
            
            return {
                'TGS_2600': {'voltage': v1, 'ppm': ppm1},
                'TGS_2602': {'voltage': v2, 'ppm': ppm2},
                'TGS_2610': {'voltage': v3, 'ppm': ppm3}
            }
            
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return None
    
    def predict_gas_type(self, sensor_data, method='hybrid'):
        """Enhanced prediction with multiple methods"""
        try:
            if not sensor_data:
                return "unknown", 0.0, "no_data"
            
            # Prepare features
            features = [
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['ppm'],
                sensor_data['TGS_2610']['ppm']
            ]
            
            # Use adaptive classifier
            if self.adaptive_classifier.is_trained:
                prediction, confidence = self.adaptive_classifier.predict(features, method=method)
                return prediction, confidence, f"adaptive_{method}"
            
            # Fallback to legacy model
            elif self.model is not None and self.scaler is not None:
                features_array = np.array([features])
                features_scaled = self.scaler.transform(features_array)
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
                gas_type = self.label_encoder.inverse_transform([prediction])[0]
                return gas_type, confidence, "legacy_ml"
            
            # Final fallback to original rules
            else:
                gas_type, confidence = self.original_rule_based_classification(features)
                return gas_type, confidence, "original_rules"
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "unknown", 0.0, "error"
    
    def original_rule_based_classification(self, features):
        """Original rule-based method"""
        tgs2600, tgs2602, tgs2610 = features
        
        if tgs2600 < 200 and tgs2602 < 60 and tgs2610 < 300:
            return "normal", 0.85
        elif tgs2600 > 3000 and tgs2602 > 120 and tgs2610 > 2000:
            return "alkohol", 0.80
        elif tgs2600 > 1000 and tgs2600 < 2000 and tgs2602 > 400 and tgs2610 > 1500:
            return "pertalite", 0.75
        elif tgs2600 > 500 and tgs2600 < 1800 and tgs2602 > 140 and tgs2610 > 1800:
            return "pertamax", 0.75
        else:
            return "unknown", 0.60
    
    def save_to_csv(self, timestamp, sensor_data, predicted_gas, confidence, method):
        """Enhanced CSV saving with method tracking"""
        try:
            row = [
                timestamp,
                sensor_data['TGS_2600']['voltage'],
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['voltage'],
                sensor_data['TGS_2602']['ppm'],
                sensor_data['TGS_2610']['voltage'],
                sensor_data['TGS_2610']['ppm'],
                predicted_gas,
                confidence,
                method
            ]
            
            import os
            write_header = not os.path.exists(self.csv_filename)
            
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                
                if write_header:
                    header = [
                        'timestamp',
                        'TGS2600_voltage', 'TGS2600_ppm',
                        'TGS2602_voltage', 'TGS2602_ppm',
                        'TGS2610_voltage', 'TGS2610_ppm',
                        'predicted_gas', 'confidence', 'method'
                    ]
                    writer.writerow(header)
                
                writer.writerow(row)
                
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def run_detection(self, interval=2.0, method='hybrid'):
        """Enhanced detection loop with method selection"""
        
        print(f"\n=== Enhanced Gas Detection System ===")
        print(f"Classification Method: {method.upper()}")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        
        # Display current rules if adaptive classifier is trained
        if self.adaptive_classifier.is_trained:
            self.adaptive_classifier.print_current_rules()
        
        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Read sensors
                sensor_data = self.read_all_sensors()
                
                if sensor_data:
                    # Predict with selected method
                    predicted_gas, confidence, used_method = self.predict_gas_type(sensor_data, method)
                    
                    # Display results
                    print(f"\n[{timestamp}] - Method: {used_method}")
                    print(f"TGS2600: {sensor_data['TGS_2600']['ppm']:6.1f} ppm ({sensor_data['TGS_2600']['voltage']:.3f}V)")
                    print(f"TGS2602: {sensor_data['TGS_2602']['ppm']:6.1f} ppm ({sensor_data['TGS_2602']['voltage']:.3f}V)")
                    print(f"TGS2610: {sensor_data['TGS_2610']['ppm']:6.1f} ppm ({sensor_data['TGS_2610']['voltage']:.3f}V)")
                    print(f"→ Predicted: {predicted_gas.upper()} (Confidence: {confidence:.2f})")
                    
                    # Save to CSV
                    self.save_to_csv(timestamp, sensor_data, predicted_gas, confidence, used_method)
                
                else:
                    print(f"[{timestamp}] Error reading sensors")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\nSystem stopped by user")
            print(f"Data saved to: {self.csv_filename}")
        except Exception as e:
            print(f"\nError in main loop: {e}")
        finally:
            print("Enhanced gas detection system shutdown complete")


def main():
    """Main function with method selection"""
    print("=== Enhanced USV Gas Detection System ===")
    print("Raspberry Pi 4 + ADS1115 + TGS Sensor Array")
    print("with Adaptive Rule-Based Classification")
    print("-" * 60)
    
    # Initialize enhanced system
    detector = EnhancedGasDetectionSystem()
    
    # Load model and training data
    detector.load_model_and_data()
    
    # Select classification method
    print("\nAvailable classification methods:")
    print("1. adaptive - Use adaptive rules (recommended)")
    print("2. hybrid - Combine adaptive + tree (best accuracy)")  
    print("3. tree - Decision tree only")
    print("4. original - Original fixed rules")
    
    try:
        choice = input("\nSelect method (1-4) or press Enter for hybrid: ").strip()
        method_map = {
            '1': 'adaptive',
            '2': 'hybrid', 
            '3': 'tree',
            '4': 'original',
            '': 'hybrid'
        }
        method = method_map.get(choice, 'hybrid')
    except:
        method = 'hybrid'
    
    print(f"Selected method: {method}")
    
    # Start detection
    detector.run_detection(interval=2.0, method=method)


if __name__ == "__main__":
    main()