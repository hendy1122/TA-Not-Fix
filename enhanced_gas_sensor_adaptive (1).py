#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System with Voltage Transformation & Hardware Integration
Solusi untuk preserve training data saat hardware adjustment (potentiometer tuning)
Version: 4.0 with Voltage Transformation & Hardware Bridge
"""

import time
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import logging
import math
from pathlib import Path

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
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import joblib

class VoltageTransformationManager:
    """Advanced Voltage Transformation Manager untuk preserve training data saat hardware adjustment"""
    
    def __init__(self, logger):
        self.logger = logger
        self.voltage_history = {}
        self.transformation_matrices = {}
        self.hardware_adjustments = []
        self.response_time_improvements = {}
        
        # Load previous transformation data
        self.load_transformation_data()
        
    def detect_hardware_adjustment(self, sensor_array):
        """Detect potentiometer adjustment dari voltage change pattern"""
        self.logger.info("🔧 Detecting hardware adjustment (potentiometer tuning)...")
        
        current_voltages = {}
        baseline_changes = {}
        
        # Read current voltages
        for sensor_name, config in sensor_array.sensor_config.items():
            current_voltage = config['channel'].voltage
            old_baseline = config.get('baseline_voltage', current_voltage)
            
            current_voltages[sensor_name] = current_voltage
            voltage_change = current_voltage - old_baseline
            baseline_changes[sensor_name] = {
                'old_voltage': old_baseline,
                'new_voltage': current_voltage,
                'change': voltage_change,
                'change_percent': abs(voltage_change / old_baseline) * 100 if old_baseline > 0 else 0
            }
        
        # Analyze if hardware adjustment occurred
        adjustment_detected = False
        adjusted_sensors = []
        
        print("\n🔧 HARDWARE ADJUSTMENT DETECTION:")
        print("-" * 60)
        
        for sensor_name, change_data in baseline_changes.items():
            change_percent = change_data['change_percent']
            
            print(f"{sensor_name}:")
            print(f"  Old Baseline: {change_data['old_voltage']:.3f}V")
            print(f"  New Voltage:  {change_data['new_voltage']:.3f}V")
            print(f"  Change:       {change_data['change']:+.3f}V ({change_percent:.1f}%)")
            
            # Hardware adjustment criteria
            if change_percent > 15:  # >15% voltage change suggests potentiometer adjustment
                adjustment_detected = True
                adjusted_sensors.append(sensor_name)
                print(f"  Status:       🔧 HARDWARE ADJUSTMENT DETECTED")
            elif change_percent > 5:
                print(f"  Status:       ⚠️  Significant change - possible adjustment")
            else:
                print(f"  Status:       ✅ Stable - no adjustment")
            print()
        
        if adjustment_detected:
            print(f"🎯 HARDWARE ADJUSTMENT CONFIRMED for: {', '.join(adjusted_sensors)}")
            print("✅ System can preserve existing training data with voltage transformation!")
            
            # Record adjustment
            adjustment_record = {
                'timestamp': datetime.now().isoformat(),
                'type': 'potentiometer_adjustment',
                'sensors_adjusted': adjusted_sensors,
                'voltage_changes': baseline_changes,
                'adjustment_reason': 'improve_response_time'
            }
            
            self.hardware_adjustments.append(adjustment_record)
            return True, baseline_changes
        else:
            print("✅ No significant hardware adjustment detected")
            return False, baseline_changes
            
    def calculate_transformation_matrix(self, old_voltages, new_voltages, training_data):
        """Calculate transformation matrix untuk convert old training data ke new voltage reference"""
        self.logger.info("🧮 Calculating voltage transformation matrix...")
        
        transformation_factors = {}
        
        for sensor_name in old_voltages.keys():
            old_v = old_voltages[sensor_name]
            new_v = new_voltages[sensor_name]
            
            if old_v > 0 and new_v > 0:
                # Calculate voltage transformation factor
                voltage_factor = new_v / old_v
                
                # Calculate resistance transformation factor
                # Rs_old = Rl * (Vcc - V_old) / V_old
                # Rs_new = Rl * (Vcc - V_new) / V_new
                # Factor = Rs_new / Rs_old
                
                Vcc = 5.0
                resistance_factor = ((Vcc - new_v) / new_v) / ((Vcc - old_v) / old_v)
                
                # Calculate PPM transformation based on sensitivity curves
                ppm_factor = self.calculate_ppm_transformation_factor(sensor_name, old_v, new_v)
                
                transformation_factors[sensor_name] = {
                    'voltage_factor': voltage_factor,
                    'resistance_factor': resistance_factor,
                    'ppm_factor': ppm_factor,
                    'old_voltage': old_v,
                    'new_voltage': new_v
                }
                
                self.logger.info(f"{sensor_name} transformation factors:")
                self.logger.info(f"  Voltage: {voltage_factor:.3f}")
                self.logger.info(f"  Resistance: {resistance_factor:.3f}")
                self.logger.info(f"  PPM: {ppm_factor:.3f}")
        
        return transformation_factors
        
    def calculate_ppm_transformation_factor(self, sensor_name, old_voltage, new_voltage):
        """Calculate PPM transformation factor based on sensor sensitivity curves"""
        
        # Sensor-specific sensitivity analysis
        if sensor_name == 'TGS2600':
            # TGS2600 logarithmic response: PPM ∝ (Rs/R0)^-2.5
            # Lower voltage = lower Rs = higher sensitivity
            sensitivity_improvement = (old_voltage / new_voltage) ** 2.5
            
        elif sensor_name == 'TGS2602':
            # TGS2602 logarithmic response: PPM ∝ (Rs/R0)^-1.8
            sensitivity_improvement = (old_voltage / new_voltage) ** 1.8
            
        elif sensor_name == 'TGS2610':
            # TGS2610 logarithmic response: PPM ∝ (Rs/R0)^-1.2
            sensitivity_improvement = (old_voltage / new_voltage) ** 1.2
            
        else:
            # Default linear approximation
            sensitivity_improvement = old_voltage / new_voltage
        
        return sensitivity_improvement
        
    def transform_training_data(self, training_data, transformation_factors):
        """Transform existing training data ke new voltage reference"""
        self.logger.info("🔄 Transforming existing training data to new voltage reference...")
        
        if training_data is None or transformation_factors is None:
            return training_data
            
        transformed_data = training_data.copy()
        transformation_applied = False
        
        print("\n🔄 TRAINING DATA TRANSFORMATION:")
        print("-" * 50)
        
        for sensor_name, factors in transformation_factors.items():
            voltage_col = f'{sensor_name}_voltage'
            resistance_col = f'{sensor_name}_resistance'
            compensated_resistance_col = f'{sensor_name}_compensated_resistance'
            ppm_col = f'{sensor_name}_ppm'
            
            if voltage_col in transformed_data.columns:
                # Transform voltage data (direct scaling)
                old_voltages = transformed_data[voltage_col].copy()
                transformed_data[voltage_col] = old_voltages * factors['voltage_factor']
                
                print(f"{sensor_name} Voltage:")
                print(f"  Old range: {old_voltages.min():.3f}V - {old_voltages.max():.3f}V")
                print(f"  New range: {transformed_data[voltage_col].min():.3f}V - {transformed_data[voltage_col].max():.3f}V")
                transformation_applied = True
                
            if resistance_col in transformed_data.columns:
                # Transform resistance data
                old_resistances = transformed_data[resistance_col].copy()
                transformed_data[resistance_col] = old_resistances * factors['resistance_factor']
                
                print(f"{sensor_name} Resistance:")
                print(f"  Old range: {old_resistances.min():.0f}Ω - {old_resistances.max():.0f}Ω")
                print(f"  New range: {transformed_data[resistance_col].min():.0f}Ω - {transformed_data[resistance_col].max():.0f}Ω")
                
            if compensated_resistance_col in transformed_data.columns:
                # Transform compensated resistance data
                old_comp_resistances = transformed_data[compensated_resistance_col].copy()
                transformed_data[compensated_resistance_col] = old_comp_resistances * factors['resistance_factor']
                
            if ppm_col in transformed_data.columns:
                # Transform PPM data (inverse relationship due to improved sensitivity)
                old_ppm = transformed_data[ppm_col].copy()
                transformed_data[ppm_col] = old_ppm / factors['ppm_factor']
                
                print(f"{sensor_name} PPM:")
                print(f"  Old range: {old_ppm.min():.1f} - {old_ppm.max():.1f} ppm")
                print(f"  New range: {transformed_data[ppm_col].min():.1f} - {transformed_data[ppm_col].max():.1f} ppm")
                print(f"  Sensitivity improved: {factors['ppm_factor']:.2f}x")
                
            print()
        
        if transformation_applied:
            # Save transformed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"data/training_data_backup_{timestamp}.csv"
            training_data.to_csv(backup_file, index=False)
            
            transformed_file = f"data/training_data_transformed_{timestamp}.csv"
            transformed_data.to_csv(transformed_file, index=False)
            
            print(f"✅ Original data backed up: {backup_file}")
            print(f"✅ Transformed data saved: {transformed_file}")
            
            # Update transformation matrices
            self.transformation_matrices[timestamp] = transformation_factors
            self.save_transformation_data()
            
        return transformed_data
        
    def apply_hardware_adjustment_workflow(self, sensor_array):
        """Complete workflow untuk handle hardware adjustment"""
        self.logger.info("🔧 Starting hardware adjustment workflow...")
        
        print("\n" + "="*80)
        print("🔧 HARDWARE ADJUSTMENT WORKFLOW")
        print("Preserve Training Data saat Potentiometer Adjustment")
        print("="*80)
        
        # Step 1: Detect adjustment
        adjustment_detected, voltage_changes = self.detect_hardware_adjustment(sensor_array)
        
        if not adjustment_detected:
            print("✅ No hardware adjustment detected - continue normal operation")
            return False
            
        # Step 2: Confirm with user
        print("\n🎯 HARDWARE ADJUSTMENT WORKFLOW OPTIONS:")
        print("1. Transform existing training data (RECOMMENDED)")
        print("2. Keep old data + recalibrate + retrain")
        print("3. Fresh start (collect all data again)")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            return self.execute_data_transformation_workflow(sensor_array, voltage_changes)
        elif choice == '2':
            return self.execute_hybrid_workflow(sensor_array)
        elif choice == '3':
            return self.execute_fresh_start_workflow(sensor_array)
        else:
            print("Invalid choice - defaulting to data transformation")
            return self.execute_data_transformation_workflow(sensor_array, voltage_changes)
            
    def execute_data_transformation_workflow(self, sensor_array, voltage_changes):
        """Execute data transformation workflow"""
        self.logger.info("🔄 Executing data transformation workflow...")
        
        # Step 1: Load existing training data
        training_data = sensor_array.load_training_data()
        if training_data is None:
            print("❌ No existing training data found - need to collect fresh data")
            return False
            
        # Step 2: Calculate transformation factors
        old_voltages = {name: change['old_voltage'] for name, change in voltage_changes.items()}
        new_voltages = {name: change['new_voltage'] for name, change in voltage_changes.items()}
        
        transformation_factors = self.calculate_transformation_matrix(
            old_voltages, new_voltages, training_data
        )
        
        # Step 3: Transform training data
        transformed_data = self.transform_training_data(training_data, transformation_factors)
        
        if transformed_data is not None:
            # Step 4: Update sensor configuration with new baselines
            self.update_sensor_baselines(sensor_array, new_voltages)
            
            # Step 5: Retrain model with transformed data
            print("\n🤖 Retraining model with transformed data...")
            if self.retrain_model_with_transformed_data(sensor_array, transformed_data):
                print("✅ HARDWARE ADJUSTMENT WORKFLOW COMPLETED SUCCESSFULLY!")
                print("🎯 Benefits achieved:")
                print("  - Training data preserved (no data collection needed)")
                print("  - Improved sensor response time")
                print("  - Enhanced sensitivity")
                print("  - Model automatically updated")
                return True
            else:
                print("❌ Model retraining failed")
                return False
        else:
            print("❌ Data transformation failed")
            return False
            
    def execute_hybrid_workflow(self, sensor_array):
        """Execute hybrid workflow (keep old + new calibration)"""
        print("\n🔄 HYBRID WORKFLOW: Old data + New calibration")
        
        # Recalibrate with new voltage settings
        print("Step 1: Recalibrating with new voltage settings...")
        duration = int(input("Calibration duration (seconds, default 300): ") or 300)
        sensor_array.calibrate_sensors(duration)
        
        # Option to collect additional training data
        collect_new = input("Collect additional training data for problem gases? (y/n): ").lower()
        if collect_new == 'y':
            gas_types = input("Enter gas types (comma separated): ").split(',')
            for gas_type in gas_types:
                gas_type = gas_type.strip()
                print(f"Collecting data for {gas_type}...")
                sensor_array.collect_training_data(gas_type, 300)
        
        # Retrain model
        print("Retraining model with combined data...")
        return sensor_array.train_model()
        
    def execute_fresh_start_workflow(self, sensor_array):
        """Execute fresh start workflow"""
        print("\n🆕 FRESH START WORKFLOW")
        
        # Reset all data
        confirm = input("This will delete all existing data. Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Fresh start cancelled")
            return False
            
        # Reset adaptive data
        sensor_array.drift_manager.drift_compensation_factors = {}
        sensor_array.drift_manager.baseline_history = {}
        sensor_array.adaptive_manager.adaptation_history = []
        
        print("✅ All adaptive data reset - ready for fresh start")
        print("Please run:")
        print("1. Menu 1 → Calibration")
        print("2. Menu 2 → Collect training data for all gas types")
        print("3. Menu 3 → Train model")
        
        return True
        
    def update_sensor_baselines(self, sensor_array, new_voltages):
        """Update sensor baselines dengan new voltage settings"""
        for sensor_name, new_voltage in new_voltages.items():
            if sensor_name in sensor_array.sensor_config:
                # Update baseline voltage
                sensor_array.sensor_config[sensor_name]['baseline_voltage'] = new_voltage
                
                # Calculate new R0 from new voltage
                new_resistance = sensor_array.voltage_to_resistance(
                    new_voltage, 
                    sensor_array.sensor_config[sensor_name]['load_resistance']
                )
                sensor_array.sensor_config[sensor_name]['R0'] = new_resistance
                
                self.logger.info(f"{sensor_name} baseline updated:")
                self.logger.info(f"  New voltage: {new_voltage:.3f}V")
                self.logger.info(f"  New R0: {new_resistance:.1f}Ω")
                
    def retrain_model_with_transformed_data(self, sensor_array, transformed_data):
        """Retrain model dengan transformed training data"""
        try:
            # Temporarily replace training data
            original_load_method = sensor_array.load_training_data
            sensor_array.load_training_data = lambda: transformed_data
            
            # Retrain model
            success = sensor_array.train_model()
            
            # Restore original method
            sensor_array.load_training_data = original_load_method
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in model retraining: {e}")
            return False
            
    def estimate_response_time_improvement(self, old_voltage, new_voltage, sensor_name):
        """Estimate response time improvement dari voltage adjustment"""
        # Response time inversely related to sensitivity
        # Lower voltage = higher sensitivity = faster response
        
        if old_voltage <= new_voltage:
            return 1.0  # No improvement
            
        # Calculate improvement factor based on voltage ratio
        voltage_ratio = old_voltage / new_voltage
        
        # Sensor-specific response time factors
        response_factors = {
            'TGS2600': 2.0,  # More sensitive to voltage changes
            'TGS2602': 1.8,
            'TGS2610': 1.5
        }
        
        factor = response_factors.get(sensor_name, 1.7)
        improvement = voltage_ratio ** (1/factor)
        
        return improvement
        
    def save_transformation_data(self):
        """Save transformation data"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'voltage_history': self.voltage_history,
            'transformation_matrices': self.transformation_matrices,
            'hardware_adjustments': self.hardware_adjustments,
            'response_time_improvements': self.response_time_improvements
        }
        
        try:
            with open('voltage_transformation_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info("Voltage transformation data saved")
        except Exception as e:
            self.logger.error(f"Error saving transformation data: {e}")
            
    def load_transformation_data(self):
        """Load transformation data"""
        try:
            with open('voltage_transformation_data.json', 'r') as f:
                data = json.load(f)
            
            self.voltage_history = data.get('voltage_history', {})
            self.transformation_matrices = data.get('transformation_matrices', {})
            self.hardware_adjustments = data.get('hardware_adjustments', [])
            self.response_time_improvements = data.get('response_time_improvements', {})
            
            self.logger.info("Voltage transformation data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No previous transformation data found")
        except Exception as e:
            self.logger.error(f"Error loading transformation data: {e}")

class AdaptiveModelManager:
    """Advanced Adaptive Model Manager untuk update model tanpa collecting data ulang"""
    
    def __init__(self, logger):
        self.logger = logger
        self.adaptation_history = []
        self.baseline_shift_patterns = {}
        self.model_confidence_threshold = 0.6
        self.adaptation_data = {}
        self.last_adaptation = None
        
        # Load previous adaptation data
        self.load_adaptation_data()
        
    def analyze_prediction_confidence(self, recent_predictions):
        """Analisis confidence prediksi untuk deteksi drift model"""
        if not recent_predictions:
            return 1.0, "No recent predictions"
            
        confidences = [p['confidence'] for p in recent_predictions]
        avg_confidence = np.mean(confidences)
        confidence_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]
        
        status = "EXCELLENT" if avg_confidence > 0.8 else \
                "GOOD" if avg_confidence > 0.6 else \
                "MODERATE" if avg_confidence > 0.4 else "POOR"
                
        return avg_confidence, status, confidence_trend
        
    def detect_model_drift(self, sensor_array, num_samples=20):
        """Deteksi apakah model perlu adaptasi berdasarkan confidence pattern"""
        self.logger.info("🔍 Detecting model drift through confidence analysis...")
        
        predictions = []
        
        print(f"Analyzing {num_samples} predictions for model drift detection...")
        
        for i in range(num_samples):
            readings = sensor_array.read_sensors()
            predicted_gas, confidence = sensor_array.predict_gas(readings)
            
            predictions.append({
                'timestamp': datetime.now(),
                'prediction': predicted_gas,
                'confidence': confidence,
                'readings': readings
            })
            
            print(f"\rProgress: {i+1}/{num_samples} | Current: {predicted_gas} ({confidence:.2f})", end="")
            time.sleep(2)
        
        print()
        
        # Analyze confidence pattern
        avg_confidence, status, trend = self.analyze_prediction_confidence(predictions)
        
        # Detect if adaptation is needed
        adaptation_needed = False
        reasons = []
        
        if avg_confidence < self.model_confidence_threshold:
            adaptation_needed = True
            reasons.append(f"Low average confidence: {avg_confidence:.2f}")
            
        if trend < -0.01:  # Decreasing confidence trend
            adaptation_needed = True
            reasons.append(f"Declining confidence trend: {trend:.3f}")
            
        # Check for inconsistent predictions
        prediction_changes = sum(1 for i in range(1, len(predictions)) 
                               if predictions[i]['prediction'] != predictions[i-1]['prediction'])
        instability = prediction_changes / len(predictions)
        
        if instability > 0.3:  # More than 30% prediction changes
            adaptation_needed = True
            reasons.append(f"High prediction instability: {instability:.1%}")
        
        print(f"\n📊 MODEL DRIFT ANALYSIS:")
        print(f"Average Confidence: {avg_confidence:.3f} ({status})")
        print(f"Confidence Trend: {trend:+.3f}")
        print(f"Prediction Instability: {instability:.1%}")
        
        if adaptation_needed:
            print(f"⚠️  MODEL ADAPTATION NEEDED!")
            print(f"Reasons: {'; '.join(reasons)}")
        else:
            print(f"✅ Model performing well - no adaptation needed")
            
        return adaptation_needed, reasons, predictions
        
    def quick_model_adaptation(self, sensor_array, gas_type_hint=None, samples=30):
        """Quick model adaptation tanpa full retraining"""
        self.logger.info("🔄 Performing quick model adaptation...")
        
        if gas_type_hint:
            print(f"Collecting adaptation data for: {gas_type_hint}")
        else:
            print("Collecting general adaptation data...")
            gas_type_hint = input("Enter current gas type (or 'normal' for clean air): ").strip().lower()
        
        print(f"Taking {samples} samples for model adaptation...")
        print("Ensure stable gas environment during sampling...")
        
        adaptation_samples = []
        
        for i in range(samples):
            readings = sensor_array.read_sensors()
            
            # Create feature vector
            features = self.create_feature_vector(readings, sensor_array)
            
            adaptation_samples.append({
                'features': features,
                'gas_type': gas_type_hint,
                'timestamp': datetime.now(),
                'readings': readings
            })
            
            print(f"\rCollecting: {i+1}/{samples}", end="")
            time.sleep(1)
        
        print()
        
        # Apply adaptation to model
        success = self.apply_model_adaptation(adaptation_samples, sensor_array)
        
        if success:
            self.logger.info("✅ Quick model adaptation completed successfully")
            # Save adaptation data
            self.save_adaptation_data()
            return True
        else:
            self.logger.error("❌ Model adaptation failed")
            return False
            
    def create_feature_vector(self, readings, sensor_array):
        """Create feature vector dari readings"""
        features = []
        
        # Basic sensor features
        for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
            if sensor_name in readings:
                data = readings[sensor_name]
                features.extend([
                    data.get('voltage', 0),
                    data.get('resistance', 0),
                    data.get('compensated_resistance', 0),
                    data.get('rs_r0_ratio', 0) if data.get('rs_r0_ratio') else 0,
                    data.get('ppm', 0),
                    data.get('drift_factor', 1.0)
                ])
        
        # Environmental features
        features.extend([
            sensor_array.current_temperature,
            sensor_array.current_humidity
        ])
        
        return np.array(features)
        
    def apply_model_adaptation(self, adaptation_samples, sensor_array):
        """Apply adaptation ke existing model"""
        try:
            if not sensor_array.is_model_trained:
                self.logger.error("No trained model to adapt")
                return False
                
            # Prepare adaptation data
            X_adapt = np.array([sample['features'] for sample in adaptation_samples])
            y_adapt = np.array([sample['gas_type'] for sample in adaptation_samples])
            
            # Handle missing values
            X_adapt = np.nan_to_num(X_adapt, nan=0.0)
            
            # Scale features using existing scaler
            X_adapt_scaled = sensor_array.scaler.transform(X_adapt)
            
            # Get current model predictions for comparison
            y_pred_before = sensor_array.model.predict(X_adapt_scaled)
            accuracy_before = accuracy_score(y_adapt, y_pred_before)
            
            # Incremental learning approach - add new samples to training
            # Load existing training data if possible
            try:
                existing_data = sensor_array.load_training_data()
                if existing_data is not None:
                    # Create features from existing data
                    feature_columns = []
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        feature_columns.extend([
                            f'{sensor}_voltage', f'{sensor}_resistance', 
                            f'{sensor}_compensated_resistance', f'{sensor}_rs_r0_ratio',
                            f'{sensor}_ppm', f'{sensor}_drift_factor'
                        ])
                    feature_columns.extend(['temperature', 'humidity'])
                    
                    # Filter available columns
                    available_columns = [col for col in feature_columns if col in existing_data.columns]
                    
                    # Add default drift factors if not present
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        drift_col = f'{sensor}_drift_factor'
                        if drift_col not in existing_data.columns:
                            existing_data[drift_col] = 1.0
                            if drift_col in feature_columns:
                                available_columns.append(drift_col)
                    
                    X_existing = existing_data[available_columns].values
                    y_existing = existing_data['gas_type'].values
                    
                    # Handle missing values
                    X_existing = np.nan_to_num(X_existing, nan=0.0)
                    
                    # Combine existing and adaptation data
                    X_combined = np.vstack([X_existing, X_adapt])
                    y_combined = np.hstack([y_existing, y_adapt])
                    
                    # Re-scale all data
                    X_combined_scaled = sensor_array.scaler.fit_transform(X_combined)
                    
                    # Retrain model with combined data
                    sensor_array.model.fit(X_combined_scaled, y_combined)
                    
                    self.logger.info(f"Model adapted with {len(adaptation_samples)} new samples")
                    self.logger.info(f"Total training samples: {len(X_combined)}")
                    
                else:
                    # Just partial fit with new data
                    sensor_array.model.fit(X_adapt_scaled, y_adapt)
                    self.logger.info(f"Model adapted with {len(adaptation_samples)} samples (no existing data)")
                    
            except Exception as e:
                self.logger.warning(f"Could not load existing training data: {e}")
                # Just fit with adaptation data
                sensor_array.model.fit(X_adapt_scaled, y_adapt)
                
            # Test adaptation effectiveness
            y_pred_after = sensor_array.model.predict(X_adapt_scaled)
            accuracy_after = accuracy_score(y_adapt, y_pred_after)
            
            self.logger.info(f"Adaptation accuracy: {accuracy_before:.3f} → {accuracy_after:.3f}")
            
            # Save adapted model
            joblib.dump(sensor_array.model, 'models/gas_classifier.pkl')
            joblib.dump(sensor_array.scaler, 'models/scaler.pkl')
            
            # Record adaptation
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'samples_count': len(adaptation_samples),
                'gas_type': adaptation_samples[0]['gas_type'],
                'accuracy_before': accuracy_before,
                'accuracy_after': accuracy_after,
                'adaptation_type': 'quick_adaptation'
            }
            
            self.adaptation_history.append(adaptation_record)
            self.last_adaptation = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model adaptation: {e}")
            return False
            
    def auto_baseline_adjustment(self, sensor_array, current_readings):
        """Automatic baseline adjustment based on clean air detection"""
        # Detect if current environment is clean air
        is_clean_air = self.detect_clean_air_environment(current_readings)
        
        if is_clean_air:
            self.logger.info("🔄 Auto-adjusting baselines - clean air detected")
            
            # Update baselines
            for sensor_name, data in current_readings.items():
                if sensor_name in sensor_array.sensor_config:
                    current_voltage = data['voltage']
                    current_resistance = data['resistance']
                    
                    # Get existing baseline
                    existing_baseline = sensor_array.sensor_config[sensor_name].get('baseline_voltage', 0.4)
                    existing_r0 = sensor_array.sensor_config[sensor_name].get('R0', 1000)
                    
                    # Calculate adjustment factor
                    voltage_drift = abs(current_voltage - existing_baseline) / existing_baseline
                    
                    if 0.05 < voltage_drift < 0.3:  # 5-30% drift
                        # Apply gradual adjustment
                        adjustment_factor = 0.1  # 10% adjustment per update
                        new_baseline = existing_baseline + (current_voltage - existing_baseline) * adjustment_factor
                        new_r0 = existing_r0 + (current_resistance - existing_r0) * adjustment_factor
                        
                        sensor_array.sensor_config[sensor_name]['baseline_voltage'] = new_baseline
                        sensor_array.sensor_config[sensor_name]['R0'] = new_r0
                        
                        self.logger.info(f"{sensor_name}: Baseline adjusted {existing_baseline:.3f}V → {new_baseline:.3f}V")
                        
            return True
        return False
        
    def detect_clean_air_environment(self, readings):
        """Detect if current environment is clean air"""
        # Simple heuristic: if all sensors show low PPM and stable readings
        ppm_values = []
        stability_check = True
        
        for sensor_name, data in readings.items():
            if sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm = data.get('ppm', 0)
                ppm_values.append(ppm)
                
                # Check if reading is stable (low drift compensation)
                drift_factor = data.get('drift_factor', 1.0)
                if abs(1 - drift_factor) > 0.15:  # More than 15% drift
                    stability_check = False
        
        avg_ppm = np.mean(ppm_values) if ppm_values else 0
        max_ppm = max(ppm_values) if ppm_values else 0
        
        # Clean air criteria
        is_clean = (avg_ppm < 5 and max_ppm < 10 and stability_check)
        
        return is_clean
        
    def save_adaptation_data(self):
        """Save adaptation data"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'adaptation_history': self.adaptation_history,
            'baseline_shift_patterns': self.baseline_shift_patterns,
            'last_adaptation': self.last_adaptation.isoformat() if self.last_adaptation else None,
            'model_confidence_threshold': self.model_confidence_threshold
        }
        
        try:
            with open('adaptive_model_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info("Adaptation data saved")
        except Exception as e:
            self.logger.error(f"Error saving adaptation data: {e}")
            
    def load_adaptation_data(self):
        """Load adaptation data"""
        try:
            with open('adaptive_model_data.json', 'r') as f:
                data = json.load(f)
            
            self.adaptation_history = data.get('adaptation_history', [])
            self.baseline_shift_patterns = data.get('baseline_shift_patterns', {})
            self.model_confidence_threshold = data.get('model_confidence_threshold', 0.6)
            
            if data.get('last_adaptation'):
                self.last_adaptation = datetime.fromisoformat(data['last_adaptation'])
                
            self.logger.info("Adaptation data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No previous adaptation data found")
        except Exception as e:
            self.logger.error(f"Error loading adaptation data: {e}")

class DriftCompensationManager:
    """Enhanced Drift Compensation Manager dengan voltage transformation support"""
    
    def __init__(self, logger):
        self.logger = logger
        self.baseline_history = {}
        self.drift_compensation_factors = {}
        self.last_calibration_time = None
        self.daily_check_done = False
        self.adaptive_drift_learning = True
        
        # Load previous drift data
        self.load_drift_data()
        
    def is_daily_check_needed(self):
        """Check jika daily drift check perlu dilakukan"""
        if not self.daily_check_done:
            return True
            
        # Check if it's a new day
        if hasattr(self, 'last_check_date'):
            today = datetime.now().date()
            if today != self.last_check_date:
                return True
                
        return False
        
    def smart_drift_detection(self, sensor_array, adaptive_manager):
        """Smart drift detection dengan adaptive learning"""
        self.logger.info("🧠 Performing smart drift detection with adaptive learning...")
        
        # First check model confidence
        adaptation_needed, reasons, predictions = adaptive_manager.detect_model_drift(sensor_array, 15)
        
        if adaptation_needed:
            print("\n⚠️  SMART DRIFT DETECTION RESULTS:")
            print("Model confidence has decreased, likely due to sensor drift.")
            print("Recommended actions:")
            print("1. Quick drift compensation (automatic)")
            print("2. Model adaptation with current environment")
            print("3. Full recalibration if needed")
            
            return True, reasons
        else:
            print("\n✅ Smart drift detection: Sensors and model performing well")
            return False, []
            
    def intelligent_drift_compensation(self, sensor_array, adaptive_manager):
        """Intelligent drift compensation dengan adaptive learning"""
        self.logger.info("🤖 Performing intelligent drift compensation...")
        
        # Measure current baseline
        print("\n⚠️  IMPORTANT: Ensure sensors are in CLEAN AIR")
        response = input("Are sensors in clean air environment? (y/n): ").lower()
        if response != 'y':
            print("❌ Drift compensation cancelled")
            return False
            
        current_baseline = self.measure_clean_air_baseline(sensor_array, duration=60)
        
        # Analyze drift patterns
        compensation_applied = False
        adaptation_needed = False
        
        print("\n🧠 INTELLIGENT DRIFT ANALYSIS:")
        print("-" * 50)
        
        for sensor_name, current_voltage in current_baseline.items():
            if sensor_name in self.baseline_history and self.baseline_history[sensor_name]:
                # Get baseline history
                history = self.baseline_history[sensor_name]
                previous_voltage = history[-1]
                
                # Calculate drift
                voltage_drift = current_voltage - previous_voltage
                drift_percent = abs(voltage_drift / previous_voltage) * 100
                
                print(f"{sensor_name}:")
                print(f"  Previous: {previous_voltage:.3f}V")
                print(f"  Current:  {current_voltage:.3f}V")
                print(f"  Drift:    {voltage_drift:+.3f}V ({drift_percent:.1f}%)")
                
                # Intelligent compensation decision
                if drift_percent > 20:  # High drift
                    print(f"  Status:   ⚠️  HIGH DRIFT - Model adaptation recommended")
                    adaptation_needed = True
                    
                elif drift_percent > 5:  # Moderate drift
                    compensation_factor = previous_voltage / current_voltage
                    self.drift_compensation_factors[sensor_name] = compensation_factor
                    compensation_applied = True
                    print(f"  Status:   🔧 Moderate drift - Auto compensation applied")
                    print(f"  Factor:   {compensation_factor:.3f}")
                    
                elif drift_percent > 2:  # Minor drift
                    if self.adaptive_drift_learning:
                        # Gradual adaptation
                        if sensor_name in self.drift_compensation_factors:
                            current_factor = self.drift_compensation_factors[sensor_name]
                            new_factor = (current_factor + previous_voltage / current_voltage) / 2
                        else:
                            new_factor = (1.0 + previous_voltage / current_voltage) / 2
                        
                        self.drift_compensation_factors[sensor_name] = new_factor
                        compensation_applied = True
                        print(f"  Status:   🔄 Minor drift - Adaptive compensation")
                        print(f"  Factor:   {new_factor:.3f}")
                    else:
                        print(f"  Status:   ✅ Minor drift - Within tolerance")
                        
                else:  # Minimal drift
                    print(f"  Status:   ✅ Excellent stability")
                    # Remove compensation if previously applied
                    if sensor_name in self.drift_compensation_factors:
                        del self.drift_compensation_factors[sensor_name]
                
                print()
        
        # Update baseline history
        for sensor_name, voltage in current_baseline.items():
            if sensor_name not in self.baseline_history:
                self.baseline_history[sensor_name] = []
            
            self.baseline_history[sensor_name].append(voltage)
            
            # Keep only last 30 days
            if len(self.baseline_history[sensor_name]) > 30:
                self.baseline_history[sensor_name].pop(0)
        
        # Mark daily check as done
        self.daily_check_done = True
        self.last_check_date = datetime.now().date()
        
        # Save drift data
        self.save_drift_data()
        
        # Intelligent action recommendations
        if adaptation_needed:
            print("🤖 INTELLIGENT RECOMMENDATION:")
            print("High drift detected - Quick model adaptation recommended")
            response = input("Perform automatic model adaptation now? (y/n): ").lower()
            if response == 'y':
                gas_type = input("Current gas environment (or 'normal' for clean air): ").strip()
                adaptive_manager.quick_model_adaptation(sensor_array, gas_type, 20)
                
        elif compensation_applied:
            print("✅ DRIFT COMPENSATION APPLIED - System automatically adjusted")
        else:
            print("✅ ALL SENSORS STABLE - No compensation needed")
            
        return True
        
    def measure_clean_air_baseline(self, sensor_array, duration=60):
        """Measure stable baseline voltage in clean air"""
        self.logger.info(f"Measuring clean air baseline for {duration} seconds...")
        
        readings = {sensor: [] for sensor in sensor_array.sensor_config.keys()}
        
        start_time = time.time()
        sample_count = 0
        
        print(f"Collecting baseline data for {duration} seconds...")
        
        while time.time() - start_time < duration:
            # Read raw sensors (without compensation)
            for sensor_name, config in sensor_array.sensor_config.items():
                voltage = config['channel'].voltage
                readings[sensor_name].append(voltage)
            
            sample_count += 1
            remaining = int(duration - (time.time() - start_time))
            
            if sample_count % 10 == 0:
                print(f"  Progress... {remaining}s remaining (samples: {sample_count})")
            
            time.sleep(2)
        
        # Calculate stable baseline using statistical filtering
        baseline = {}
        
        for sensor_name, voltages in readings.items():
            voltages_array = np.array(voltages)
            
            # Remove outliers (beyond 2 standard deviations)
            mean_v = np.mean(voltages_array)
            std_v = np.std(voltages_array)
            
            # Filter outliers
            mask = np.abs(voltages_array - mean_v) <= 2 * std_v
            filtered_voltages = voltages_array[mask]
            
            # Use median for robustness
            baseline_voltage = np.median(filtered_voltages)
            stability = (std_v / mean_v) * 100
            
            baseline[sensor_name] = baseline_voltage
            
            self.logger.info(f"{sensor_name}: {baseline_voltage:.3f}V ± {std_v:.3f}V (stability: {stability:.1f}%)")
        
        return baseline
        
    def apply_drift_compensation(self, sensor_name, raw_voltage):
        """Apply drift compensation to voltage reading"""
        if sensor_name in self.drift_compensation_factors:
            compensated_voltage = raw_voltage * self.drift_compensation_factors[sensor_name]
            return compensated_voltage
        return raw_voltage
        
    def save_drift_data(self):
        """Save drift compensation data"""
        drift_data = {
            'timestamp': datetime.now().isoformat(),
            'last_calibration_time': self.last_calibration_time.isoformat() if self.last_calibration_time else None,
            'baseline_history': self.baseline_history,
            'drift_compensation_factors': self.drift_compensation_factors,
            'daily_check_done': self.daily_check_done,
            'adaptive_drift_learning': self.adaptive_drift_learning,
            'last_check_date': self.last_check_date.isoformat() if hasattr(self, 'last_check_date') else None
        }
        
        try:
            with open('drift_compensation_data.json', 'w') as f:
                json.dump(drift_data, f, indent=2)
            self.logger.info("Drift compensation data saved")
        except Exception as e:
            self.logger.error(f"Error saving drift data: {e}")
            
    def load_drift_data(self):
        """Load drift compensation data"""
        try:
            with open('drift_compensation_data.json', 'r') as f:
                drift_data = json.load(f)
            
            self.baseline_history = drift_data.get('baseline_history', {})
            self.drift_compensation_factors = drift_data.get('drift_compensation_factors', {})
            self.daily_check_done = drift_data.get('daily_check_done', False)
            self.adaptive_drift_learning = drift_data.get('adaptive_drift_learning', True)
            
            if drift_data.get('last_calibration_time'):
                self.last_calibration_time = datetime.fromisoformat(drift_data['last_calibration_time'])
                
            if drift_data.get('last_check_date'):
                self.last_check_date = datetime.fromisoformat(drift_data['last_check_date']).date()
                
            self.logger.info("Drift compensation data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No previous drift data found - starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading drift data: {e}")

class EnhancedDatasheetGasSensorArrayV4:
    """Enhanced Gas Sensor Array with Voltage Transformation & Hardware Integration v4.0"""
    
    def __init__(self):
        """Initialize enhanced gas sensor array system with voltage transformation"""
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

            # Setup analog inputs for each sensor
            self.tgs2600 = AnalogIn(self.ads, ADS.P0)  # Channel A0
            self.tgs2602 = AnalogIn(self.ads, ADS.P1)  # Channel A1
            self.tgs2610 = AnalogIn(self.ads, ADS.P2)  # Channel A2

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
                'extended_sensitivity': 2.5
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
                'extended_sensitivity': 3.0
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
                'extended_sensitivity': 2.0
            }
        }

        # Initialize Advanced Managers
        self.drift_manager = DriftCompensationManager(self.logger)
        self.adaptive_manager = AdaptiveModelManager(self.logger)
        self.voltage_transformer = VoltageTransformationManager(self.logger)

        # Environmental compensation parameters
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0
        self.current_humidity = 65.0

        # Data storage
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.data_file = f"gas_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Machine Learning components
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False

        # Create data directory
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)

        self.logger.info("Enhanced Gas Sensor Array System v4.0 with Voltage Transformation initialized")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """Convert ADC voltage to sensor resistance using voltage divider"""
        if voltage <= 0.001:  # Avoid division by zero
            return float('inf')

        circuit_voltage = 5.0

        if voltage >= circuit_voltage:
            return 0.1  # Very low resistance

        sensor_resistance = load_resistance * (circuit_voltage - voltage) / voltage
        return max(1, sensor_resistance)  # Minimum 1Ω

    def temperature_compensation(self, sensor_name, raw_value, temperature):
        """Apply temperature compensation based on datasheet curves"""
        if not self.temp_compensation_enabled:
            return raw_value

        # Temperature compensation factors based on datasheet (simplified)
        temp_factors = {
            'TGS2600': {
                -20: 1.8, -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0,
                30: 0.95, 40: 0.9, 50: 0.85
            },
            'TGS2602': {
                -10: 1.5, 0: 1.3, 10: 1.1, 20: 1.0,
                30: 0.9, 40: 0.85, 50: 0.8, 60: 0.75
            },
            'TGS2610': {
                -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0,
                30: 0.95, 40: 0.9, 50: 0.85
            }
        }

        # Linear interpolation for temperature compensation
        temp_curve = temp_factors.get(sensor_name, {20: 1.0})
        temps = sorted(temp_curve.keys())

        if temperature <= temps[0]:
            factor = temp_curve[temps[0]]
        elif temperature >= temps[-1]:
            factor = temp_curve[temps[-1]]
        else:
            # Linear interpolation
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
        """Apply humidity compensation based on datasheet curves"""
        if not self.humidity_compensation_enabled:
            return raw_value

        # Humidity compensation factors (simplified from datasheet)
        humidity_factors = {
            'TGS2600': {35: 1.1, 65: 1.0, 95: 0.9},
            'TGS2602': {40: 1.05, 65: 1.0, 85: 0.95, 100: 0.9},
            'TGS2610': {40: 1.1, 65: 1.0, 85: 0.95}
        }

        # Linear interpolation for humidity compensation
        humidity_curve = humidity_factors.get(sensor_name, {65: 1.0})
        humidities = sorted(humidity_curve.keys())

        if humidity <= humidities[0]:
            factor = humidity_curve[humidities[0]]
        elif humidity >= humidities[-1]:
            factor = humidity_curve[humidities[-1]]
        else:
            # Linear interpolation
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
        """Enhanced PPM calculation with hybrid approach"""
        config = self.sensor_config[sensor_name]
        R0 = config.get('R0')

        if R0 is None or R0 == 0:
            return self.simplified_ppm_calculation(sensor_name, resistance)

        rs_r0_ratio = resistance / R0

        # Choose calculation mode
        if config['use_extended_mode']:
            return self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            return self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Original datasheet calculation"""
        if sensor_name == 'TGS2600':
            if gas_type == 'hydrogen' or gas_type == 'auto':
                if rs_r0_ratio < 0.3:
                    ppm = 30
                elif rs_r0_ratio > 0.9:
                    ppm = 0
                else:
                    ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                    ppm = min(ppm, 30)
            elif gas_type == 'alcohol':
                if rs_r0_ratio < 0.2:
                    ppm = 30
                elif rs_r0_ratio > 0.8:
                    ppm = 0
                else:
                    ppm = 40 * ((0.4 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 30)
            else:
                ppm = 30 * ((0.5 / rs_r0_ratio) ** 2.0)
                ppm = min(ppm, 30)

        elif sensor_name == 'TGS2602':
            if gas_type == 'alcohol' or gas_type == 'auto':
                if rs_r0_ratio < 0.08:
                    ppm = 30
                elif rs_r0_ratio > 0.9:
                    ppm = 0
                else:
                    ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                    ppm = min(ppm, 30)
            elif gas_type == 'toluene':
                ppm = 25 * ((0.2 / rs_r0_ratio) ** 1.5)
                ppm = min(ppm, 30)
            else:
                ppm = 20 * ((0.3 / rs_r0_ratio) ** 1.6)
                ppm = min(ppm, 30)

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.45:
                ppm = 25
            elif rs_r0_ratio > 0.95:
                ppm = 0
            else:
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                ppm = min(ppm, 25)
        else:
            ppm = 0

        return max(0, ppm)

    def extended_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Extended range calculation for training data"""
        config = self.sensor_config[sensor_name]
        sensitivity = config['extended_sensitivity']

        if rs_r0_ratio >= 1.0:
            return 0

        # Base calculation without upper limits
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

        # Apply gas-specific multipliers
        gas_multipliers = {
            'alcohol': 1.0,
            'pertalite': 1.3,
            'pertamax': 1.6,
            'dexlite': 1.9,
            'biosolar': 2.2,
            'hydrogen': 0.8,
            'toluene': 1.1,
            'ammonia': 0.9,
            'butane': 1.2,
            'propane': 1.4,
            'normal': 0.7
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Simplified PPM calculation when R0 is not available"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 0.4)

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

    def set_sensor_mode(self, mode='datasheet'):
        """Set calculation mode for all sensors"""
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

    def read_sensors(self):
        """Enhanced sensor reading with drift compensation and voltage transformation"""
        readings = {}

        for sensor_name, config in self.sensor_config.items():
            try:
                # Read raw voltage
                raw_voltage = config['channel'].voltage
                
                # Apply drift compensation
                compensated_voltage = self.drift_manager.apply_drift_compensation(sensor_name, raw_voltage)

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

                # Convert to PPM
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)

                # Current mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"
                
                # Drift compensation info
                drift_factor = self.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                drift_applied = abs(1 - drift_factor) > 0.01

                readings[sensor_name] = {
                    'voltage': compensated_voltage,
                    'raw_voltage': raw_voltage,
                    'resistance': resistance,
                    'compensated_resistance': compensated_resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'R0': R0,
                    'mode': current_mode,
                    'target_gases': config['target_gases'],
                    'drift_compensation_applied': drift_applied,
                    'drift_factor': drift_factor
                }

            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'raw_voltage': 0, 'resistance': 0, 'compensated_resistance': 0,
                    'rs_r0_ratio': None, 'ppm': 0, 'R0': None, 'mode': 'Error',
                    'target_gases': [], 'drift_compensation_applied': False, 'drift_factor': 1.0
                }

        return readings

    def calibrate_sensors(self, duration=600):
        """Enhanced calibration with voltage transformation support"""
        self.logger.info(f"Starting enhanced sensor calibration for {duration} seconds...")
        self.logger.info("Ensure sensors are in CLEAN AIR environment!")
        self.logger.info("Sensors should be warmed up for at least 10 minutes")

        input("Press Enter when sensors are in clean air and warmed up...")

        readings = {sensor: {'voltages': [], 'resistances': []}
                   for sensor in self.sensor_config.keys()}

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration:
            for sensor_name, config in self.sensor_config.items():
                # Use raw voltage for calibration (without drift compensation)
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])

                # Apply environmental compensation
                resistance = self.temperature_compensation(sensor_name, resistance, self.current_temperature)
                resistance = self.humidity_compensation(sensor_name, resistance, self.current_humidity)

                readings[sensor_name]['voltages'].append(voltage)
                readings[sensor_name]['resistances'].append(resistance)

            sample_count += 1
            time.sleep(2)

            remaining = int(duration - (time.time() - start_time))
            if remaining % 30 == 0 and remaining > 0:
                print(f"Calibration remaining: {remaining} seconds (Samples: {sample_count})")

        # Calculate calibration parameters
        calibration_results = {}

        for sensor_name in self.sensor_config.keys():
            voltages = readings[sensor_name]['voltages']
            resistances = readings[sensor_name]['resistances']

            # Statistical analysis
            voltage_mean = np.mean(voltages)
            voltage_std = np.std(voltages)
            resistance_mean = np.mean(resistances)
            resistance_std = np.std(resistances)

            # Set R0 (baseline resistance in clean air)
            self.sensor_config[sensor_name]['R0'] = resistance_mean
            self.sensor_config[sensor_name]['baseline_voltage'] = voltage_mean

            calibration_results[sensor_name] = {
                'R0': resistance_mean,
                'R0_std': resistance_std,
                'baseline_voltage': voltage_mean,
                'voltage_std': voltage_std,
                'sample_count': len(voltages),
                'stability': (voltage_std / voltage_mean) * 100
            }

            self.logger.info(f"{sensor_name} Calibration:")
            self.logger.info(f"  R0: {resistance_mean:.1f}Ω ± {resistance_std:.1f}Ω")
            self.logger.info(f"  Baseline Voltage: {voltage_mean:.3f}V ± {voltage_std:.3f}V")
            self.logger.info(f"  Stability: {calibration_results[sensor_name]['stability']:.2f}%")

        # Update drift manager calibration time
        self.drift_manager.last_calibration_time = datetime.now()
        
        # Reset drift compensation factors after new calibration
        self.drift_manager.drift_compensation_factors = {}
        self.drift_manager.daily_check_done = False

        # Save calibration data
        calib_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'temperature': self.current_temperature,
            'humidity': self.current_humidity,
            'sensors': calibration_results
        }

        calib_file = f"calibration/calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(calib_file, 'w') as f:
            json.dump(calib_data, f, indent=2)

        with open('sensor_calibration.json', 'w') as f:
            json.dump(calib_data, f, indent=2)

        # Save updated drift data
        self.drift_manager.save_drift_data()

        self.logger.info(f"Enhanced calibration completed and saved to {calib_file}")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)

            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']

            # Update drift manager calibration time
            if 'timestamp' in calib_data:
                self.drift_manager.last_calibration_time = datetime.fromisoformat(calib_data['timestamp'])

            self.logger.info("Calibration data loaded successfully")
            self.logger.info(f"Calibration date: {calib_data.get('timestamp', 'Unknown')}")

            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Please run calibration first.")
            return False
        except KeyError as e:
            self.logger.error(f"Invalid calibration file format: {e}")
            return False

    def collect_training_data(self, gas_type, duration=300):
        """Enhanced training data collection with voltage transformation support"""
        # Auto-switch to extended mode for training
        self.set_sensor_mode('extended')

        self.logger.info(f"Collecting enhanced training data for {gas_type} with voltage transformation support")
        self.logger.info(f"Duration: {duration}s (5 minutes optimal timing)")

        # Enhanced gas type validation
        valid_gases = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar',
                      'hydrogen', 'toluene', 'ammonia', 'butane', 'propane']
        if gas_type not in valid_gases:
            self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
            return None

        if gas_type == 'normal':
            input(f"Ensure sensors are in CLEAN AIR (no gas). Press Enter to start...")
            self.logger.info("Collecting NORMAL/CLEAN AIR data...")
        else:
            input(f"Prepare to spray {gas_type}. Press Enter to start data collection...")

        training_data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            timestamp = datetime.now()
            readings = self.read_sensors()

            # Enhanced data row with voltage transformation support
            data_row = {
                'timestamp': timestamp,
                'gas_type': gas_type,
                'temperature': self.current_temperature,
                'humidity': self.current_humidity,

                # TGS2600 data with voltage transformation info
                'TGS2600_voltage': readings['TGS2600']['voltage'],
                'TGS2600_raw_voltage': readings['TGS2600']['raw_voltage'],
                'TGS2600_resistance': readings['TGS2600']['resistance'],
                'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                'TGS2600_ppm': readings['TGS2600']['ppm'],
                'TGS2600_drift_factor': readings['TGS2600']['drift_factor'],

                # TGS2602 data with voltage transformation info
                'TGS2602_voltage': readings['TGS2602']['voltage'],
                'TGS2602_raw_voltage': readings['TGS2602']['raw_voltage'],
                'TGS2602_resistance': readings['TGS2602']['resistance'],
                'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                'TGS2602_ppm': readings['TGS2602']['ppm'],
                'TGS2602_drift_factor': readings['TGS2602']['drift_factor'],

                # TGS2610 data with voltage transformation info
                'TGS2610_voltage': readings['TGS2610']['voltage'],
                'TGS2610_raw_voltage': readings['TGS2610']['raw_voltage'],
                'TGS2610_resistance': readings['TGS2610']['resistance'],
                'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                'TGS2610_ppm': readings['TGS2610']['ppm'],
                'TGS2610_drift_factor': readings['TGS2610']['drift_factor']
            }

            training_data.append(data_row)

            # Enhanced display
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            status = "CLEAN AIR" if gas_type == 'normal' else gas_type.upper()
            
            # Show voltage transformation status
            voltage_status = "VT_READY" if hasattr(self, 'voltage_transformer') else "STANDARD"
            
            print(f"\rTime: {remaining:.1f}s | Status: {status} | Mode: {voltage_status} | "
                  f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                  f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                  f"2610: {readings['TGS2610']['ppm']:.0f}ppm", end="")

            time.sleep(1.0)

        print()

        # Save enhanced training data
        filename = f"data/training_{gas_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False)

        self.logger.info(f"Enhanced training data with voltage transformation support saved to {filename}")
        self.logger.info(f"Collected {len(training_data)} samples for {gas_type}")

        return training_data

    def load_training_data(self):
        """Load and validate training data"""
        data_files = list(Path("data").glob("training_*.csv"))
        if not data_files:
            self.logger.error("No training data files found!")
            return None

        all_data = []
        gas_counts = {}

        for file in data_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)

                # Count samples per gas type
                if 'gas_type' in df.columns:
                    gas_type = df['gas_type'].iloc[0]
                    gas_counts[gas_type] = gas_counts.get(gas_type, 0) + len(df)

                self.logger.info(f"Loaded {len(df)} samples from {file.name}")
            except Exception as e:
                self.logger.error(f"Error loading {file.name}: {e}")

        if not all_data:
            return None

        combined_data = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"Total training samples: {len(combined_data)}")
        self.logger.info("Samples per gas type:")
        for gas, count in gas_counts.items():
            self.logger.info(f"  {gas}: {count} samples")

        return combined_data

    def train_model(self):
        """Enhanced model training with voltage transformation features"""
        self.logger.info("Training enhanced ML model with voltage transformation features...")

        # Load all training data
        training_data = self.load_training_data()
        if training_data is None:
            return False

        # Enhanced feature selection (include voltage transformation features)
        feature_columns = [
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm', 'TGS2600_drift_factor',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm', 'TGS2602_drift_factor',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm', 'TGS2610_drift_factor',
            'temperature', 'humidity'
        ]

        # Filter available columns
        available_columns = [col for col in feature_columns if col in training_data.columns]
        
        # Add default drift factors if not present
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            drift_col = f'{sensor}_drift_factor'
            if drift_col not in training_data.columns:
                training_data[drift_col] = 1.0
                if drift_col in feature_columns:
                    available_columns.append(drift_col)

        X = training_data[available_columns].values
        y = training_data['gas_type'].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.logger.info(f"Training classes: {list(unique_classes)}")
        for cls, count in zip(unique_classes, class_counts):
            self.logger.info(f"  {cls}: {count} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Enhanced Random Forest
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        self.logger.info(f"Model accuracy: {accuracy:.3f}")
        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Save model and metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'feature_columns': available_columns,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': list(unique_classes),
            'model_type': 'voltage_transformation_enhanced',
            'drift_compensation_enabled': True,
            'adaptive_learning_enabled': True,
            'voltage_transformation_enabled': True
        }

        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')

        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        self.is_model_trained = True
        self.logger.info("Enhanced voltage transformation model trained and saved successfully")

        return True

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_model_trained = True
            self.logger.info("Enhanced voltage transformation model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction with voltage transformation support"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0

        # Prepare enhanced features
        try:
            # Load model metadata to get feature order
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            feature_columns = metadata['feature_columns']
        except:
            # Fallback to basic features
            feature_columns = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                             'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']

        # Build feature vector
        feature_vector = []
        for feature in feature_columns:
            if feature == 'temperature':
                feature_vector.append(self.current_temperature)
            elif feature == 'humidity':
                feature_vector.append(self.current_humidity)
            else:
                # Parse sensor and measurement type
                parts = feature.split('_')
                sensor = parts[0]
                measurement = '_'.join(parts[1:])

                if sensor in readings and measurement in readings[sensor]:
                    value = readings[sensor][measurement]
                    feature_vector.append(value if value is not None else 0.0)
                else:
                    # Default values for drift factors
                    if measurement == 'drift_factor':
                        feature_vector.append(1.0)
                    else:
                        feature_vector.append(0.0)

        features = np.array([feature_vector])

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()

        return prediction, confidence

    def continuous_monitoring_with_voltage_transformation(self, duration=None, monitoring_mode='datasheet'):
        """Enhanced continuous monitoring with voltage transformation support"""
        # Set monitoring mode
        self.set_sensor_mode(monitoring_mode)

        self.logger.info(f"Starting voltage transformation monitoring in {monitoring_mode.upper()} mode...")
        self.is_collecting = True

        # Enhanced CSV fields
        fieldnames = [
            'timestamp', 'temperature', 'humidity', 'sensor_mode',
            'TGS2600_voltage', 'TGS2600_raw_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm', 'TGS2600_drift_factor',
            'TGS2602_voltage', 'TGS2602_raw_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm', 'TGS2602_drift_factor',
            'TGS2610_voltage', 'TGS2610_raw_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm', 'TGS2610_drift_factor',
            'predicted_gas', 'confidence', 'voltage_transformation_active'
        ]

        monitoring_file = f"data/monitoring_{monitoring_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(monitoring_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()
            sample_count = 0

            try:
                while self.is_collecting:
                    timestamp = datetime.now()
                    readings = self.read_sensors()
                    predicted_gas, confidence = self.predict_gas(readings)

                    # Check if voltage transformation is active
                    voltage_transformation_active = len(self.voltage_transformer.transformation_matrices) > 0

                    # Enhanced data row with voltage transformation info
                    data_row = {
                        'timestamp': timestamp,
                        'temperature': self.current_temperature,
                        'humidity': self.current_humidity,
                        'sensor_mode': monitoring_mode,

                        'TGS2600_voltage': readings['TGS2600']['voltage'],
                        'TGS2600_raw_voltage': readings['TGS2600']['raw_voltage'],
                        'TGS2600_resistance': readings['TGS2600']['resistance'],
                        'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                        'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                        'TGS2600_ppm': readings['TGS2600']['ppm'],
                        'TGS2600_drift_factor': readings['TGS2600']['drift_factor'],

                        'TGS2602_voltage': readings['TGS2602']['voltage'],
                        'TGS2602_raw_voltage': readings['TGS2602']['raw_voltage'],
                        'TGS2602_resistance': readings['TGS2602']['resistance'],
                        'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                        'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                        'TGS2602_ppm': readings['TGS2602']['ppm'],
                        'TGS2602_drift_factor': readings['TGS2602']['drift_factor'],

                        'TGS2610_voltage': readings['TGS2610']['voltage'],
                        'TGS2610_raw_voltage': readings['TGS2610']['raw_voltage'],
                        'TGS2610_resistance': readings['TGS2610']['resistance'],
                        'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                        'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                        'TGS2610_ppm': readings['TGS2610']['ppm'],
                        'TGS2610_drift_factor': readings['TGS2610']['drift_factor'],

                        'predicted_gas': predicted_gas,
                        'confidence': confidence,
                        'voltage_transformation_active': voltage_transformation_active
                    }

                    writer.writerow(data_row)
                    sample_count += 1

                    # Enhanced display with voltage transformation info
                    drift_status = "DRIFT_COMP" if any(readings[s]['drift_compensation_applied'] for s in readings.keys()) else "RAW"
                    vt_status = "VT_ACTIVE" if voltage_transformation_active else "VT_READY"
                    
                    print(f"\r{timestamp.strftime('%H:%M:%S')} | Mode: {monitoring_mode.title()} | "
                          f"Status: {drift_status}/{vt_status} | "
                          f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                          f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                          f"2610: {readings['TGS2610']['ppm']:.0f}ppm | "
                          f"Predicted: {predicted_gas} ({confidence:.2f})", end="")

                    # Detection alert for high concentrations
                    max_ppm = max(readings['TGS2600']['ppm'], readings['TGS2602']['ppm'], readings['TGS2610']['ppm'])
                    if max_ppm > 50 and confidence > 0.7:
                        print(f"\n*** GAS DETECTION ALERT: {predicted_gas} detected! ***")

                    # Check duration
                    if duration and (time.time() - start_time) >= duration:
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")

        self.is_collecting = False
        self.logger.info(f"Enhanced voltage transformation monitoring data saved to {monitoring_file}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_collecting = False

    def set_environmental_conditions(self, temperature=None, humidity=None):
        """Set current environmental conditions for compensation"""
        if temperature is not None:
            self.current_temperature = temperature
            self.logger.info(f"Temperature set to {temperature}°C")

        if humidity is not None:
            self.current_humidity = humidity
            self.logger.info(f"Humidity set to {humidity}%RH")

def main():
    """Enhanced main function with voltage transformation workflow"""
    gas_sensor = EnhancedDatasheetGasSensorArrayV4()

    # Load existing calibration if available
    gas_sensor.load_calibration()

    # Load existing model if available
    gas_sensor.load_model()

    # Check for hardware adjustment (potentiometer changes)
    adjustment_detected, voltage_changes = gas_sensor.voltage_transformer.detect_hardware_adjustment(gas_sensor)
    if adjustment_detected:
        print("\n" + "="*80)
        print("🔧 HARDWARE ADJUSTMENT DETECTED!")
        print("Potentiometer adjustment detected - voltage optimization possible!")
        print("System can preserve all existing training data!")
        print("="*80)
        
        response = input("Handle hardware adjustment with data preservation? (y/n): ").lower()
        if response == 'y':
            gas_sensor.voltage_transformer.apply_hardware_adjustment_workflow(gas_sensor)

    # Check if daily drift check is needed
    if gas_sensor.drift_manager.is_daily_check_needed():
        print("\n" + "="*70)
        print("🧠 DAILY INTELLIGENT CHECK RECOMMENDED")
        print("Advanced system monitoring for optimal performance.")
        print("="*70)
        
        response = input("Run daily intelligent check? (y/n): ").lower()
        if response == 'y':
            gas_sensor.drift_manager.intelligent_drift_compensation(gas_sensor, gas_sensor.adaptive_manager)

    while True:
        print("\n" + "="*90)
        print("🚀 ENHANCED Gas Sensor Array System v4.0 - VOLTAGE TRANSFORMATION")
        print("Hardware Adjustment Support + Training Data Preservation")
        print("="*90)
        print("1. Calibrate sensors (Enhanced R0 determination)")
        print("2. Collect training data (Optimal timing)")
        print("3. Train machine learning model (With voltage transformation)")
        print("4. Start enhanced monitoring - Datasheet mode")
        print("5. Start enhanced monitoring - Extended mode")
        print("6. Test single reading (Detailed voltage analysis)")
        print("7. Set environmental conditions (T°C, %RH)")
        print("8. Switch sensor calculation mode")
        print("9. View comprehensive sensor diagnostics")
        print("10. Exit")
        print("-" * 60)
        print("🔧 VOLTAGE TRANSFORMATION FEATURES:")
        print("11. Detect hardware adjustment (potentiometer changes)")
        print("12. Apply voltage transformation workflow")
        print("13. Quick model adaptation (NO data collection needed!)")
        print("14. Smart drift detection & compensation")
        print("15. Comprehensive system health & voltage analysis")
        print("16. Reset all learning data")
        print("-" * 60)
        print("🎯 RESPONSE TIME OPTIMIZATION:")
        print("17. 🌟 POTENTIOMETER ADJUSTMENT HELPER")
        print("18. Estimate response time improvement")
        print("19. Voltage-to-PPM sensitivity analysis")
        print("-"*90)

        try:
            choice = input("Select option (1-19): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 600): ") or 600)
                print("⚠️  IMPORTANT: Ensure sensors are warmed up for at least 10 minutes in clean air!")
                confirm = input("Continue with calibration? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.calibrate_sensors(duration)

            elif choice == '2':
                gas_types = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar',
                           'hydrogen', 'toluene', 'ammonia', 'butane', 'propane']
                print("Available gas types:", ', '.join(gas_types))
                print("⚠️  IMPORTANT: Collect 'normal' data first for baseline!")
                gas_type = input("Enter gas type: ").strip().lower()

                if gas_type not in gas_types:
                    print("Invalid gas type!")
                    continue

                duration = int(input("Collection duration (seconds, default 300): ") or 300)
                gas_sensor.collect_training_data(gas_type, duration)

            elif choice == '3':
                print("🤖 Training voltage transformation model...")
                if gas_sensor.train_model():
                    print("✅ Enhanced voltage transformation model training completed!")
                else:
                    print("❌ Model training failed!")

            elif choice == '4':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None
                gas_sensor.continuous_monitoring_with_voltage_transformation(duration, 'datasheet')

            elif choice == '5':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None
                gas_sensor.continuous_monitoring_with_voltage_transformation(duration, 'extended')

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*80)
                print("🔍 DETAILED VOLTAGE TRANSFORMATION ANALYSIS")
                print("="*80)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Compensated Voltage: {data['voltage']:.3f}V")
                    print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                    
                    if data['drift_compensation_applied']:
                        drift_percent = abs(1 - data['drift_factor']) * 100
                        print(f"  🔧 Drift Compensation: {drift_percent:.1f}%")
                    
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    print(f"  PPM: {data['ppm']:.0f}")
                    
                    # Voltage analysis
                    if data['raw_voltage'] > 2.8:
                        print(f"  ⚠️  HIGH VOLTAGE - Consider potentiometer adjustment for faster response")
                    elif data['raw_voltage'] < 1.0:
                        print(f"  ⚠️  LOW VOLTAGE - Check sensor connections")
                    else:
                        print(f"  ✅ Voltage optimal for good response time")

                print(f"\n🤖 PREDICTION:")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")

            elif choice == '7':
                print("\nSet Environmental Conditions:")
                temp_input = input("Temperature (°C, current: {:.1f}): ".format(gas_sensor.current_temperature))
                humidity_input = input("Humidity (%RH, current: {:.1f}): ".format(gas_sensor.current_humidity))

                temp = float(temp_input) if temp_input else None
                humidity = float(humidity_input) if humidity_input else None

                gas_sensor.set_environmental_conditions(temp, humidity)

            elif choice == '8':
                current_mode = 'Extended' if gas_sensor.sensor_config['TGS2600']['use_extended_mode'] else 'Datasheet'
                print(f"\nCurrent mode: {current_mode}")
                print("1. Datasheet mode (accurate detection)")
                print("2. Extended mode (full range)")

                mode_choice = input("Select mode (1-2): ").strip()
                if mode_choice == '1':
                    gas_sensor.set_sensor_mode('datasheet')
                elif mode_choice == '2':
                    gas_sensor.set_sensor_mode('extended')

            elif choice == '9':
                print("\n" + "="*80)
                print("🔍 COMPREHENSIVE VOLTAGE & SENSOR DIAGNOSTICS")
                print("="*80)

                readings = gas_sensor.read_sensors()
                
                for sensor_name, data in readings.items():
                    config = gas_sensor.sensor_config[sensor_name]
                    
                    print(f"\n{sensor_name}:")
                    print(f"  Current Voltage: {data['raw_voltage']:.3f}V")
                    print(f"  Baseline Voltage: {config.get('baseline_voltage', 'Not set')}")
                    print(f"  Response Time Status: ", end="")
                    
                    if data['raw_voltage'] > 2.8:
                        print("⚠️  SLOW (High voltage - adjust potentiometer DOWN)")
                    elif data['raw_voltage'] > 2.0:
                        print("🔶 MODERATE (Consider slight adjustment)")
                    else:
                        print("✅ FAST (Optimal voltage range)")
                        
                    print(f"  PPM Reading: {data['ppm']:.0f}")
                    print(f"  Target Gases: {', '.join(config['target_gases'])}")

            elif choice == '11':
                print("\n🔧 HARDWARE ADJUSTMENT DETECTION")
                adjustment_detected, voltage_changes = gas_sensor.voltage_transformer.detect_hardware_adjustment(gas_sensor)
                
                if adjustment_detected:
                    print("✅ Hardware adjustment detected - ready for voltage transformation!")
                else:
                    print("ℹ️  No recent hardware adjustment detected")

            elif choice == '12':
                print("\n🔄 VOLTAGE TRANSFORMATION WORKFLOW")
                gas_sensor.voltage_transformer.apply_hardware_adjustment_workflow(gas_sensor)

            elif choice == '13':
                print("\n🚀 QUICK MODEL ADAPTATION")
                gas_type = input("Current gas type (or 'normal' for clean air): ").strip().lower()
                samples = int(input("Number of adaptation samples (default 30): ") or 30)
                
                success = gas_sensor.adaptive_manager.quick_model_adaptation(gas_sensor, gas_type, samples)
                
                if success:
                    print("✅ Quick adaptation completed successfully!")
                else:
                    print("❌ Adaptation failed")

            elif choice == '14':
                print("\n🧠 SMART DRIFT DETECTION & COMPENSATION")
                gas_sensor.drift_manager.intelligent_drift_compensation(gas_sensor, gas_sensor.adaptive_manager)

            elif choice == '15':
                print("\n📊 COMPREHENSIVE SYSTEM HEALTH & VOLTAGE ANALYSIS")
                
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)
                
                print(f"Model Confidence: {confidence:.3f}")
                print(f"Predicted Gas: {predicted_gas}")
                
                print(f"\n🔧 VOLTAGE ANALYSIS:")
                total_improvement_potential = 0
                
                for sensor_name, data in readings.items():
                    voltage = data['raw_voltage']
                    print(f"{sensor_name}: {voltage:.3f}V", end="")
                    
                    if voltage > 2.8:
                        improvement = gas_sensor.voltage_transformer.estimate_response_time_improvement(
                            voltage, 1.6, sensor_name
                        )
                        total_improvement_potential += improvement
                        print(f" → Potential {improvement:.1f}x response improvement with adjustment")
                    elif voltage > 2.0:
                        print(" → Minor improvement possible")
                    else:
                        print(" → Already optimized")
                
                if total_improvement_potential > 3:
                    print(f"\n🎯 RECOMMENDATION: Potentiometer adjustment recommended!")
                    print(f"   Potential response time improvement: {total_improvement_potential/3:.1f}x average")

            elif choice == '16':
                print("\n🔄 RESET ALL LEARNING DATA")
                confirm = input("Type 'RESET' to confirm complete reset: ")
                if confirm == 'RESET':
                    # Reset all managers
                    gas_sensor.drift_manager.drift_compensation_factors = {}
                    gas_sensor.drift_manager.baseline_history = {}
                    gas_sensor.adaptive_manager.adaptation_history = []
                    gas_sensor.voltage_transformer.transformation_matrices = {}
                    gas_sensor.voltage_transformer.hardware_adjustments = []
                    
                    # Save reset state
                    gas_sensor.drift_manager.save_drift_data()
                    gas_sensor.adaptive_manager.save_adaptation_data()
                    gas_sensor.voltage_transformer.save_transformation_data()
                    
                    print("✅ All learning data reset!")

            elif choice == '17':
                print("\n" + "="*80)
                print("🌟 POTENTIOMETER ADJUSTMENT HELPER")
                print("Guidance untuk optimize sensor response time")
                print("="*80)
                
                readings = gas_sensor.read_sensors()
                
                print("📊 CURRENT VOLTAGE STATUS:")
                adjustments_needed = []
                
                for sensor_name, data in readings.items():
                    voltage = data['raw_voltage']
                    print(f"\n{sensor_name}: {voltage:.3f}V")
                    
                    if voltage > 3.0:
                        print("  🔴 CRITICAL: Very slow response - ADJUST POTENTIOMETER DOWN significantly")
                        adjustments_needed.append(f"{sensor_name}: Turn potentiometer COUNTER-CLOCKWISE ~2-3 turns")
                    elif voltage > 2.5:
                        print("  🟡 SLOW: Moderate response - adjust potentiometer DOWN")
                        adjustments_needed.append(f"{sensor_name}: Turn potentiometer COUNTER-CLOCKWISE ~1-2 turns")
                    elif voltage > 2.0:
                        print("  🟠 MODERATE: Minor adjustment beneficial")
                        adjustments_needed.append(f"{sensor_name}: Turn potentiometer COUNTER-CLOCKWISE ~0.5-1 turn")
                    elif voltage < 1.0:
                        print("  🔵 LOW: May need adjustment UP")
                        adjustments_needed.append(f"{sensor_name}: Turn potentiometer CLOCKWISE ~0.5-1 turn")
                    else:
                        print("  ✅ OPTIMAL: Good response time expected")
                
                if adjustments_needed:
                    print(f"\n🔧 ADJUSTMENT RECOMMENDATIONS:")
                    for i, adjustment in enumerate(adjustments_needed, 1):
                        print(f"{i}. {adjustment}")
                    
                    print(f"\n📋 STEP-BY-STEP PROCEDURE:")
                    print("1. Power OFF the sensor system")
                    print("2. Locate potentiometers for each sensor")
                    print("3. Make SMALL adjustments (1/4 turn at a time)")
                    print("4. Power ON and test voltage with Menu 6")
                    print("5. Target voltage: 1.5V - 2.0V for optimal response")
                    print("6. After adjustment: Use Menu 12 (Voltage Transformation)")
                    print("   to preserve existing training data!")
                    
                    print(f"\n✅ BENEFITS after adjustment:")
                    print("- Faster gas detection response (2-5x improvement)")
                    print("- Better sensitivity to low concentrations")
                    print("- More stable readings")
                    print("- Training data preserved via voltage transformation")
                else:
                    print("\n✅ ALL SENSORS ALREADY OPTIMIZED!")
                    print("No potentiometer adjustments needed.")

            elif choice == '18':
                print("\n📈 RESPONSE TIME IMPROVEMENT ESTIMATION")
                
                readings = gas_sensor.read_sensors()
                target_voltage = float(input("Target voltage after adjustment (default 1.6V): ") or 1.6)
                
                print(f"\n📊 ESTIMATED IMPROVEMENTS with {target_voltage}V target:")
                
                for sensor_name, data in readings.items():
                    current_voltage = data['raw_voltage']
                    improvement = gas_sensor.voltage_transformer.estimate_response_time_improvement(
                        current_voltage, target_voltage, sensor_name
                    )
                    
                    print(f"{sensor_name}:")
                    print(f"  Current: {current_voltage:.3f}V")
                    print(f"  Target:  {target_voltage:.3f}V")
                    print(f"  Response improvement: {improvement:.1f}x faster")
                    
                    if improvement > 2:
                        print(f"  Status: 🚀 SIGNIFICANT IMPROVEMENT EXPECTED")
                    elif improvement > 1.5:
                        print(f"  Status: ✅ GOOD IMPROVEMENT EXPECTED")
                    else:
                        print(f"  Status: 📊 MINOR IMPROVEMENT")

            elif choice == '19':
                print("\n🔬 VOLTAGE-TO-PPM SENSITIVITY ANALYSIS")
                
                sensor_name = input("Enter sensor name (TGS2600/TGS2602/TGS2610): ").upper()
                if sensor_name not in gas_sensor.sensor_config:
                    print("Invalid sensor name!")
                    continue
                
                print(f"\n📊 SENSITIVITY ANALYSIS for {sensor_name}:")
                
                voltages = [1.5, 2.0, 2.5, 3.0, 3.5]
                print("Voltage → PPM Response (estimated for alcohol):")
                
                for voltage in voltages:
                    resistance = gas_sensor.voltage_to_resistance(voltage)
                    config = gas_sensor.sensor_config[sensor_name]
                    
                    if config.get('R0'):
                        rs_r0_ratio = resistance / config['R0']
                        ppm = gas_sensor.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, 'alcohol')
                        
                        print(f"{voltage:.1f}V → {ppm:.1f} ppm")
                        
                        if voltage > 2.8:
                            print("      ⚠️  High voltage = Poor sensitivity")
                        elif voltage < 1.2:
                            print("      ⚠️  Low voltage = May be unstable")
                        else:
                            print("      ✅ Good sensitivity range")
                    else:
                        print(f"{voltage:.1f}V → Not calibrated")

            elif choice == '10':
                print("👋 Exiting enhanced voltage transformation system...")
                # Save all data before exit
                gas_sensor.drift_manager.save_drift_data()
                gas_sensor.adaptive_manager.save_adaptation_data()
                gas_sensor.voltage_transformer.save_transformation_data()
                break

            else:
                print("❌ Invalid option!")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")
            gas_sensor.logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()