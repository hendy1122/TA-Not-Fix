#!/usr/bin/env python3
"""
Optimized Gas Sensor Array System for USV Air Pollution Detection
3-Gas Classification: Normal, Alkohol, Pertalite, Biosolar
Improved data collection protocol and model stability
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
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

class OptimizedGasSensorArray:
    def __init__(self):
        """Initialize optimized 3-gas sensor array system"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gas_sensor_optimized.log'),
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
        
        # OPTIMIZED: 3-gas configuration with unique characteristics
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['alcohol', 'hydrogen', 'carbon_monoxide'],
                'load_resistance': 10000,
                'R0': None,
                'baseline_voltage': None,
                'gas_sensitivity': {
                    'normal': 1.0,
                    'alkohol': 2.5,    # Strong response to alcohol
                    'pertalite': 1.8,   # Moderate response to gasoline
                    'biosolar': 1.2    # Weak response to diesel
                }
            },
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['alcohol', 'toluene', 'ammonia', 'voc'],
                'load_resistance': 10000,
                'R0': None,
                'baseline_voltage': None,
                'gas_sensitivity': {
                    'normal': 1.0,
                    'alkohol': 3.0,    # Very strong response to alcohol
                    'pertalite': 2.2,   # Strong response to gasoline VOCs
                    'biosolar': 1.4    # Moderate response to diesel
                }
            },
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'hydrocarbons'],
                'load_resistance': 10000,
                'R0': None,
                'baseline_voltage': None,
                'gas_sensitivity': {
                    'normal': 1.0,
                    'alkohol': 1.5,    # Weak response to alcohol
                    'pertalite': 2.8,   # Very strong response to gasoline
                    'biosolar': 2.0    # Strong response to diesel
                }
            }
        }
        
        # OPTIMIZED: Target gas types (3 gases only)
        self.target_gases = ['normal', 'alkohol', 'pertalite', 'biosolar']
        
        # Environmental compensation parameters
        self.current_temperature = 20.0
        self.current_humidity = 65.0
        
        # Data collection optimization
        self.stabilization_time = 30  # seconds to wait between gases
        self.collection_duration = 120  # 2 minutes per gas (reduced from 5)
        self.sampling_rate = 2  # 2 Hz (increased from 1 Hz)
        
        # Data storage
        self.data_queue = queue.Queue()
        self.is_collecting = False
        
        # Machine Learning components
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        
        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)
        
        self.logger.info("Optimized 3-Gas Sensor Array System initialized")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """Convert ADC voltage to sensor resistance"""
        if voltage <= 0.001:
            return float('inf')
        
        circuit_voltage = 5.0
        if voltage >= circuit_voltage:
            return 0.1
        
        sensor_resistance = load_resistance * (circuit_voltage - voltage) / voltage
        return max(1, sensor_resistance)

    def calculate_enhanced_ppm(self, sensor_name, resistance, gas_type='normal'):
        """
        OPTIMIZED: Enhanced PPM calculation with gas-specific sensitivity
        """
        config = self.sensor_config[sensor_name]
        R0 = config.get('R0')
        
        if R0 is None or R0 == 0:
            return self.simplified_ppm_calculation(sensor_name, resistance, gas_type)
        
        rs_r0_ratio = resistance / R0
        
        if rs_r0_ratio >= 1.0:
            return 0  # No gas detected
        
        # Base PPM calculation
        base_ppm = 100 * ((0.6 / rs_r0_ratio) ** 2.0)
        
        # Apply gas-specific sensitivity multiplier
        sensitivity = config['gas_sensitivity'].get(gas_type, 1.0)
        enhanced_ppm = base_ppm * sensitivity
        
        # Add sensor-specific characteristics
        if sensor_name == 'TGS2600':
            # Alcohol sensor - strong response to alcohol, moderate to gasoline
            if gas_type == 'alkohol':
                enhanced_ppm = enhanced_ppm * 1.2  # Boost alcohol detection
            elif gas_type == 'biosolar':
                enhanced_ppm = enhanced_ppm * 0.7  # Reduce diesel detection
                
        elif sensor_name == 'TGS2602':
            # VOC sensor - strong response to all organics
            if gas_type == 'pertalite':
                enhanced_ppm = enhanced_ppm * 1.1  # Boost gasoline VOC detection
                
        elif sensor_name == 'TGS2610':
            # Hydrocarbon sensor - strong response to gasoline and diesel
            if gas_type == 'alkohol':
                enhanced_ppm = enhanced_ppm * 0.6  # Reduce alcohol detection
        
        return max(0, enhanced_ppm)

    def simplified_ppm_calculation(self, sensor_name, resistance, gas_type='normal'):
        """Simplified PPM calculation when R0 is not available"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 0.4)
        baseline_resistance = self.voltage_to_resistance(baseline_voltage)
        
        if resistance >= baseline_resistance:
            return 0
        
        ratio = baseline_resistance / resistance
        base_ppm = 200 * (ratio - 1) * 0.5
        
        # Apply gas sensitivity
        sensitivity = config['gas_sensitivity'].get(gas_type, 1.0)
        return max(0, base_ppm * sensitivity)

    def calibrate_sensors(self, duration=300):
        """Enhanced sensor calibration"""
        self.logger.info(f"Starting sensor calibration for {duration} seconds...")
        self.logger.info("🔴 IMPORTANT: Turn OFF suction fan during calibration!")
        self.logger.info("🌬️  Ensure sensors are in CLEAN AIR environment!")
        
        input("Press Enter when sensors are in clean air (fan OFF)...")
        
        readings = {sensor: {'voltages': [], 'resistances': []} 
                   for sensor in self.sensor_config.keys()}
        
        start_time = time.time()
        sample_count = 0
        
        print("Calibrating", end="")
        while time.time() - start_time < duration:
            for sensor_name, config in self.sensor_config.items():
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])
                
                readings[sensor_name]['voltages'].append(voltage)
                readings[sensor_name]['resistances'].append(resistance)
            
            sample_count += 1
            
            # Progress indicator
            if sample_count % 30 == 0:
                print(".", end="", flush=True)
            
            time.sleep(2)
        
        print()  # New line
        
        # Calculate calibration parameters
        for sensor_name in self.sensor_config.keys():
            voltages = readings[sensor_name]['voltages']
            resistances = readings[sensor_name]['resistances']
            
            voltage_mean = np.mean(voltages)
            resistance_mean = np.mean(resistances)
            voltage_std = np.std(voltages)
            resistance_std = np.std(resistances)
            
            self.sensor_config[sensor_name]['R0'] = resistance_mean
            self.sensor_config[sensor_name]['baseline_voltage'] = voltage_mean
            
            stability = (voltage_std / voltage_mean) * 100
            
            self.logger.info(f"{sensor_name}:")
            self.logger.info(f"  R0: {resistance_mean:.1f}Ω ± {resistance_std:.1f}Ω")
            self.logger.info(f"  Baseline: {voltage_mean:.3f}V ± {voltage_std:.3f}V")
            self.logger.info(f"  Stability: {stability:.2f}%")
        
        # Save calibration
        calib_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'sensors': {
                name: {
                    'R0': config['R0'],
                    'baseline_voltage': config['baseline_voltage']
                }
                for name, config in self.sensor_config.items()
            }
        }
        
        with open('sensor_calibration.json', 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        self.logger.info("✅ Calibration completed and saved")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)
            
            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']
            
            self.logger.info("✅ Calibration data loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.warning("⚠️  No calibration file found")
            return False

    def read_sensors(self, gas_type='normal'):
        """Read all sensors with optimized processing"""
        readings = {}
        
        for sensor_name, config in self.sensor_config.items():
            try:
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])
                
                R0 = config.get('R0')
                rs_r0_ratio = resistance / R0 if R0 else None
                
                # Calculate enhanced PPM
                ppm = self.calculate_enhanced_ppm(sensor_name, resistance, gas_type)
                
                readings[sensor_name] = {
                    'voltage': voltage,
                    'resistance': resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'R0': R0
                }
                
            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'resistance': 0, 'rs_r0_ratio': None, 
                    'ppm': 0, 'R0': None
                }
        
        return readings

    def collect_training_data_optimized(self):
        """
        OPTIMIZED: Sequential data collection for all gases with proper protocol
        """
        self.logger.info("🚀 Starting OPTIMIZED training data collection")
        self.logger.info("📋 Protocol: Normal → Alkohol → Pertalite → Biosolar")
        self.logger.info(f"⏱️  Duration per gas: {self.collection_duration}s")
        self.logger.info(f"📊 Sampling rate: {self.sampling_rate} Hz")
        
        all_training_data = []
        
        for i, gas_type in enumerate(self.target_gases):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Step {i+1}/4: Collecting {gas_type.upper()} data")
            self.logger.info(f"{'='*50}")
            
            if gas_type == 'normal':
                self.logger.info("🌬️  CLEAN AIR - Turn OFF suction fan")
                input("Ensure clean air environment. Press Enter to start...")
            else:
                self.logger.info(f"💨 GAS: {gas_type.upper()} - Turn OFF suction fan")
                self.logger.info("🔧 Prepare to spray gas directly to sensor chamber")
                input(f"Ready to spray {gas_type}? Press Enter to start...")
            
            # Data collection for this gas
            gas_data = []
            start_time = time.time()
            sample_interval = 1.0 / self.sampling_rate
            
            print(f"Collecting {gas_type}:", end="")
            
            while time.time() - start_time < self.collection_duration:
                timestamp = datetime.now()
                readings = self.read_sensors(gas_type)
                
                # Create data row
                data_row = {
                    'timestamp': timestamp,
                    'gas_type': gas_type,
                    'temperature': self.current_temperature,
                    'humidity': self.current_humidity,
                    
                    'TGS2600_voltage': readings['TGS2600']['voltage'],
                    'TGS2600_resistance': readings['TGS2600']['resistance'],
                    'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                    'TGS2600_ppm': readings['TGS2600']['ppm'],
                    
                    'TGS2602_voltage': readings['TGS2602']['voltage'],
                    'TGS2602_resistance': readings['TGS2602']['resistance'],
                    'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                    'TGS2602_ppm': readings['TGS2602']['ppm'],
                    
                    'TGS2610_voltage': readings['TGS2610']['voltage'],
                    'TGS2610_resistance': readings['TGS2610']['resistance'],
                    'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                    'TGS2610_ppm': readings['TGS2610']['ppm']
                }
                
                gas_data.append(data_row)
                all_training_data.append(data_row)
                
                # Progress indicator
                elapsed = time.time() - start_time
                remaining = self.collection_duration - elapsed
                
                if len(gas_data) % 10 == 0:
                    print(".", end="", flush=True)
                
                time.sleep(sample_interval)
            
            print(f" ✅ {len(gas_data)} samples")
            
            # Quick data analysis
            df_gas = pd.DataFrame(gas_data)
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm_col = f'{sensor}_ppm'
                ppm_mean = df_gas[ppm_col].mean()
                ppm_max = df_gas[ppm_col].max()
                print(f"  {sensor}: avg={ppm_mean:.0f}ppm, max={ppm_max:.0f}ppm")
            
            # Stabilization period between gases (except for last gas)
            if i < len(self.target_gases) - 1:
                self.logger.info(f"⏳ Stabilization period: {self.stabilization_time}s")
                self.logger.info("💨 Turn ON suction fan for cleaning")
                
                for remaining in range(self.stabilization_time, 0, -1):
                    print(f"\rCleaning chamber: {remaining}s remaining", end="", flush=True)
                    time.sleep(1)
                print()
                
                self.logger.info("🛑 Turn OFF suction fan before next gas")
                input("Press Enter when fan is OFF and ready for next gas...")
        
        # Save complete training dataset
        df_all = pd.DataFrame(all_training_data)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/training_complete_{timestamp_str}.csv"
        df_all.to_csv(filename, index=False)
        
        self.logger.info(f"\n🎉 Training data collection completed!")
        self.logger.info(f"📁 Saved to: {filename}")
        self.logger.info(f"📊 Total samples: {len(all_training_data)}")
        
        # Final data quality analysis
        self.analyze_complete_dataset(df_all)
        
        return all_training_data

    def analyze_complete_dataset(self, df):
        """Analyze complete training dataset quality"""
        self.logger.info("\n📈 DATASET QUALITY ANALYSIS")
        self.logger.info("="*40)
        
        # Samples per gas type
        gas_counts = df['gas_type'].value_counts()
        self.logger.info("Samples per gas type:")
        for gas, count in gas_counts.items():
            self.logger.info(f"  {gas}: {count} samples")
        
        # Sensor response analysis
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            self.logger.info(f"\n{sensor} Response Analysis:")
            ppm_col = f'{sensor}_ppm'
            
            for gas in self.target_gases:
                gas_data = df[df['gas_type'] == gas][ppm_col]
                if len(gas_data) > 0:
                    mean_val = gas_data.mean()
                    max_val = gas_data.max()
                    std_val = gas_data.std()
                    self.logger.info(f"  {gas}: {mean_val:.0f}±{std_val:.0f}ppm (max: {max_val:.0f})")
        
        # Data balance check
        min_samples = gas_counts.min()
        max_samples = gas_counts.max()
        balance_ratio = min_samples / max_samples
        
        if balance_ratio > 0.8:
            self.logger.info("✅ Dataset is well balanced")
        elif balance_ratio > 0.6:
            self.logger.info("⚠️  Dataset has minor imbalance")
        else:
            self.logger.info("❌ Dataset is significantly imbalanced")

    def train_optimized_model(self):
        """Train optimized machine learning model"""
        self.logger.info("🤖 Training optimized ML model...")
        
        # Load training data
        training_data = self.load_all_training_data()
        if training_data is None:
            self.logger.error("❌ No training data found!")
            return False
        
        # Feature selection (optimized)
        feature_columns = [
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
            'temperature', 'humidity'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in training_data.columns]
        
        X = training_data[available_columns].values
        y = training_data['gas_type'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Check data distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.logger.info(f"Classes: {list(unique_classes)}")
        for cls, count in zip(unique_classes, class_counts):
            self.logger.info(f"  {cls}: {count} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimized Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"🎯 Model accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\n🔝 Top 5 Important Features:")
        for _, row in feature_importance.head().iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"\n📊 Confusion Matrix:")
        self.logger.info(f"Classes: {unique_classes}")
        self.logger.info(f"\n{cm}")
        
        # Save model
        joblib.dump(self.model, 'models/gas_classifier_optimized.pkl')
        joblib.dump(self.scaler, 'models/scaler_optimized.pkl')
        
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'feature_columns': available_columns,
            'classes': list(unique_classes),
            'model_type': 'optimized_3gas'
        }
        
        with open('models/model_metadata_optimized.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        self.is_model_trained = True
        self.logger.info("✅ Model training completed!")
        
        return True

    def load_all_training_data(self):
        """Load all training data files"""
        data_files = list(Path("data").glob("training_*.csv"))
        if not data_files:
            return None
        
        all_data = []
        for file in data_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                self.logger.info(f"Loaded {len(df)} samples from {file.name}")
            except Exception as e:
                self.logger.error(f"Error loading {file.name}: {e}")
        
        if not all_data:
            return None
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Total samples: {len(combined_data)}")
        
        return combined_data

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier_optimized.pkl')
            self.scaler = joblib.load('models/scaler_optimized.pkl')
            self.is_model_trained = True
            self.logger.info("✅ Model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("❌ No trained model found")
            return False

    def predict_gas(self, readings):
        """Predict gas type with confidence"""
        if not self.is_model_trained:
            return "Unknown", 0.0
        
        try:
            # Load metadata for feature order
            with open('models/model_metadata_optimized.json', 'r') as f:
                metadata = json.load(f)
            feature_columns = metadata['feature_columns']
        except:
            feature_columns = [
                'TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage', 
                'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm'
            ]
        
        # Build feature vector
        feature_vector = []
        for feature in feature_columns:
            if feature == 'temperature':
                feature_vector.append(self.current_temperature)
            elif feature == 'humidity':
                feature_vector.append(self.current_humidity)
            else:
                # Parse sensor and measurement
                parts = feature.split('_')
                sensor = parts[0]
                measurement = '_'.join(parts[1:])
                
                if sensor in readings and measurement in readings[sensor]:
                    value = readings[sensor][measurement]
                    feature_vector.append(value if value is not None else 0.0)
                else:
                    feature_vector.append(0.0)
        
        features = np.array([feature_vector])
        features = np.nan_to_num(features, nan=0.0)
        
        # Predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        return prediction, confidence

    def continuous_monitoring(self, duration=None):
        """Optimized continuous monitoring"""
        self.logger.info("🔍 Starting optimized monitoring...")
        self.logger.info("💨 Turn OFF suction fan during monitoring")
        
        input("Press Enter when fan is OFF and ready to monitor...")
        
        self.is_collecting = True
        
        fieldnames = [
            'timestamp', 'temperature', 'humidity',
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
            'predicted_gas', 'confidence'
        ]
        
        monitoring_file = f"data/monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
                    
                    data_row = {
                        'timestamp': timestamp,
                        'temperature': self.current_temperature,
                        'humidity': self.current_humidity,
                        
                        'TGS2600_voltage': readings['TGS2600']['voltage'],
                        'TGS2600_resistance': readings['TGS2600']['resistance'],
                        'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                        'TGS2600_ppm': readings['TGS2600']['ppm'],
                        
                        'TGS2602_voltage': readings['TGS2602']['voltage'],
                        'TGS2602_resistance': readings['TGS2602']['resistance'],
                        'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                        'TGS2602_ppm': readings['TGS2602']['ppm'],
                        
                        'TGS2610_voltage': readings['TGS2610']['voltage'],
                        'TGS2610_resistance': readings['TGS2610']['resistance'],
                        'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                        'TGS2610_ppm': readings['TGS2610']['ppm'],
                        
                        'predicted_gas': predicted_gas,
                        'confidence': confidence
                    }
                    
                    writer.writerow(data_row)
                    sample_count += 1
                    
                    # Enhanced display
                    print(f"\r{timestamp.strftime('%H:%M:%S')} | "
                          f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                          f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                          f"2610: {readings['TGS2610']['ppm']:.0f}ppm | "
                          f"🎯 {predicted_gas} ({confidence:.2f})", end="")
                    
                    # Gas detection alert
                    max_ppm = max(readings['TGS2600']['ppm'], readings['TGS2602']['ppm'], readings['TGS2610']['ppm'])
                    if max_ppm > 30 and confidence > 0.7:
                        print(f"\n🚨 GAS DETECTED: {predicted_gas.upper()} (confidence: {confidence:.2f})")
                    
                    # Check duration
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    time.sleep(0.5)  # 2 Hz monitoring
                    
            except KeyboardInterrupt:
                print("\n⏹️  Monitoring stopped by user")
            
        self.is_collecting = False
        self.logger.info(f"📁 Monitoring data saved to {monitoring_file}")
        self.logger.info(f"📊 Total samples: {sample_count}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_collecting = False

    def test_single_reading(self):
        """Test single sensor reading with detailed analysis"""
        self.logger.info("🧪 Single sensor reading test")
        self.logger.info("💨 Ensure suction fan is OFF")
        
        input("Press Enter to take reading...")
        
        readings = self.read_sensors()
        predicted_gas, confidence = self.predict_gas(readings)
        
        print("\n" + "="*60)
        print("📊 DETAILED SENSOR ANALYSIS")
        print("="*60)
        
        for sensor, data in readings.items():
            print(f"\n🔧 {sensor}:")
            print(f"  Voltage: {data['voltage']:.3f}V")
            print(f"  Resistance: {data['resistance']:.1f}Ω")
            if data['rs_r0_ratio']:
                print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
            else:
                print(f"  Rs/R0 Ratio: Not calibrated")
            print(f"  PPM: {data['ppm']:.0f}")
            print(f"  R0 (Baseline): {data['R0']:.1f}Ω" if data['R0'] else "  R0: Not calibrated")
        
        print(f"\n🌡️  Environmental:")
        print(f"  Temperature: {self.current_temperature}°C")
        print(f"  Humidity: {self.current_humidity}%RH")
        
        print(f"\n🎯 PREDICTION:")
        print(f"  Gas Type: {predicted_gas.upper()}")
        print(f"  Confidence: {confidence:.3f}")
        
        if confidence < 0.5:
            print("  ⚠️  Low confidence - uncertain detection")
        elif confidence < 0.7:
            print("  ⚠️  Medium confidence")
        else:
            print("  ✅ High confidence detection")

    def sensor_diagnostics(self):
        """Display sensor diagnostic information"""
        print("\n" + "="*60)
        print("🔧 SENSOR DIAGNOSTICS")
        print("="*60)
        
        for sensor_name, config in self.sensor_config.items():
            print(f"\n🔍 {sensor_name}:")
            print(f"  Target Gases: {', '.join(config['target_gases'])}")
            print(f"  Load Resistance: {config['load_resistance']}Ω")
            
            if config['R0']:
                print(f"  ✅ Calibrated R0: {config['R0']:.1f}Ω")
                print(f"  ✅ Baseline Voltage: {config['baseline_voltage']:.3f}V")
            else:
                print(f"  ❌ Not calibrated")
            
            print(f"  Gas Sensitivity Factors:")
            for gas, factor in config['gas_sensitivity'].items():
                print(f"    {gas}: {factor:.1f}x")

def main():
    """Optimized main function with improved workflow"""
    gas_sensor = OptimizedGasSensorArray()
    
    # Load existing calibration and model
    gas_sensor.load_calibration()
    gas_sensor.load_model()
    
    while True:
        print("\n" + "="*60)
        print("🚀 OPTIMIZED Gas Sensor Array System")
        print("3-Gas Classification: Normal, Alkohol, Pertalite, Biosolar")
        print("="*60)
        print("1. 🔧 Calibrate sensors (Required first)")
        print("2. 📊 Collect ALL training data (Optimized protocol)")
        print("3. 🤖 Train machine learning model")
        print("4. 🔍 Start continuous monitoring")
        print("5. 🧪 Test single reading (Detailed analysis)")
        print("6. 🌡️  Set environmental conditions")
        print("7. 🔧 View sensor diagnostics")
        print("8. ❓ Help & Instructions")
        print("9. ⏹️  Exit")
        print("-"*60)
        
        try:
            choice = input("Select option (1-9): ").strip()
            
            if choice == '1':
                print("\n🔧 SENSOR CALIBRATION")
                print("📋 Protocol:")
                print("  1. Turn OFF suction fan")
                print("  2. Ensure clean air environment")
                print("  3. Let sensors warm up for 10+ minutes")
                
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                gas_sensor.calibrate_sensors(duration)
                
            elif choice == '2':
                print("\n📊 OPTIMIZED TRAINING DATA COLLECTION")
                print("📋 Protocol:")
                print("  • Sequential collection: Normal → Alkohol → Pertalite → Biosolar")
                print("  • 2 minutes per gas type")
                print("  • 30s cleaning between gases")
                print("  • Fan OFF during collection, ON during cleaning")
                
                confirm = input("Start optimized data collection? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.collect_training_data_optimized()
                
            elif choice == '3':
                print("\n🤖 TRAINING MACHINE LEARNING MODEL")
                if gas_sensor.train_optimized_model():
                    print("✅ Model training completed successfully!")
                else:
                    print("❌ Model training failed!")
                
            elif choice == '4':
                print("\n🔍 CONTINUOUS MONITORING")
                print("💨 Important: Turn OFF suction fan during monitoring")
                
                duration_input = input("Duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None
                
                gas_sensor.continuous_monitoring(duration)
                
            elif choice == '5':
                gas_sensor.test_single_reading()
                
            elif choice == '6':
                print("\n🌡️  ENVIRONMENTAL CONDITIONS")
                temp_input = input(f"Temperature (°C, current: {gas_sensor.current_temperature}): ")
                humidity_input = input(f"Humidity (%RH, current: {gas_sensor.current_humidity}): ")
                
                if temp_input:
                    gas_sensor.current_temperature = float(temp_input)
                    print(f"✅ Temperature set to {gas_sensor.current_temperature}°C")
                
                if humidity_input:
                    gas_sensor.current_humidity = float(humidity_input)
                    print(f"✅ Humidity set to {gas_sensor.current_humidity}%RH")
                
            elif choice == '7':
                gas_sensor.sensor_diagnostics()
                
            elif choice == '8':
                print("\n" + "="*60)
                print("❓ HELP & INSTRUCTIONS")
                print("="*60)
                print("\n📋 WORKFLOW:")
                print("1. First time setup:")
                print("   • Calibrate sensors (option 1)")
                print("   • Collect training data (option 2)")
                print("   • Train model (option 3)")
                print("\n2. Daily operation:")
                print("   • Start monitoring (option 4)")
                print("   • Test readings (option 5)")
                print("\n🔧 SUCTION FAN USAGE:")
                print("• OFF during calibration, data collection, and monitoring")
                print("• ON only during cleaning between gases")
                print("\n⏱️  TIMING:")
                print("• Calibration: 5 minutes")
                print("• Data collection: 2 minutes per gas")
                print("• Cleaning: 30 seconds between gases")
                print("\n🎯 TARGET GASES:")
                print("• Normal (clean air)")
                print("• Alkohol (isopropyl alcohol)")
                print("• Pertalite (gasoline)")
                print("• Biosolar (diesel)")
                
            elif choice == '9':
                print("👋 Exiting optimized gas sensor system...")
                break
                
            else:
                print("❌ Invalid option!")
                
        except KeyboardInterrupt:
            print("\n⏹️  Operation cancelled by user")
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()