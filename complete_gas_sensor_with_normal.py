#!/usr/bin/env python3
"""
Fixed Gas Sensor Array System - Hybrid Approach with NORMAL Detection
Mengatasi masalah PPM stuck di nilai maksimum
Kombinasi datasheet accuracy + unlimited range + normal condition detection
Updated: Tambah deteksi kondisi normal (clean air)
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

class HybridGasSensorArray:
    def __init__(self):
        """Initialize hybrid gas sensor array system"""
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
        
        # Hybrid sensor configuration
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['hydrogen', 'carbon_monoxide', 'alcohol'],
                'load_resistance': 10000,  # 10kΩ
                'R0': None,  # Will be set during calibration
                'baseline_voltage': None,
                
                # Datasheet parameters for LOW concentration (accurate range)
                'datasheet_range': (1, 30),  # ppm - accurate range from datasheet
                'datasheet_sensitivity': {
                    'hydrogen': {'rs_r0_min': 0.3, 'rs_r0_max': 0.6},
                    'alcohol': {'rs_r0_min': 0.2, 'rs_r0_max': 0.5}
                },
                
                # Extended parameters for HIGH concentration (training purpose)
                'extended_range': (1, 500),  # ppm - extended for training
                'extended_sensitivity': 2.5,  # Sensitivity factor for high concentrations
                
                # Hybrid mode settings
                'use_datasheet_mode': False,  # Start with extended mode for training
                'concentration_threshold': 50  # Switch threshold between modes
            },
            
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['toluene', 'ammonia', 'h2s', 'alcohol'],
                'load_resistance': 10000,
                'R0': None,
                'baseline_voltage': None,
                
                'datasheet_range': (1, 30),  # ppm ethanol equivalent
                'datasheet_sensitivity': {
                    'alcohol': {'rs_r0_min': 0.08, 'rs_r0_max': 0.5},
                    'toluene': {'rs_r0_min': 0.1, 'rs_r0_max': 0.4}
                },
                
                'extended_range': (1, 300),  # ppm - extended range
                'extended_sensitivity': 3.0,
                
                'use_datasheet_mode': False,
                'concentration_threshold': 40
            },
            
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'iso_butane'],
                'load_resistance': 10000,
                'R0': None,
                'baseline_voltage': None,
                
                'datasheet_range': (1, 25),  # % LEL
                'datasheet_sensitivity': {
                    'iso_butane': {'rs_r0_min': 0.45, 'rs_r0_max': 0.62},
                    'butane': {'rs_r0_min': 0.4, 'rs_r0_max': 0.6}
                },
                
                'extended_range': (1, 200),  # % LEL - extended range
                'extended_sensitivity': 2.0,
                
                'use_datasheet_mode': False,
                'concentration_threshold': 30
            }
        }
        
        # Environmental compensation parameters
        self.current_temperature = 20.0
        self.current_humidity = 65.0
        
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
        
        self.logger.info("Hybrid Gas Sensor Array System initialized")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """Convert ADC voltage to sensor resistance"""
        if voltage <= 0.001:  # Avoid division by zero
            return float('inf')
        
        circuit_voltage = 5.0
        # Handle edge cases where voltage might be higher than expected
        if voltage >= circuit_voltage:
            return 0.1  # Very low resistance
        
        sensor_resistance = load_resistance * (circuit_voltage - voltage) / voltage
        return max(1, sensor_resistance)  # Minimum 1Ω

    def hybrid_ppm_calculation(self, sensor_name, resistance, gas_type='auto'):
        """
        Hybrid PPM calculation:
        - Datasheet mode: Accurate for low concentrations
        - Extended mode: Unlimited range for high concentrations and training
        """
        config = self.sensor_config[sensor_name]
        R0 = config.get('R0')
        
        if R0 is None or R0 == 0:
            return self.fallback_ppm_calculation(sensor_name, resistance)
        
        rs_r0_ratio = resistance / R0
        
        # Choose calculation mode
        if config['use_datasheet_mode']:
            ppm = self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            ppm = self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        
        return max(0, ppm)

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Accurate PPM calculation based on datasheet (for low concentrations)"""
        
        if sensor_name == 'TGS2600':
            if gas_type in ['hydrogen', 'auto'] and rs_r0_ratio <= 0.6:
                # Hydrogen detection curve from datasheet
                if rs_r0_ratio < 0.1:
                    return 30  # Max datasheet range
                ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                return min(ppm, 30)  # Limit to datasheet range
            elif gas_type in ['alcohol', 'auto'] and rs_r0_ratio <= 0.5:
                # Alcohol detection curve
                if rs_r0_ratio < 0.05:
                    return 30
                ppm = 40 * ((0.4 / rs_r0_ratio) ** 2.0)
                return min(ppm, 30)
                
        elif sensor_name == 'TGS2602':
            if gas_type in ['alcohol', 'auto'] and rs_r0_ratio <= 0.5:
                # Ethanol detection curve
                if rs_r0_ratio < 0.02:
                    return 30  # Max datasheet range
                ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                return min(ppm, 30)
                
        elif sensor_name == 'TGS2610':
            if rs_r0_ratio <= 0.62:
                # LP gas detection curve (% LEL)
                if rs_r0_ratio < 0.1:
                    return 25  # Max datasheet range
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                return min(ppm, 25)
        
        return 0

    def extended_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """
        Extended PPM calculation for training data collection
        Removes upper limits to capture full dynamic range
        """
        config = self.sensor_config[sensor_name]
        sensitivity = config['extended_sensitivity']
        
        if rs_r0_ratio >= 1.0:
            return 0  # No gas detected
        
        # Base calculation without upper limits
        if sensor_name == 'TGS2600':
            # Extended range for alcohol, pertalite, pertamax
            if rs_r0_ratio < 0.05:  # Very high concentration
                base_ppm = 200 + (0.05 - rs_r0_ratio) * 1000
            elif rs_r0_ratio < 0.2:  # High concentration
                base_ppm = 100 + (0.2 - rs_r0_ratio) * 500
            else:  # Normal to low concentration
                base_ppm = 50 * ((0.6 / rs_r0_ratio) ** sensitivity)
                
        elif sensor_name == 'TGS2602':
            # Extended range for VOCs
            if rs_r0_ratio < 0.02:  # Very high concentration
                base_ppm = 150 + (0.02 - rs_r0_ratio) * 2000
            elif rs_r0_ratio < 0.1:  # High concentration
                base_ppm = 75 + (0.1 - rs_r0_ratio) * 800
            else:  # Normal to low concentration
                base_ppm = 40 * ((0.3 / rs_r0_ratio) ** sensitivity)
                
        elif sensor_name == 'TGS2610':
            # Extended range for LP gas compounds
            if rs_r0_ratio < 0.1:  # Very high concentration
                base_ppm = 100 + (0.1 - rs_r0_ratio) * 1500
            elif rs_r0_ratio < 0.3:  # High concentration
                base_ppm = 50 + (0.3 - rs_r0_ratio) * 400
            else:  # Normal to low concentration
                base_ppm = 35 * ((0.7 / rs_r0_ratio) ** sensitivity)
        
        else:
            base_ppm = 0
        
        # Apply gas-specific multipliers for differentiation
        gas_multipliers = {
            'alcohol': 1.0,
            'pertalite': 1.2,
            'pertamax': 1.4,
            'dexlite': 1.6,
            'biosolar': 1.8,
            'normal': 0.8  # Normal air should have lower readings
        }
        
        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def fallback_ppm_calculation(self, sensor_name, resistance):
        """Fallback calculation when R0 is not available"""
        # Use baseline voltage as reference
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 0.4)
        
        baseline_resistance = self.voltage_to_resistance(baseline_voltage)
        
        if resistance >= baseline_resistance:
            return 0
        
        # Simple ratio-based calculation with extended range
        ratio = baseline_resistance / resistance
        max_range = config['extended_range'][1]
        
        ppm = max_range * (ratio - 1) * 0.3
        return max(0, ppm)

    def calibrate_sensors(self, duration=300):
        """Enhanced calibration for R0 determination"""
        self.logger.info(f"Starting sensor calibration for {duration} seconds...")
        self.logger.info("Ensure sensors are in CLEAN AIR environment!")
        
        input("Press Enter when sensors are in clean air and warmed up...")
        
        readings = {sensor: {'voltages': [], 'resistances': []} 
                   for sensor in self.sensor_config.keys()}
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            for sensor_name, config in self.sensor_config.items():
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])
                
                readings[sensor_name]['voltages'].append(voltage)
                readings[sensor_name]['resistances'].append(resistance)
            
            sample_count += 1
            time.sleep(2)  # 0.5 Hz sampling
            
            remaining = int(duration - (time.time() - start_time))
            if remaining % 30 == 0 and remaining > 0:
                print(f"Calibration remaining: {remaining} seconds (Samples: {sample_count})")
        
        # Calculate calibration parameters
        calibration_results = {}
        
        for sensor_name in self.sensor_config.keys():
            voltages = readings[sensor_name]['voltages']
            resistances = readings[sensor_name]['resistances']
            
            voltage_mean = np.mean(voltages)
            voltage_std = np.std(voltages)
            resistance_mean = np.mean(resistances)
            resistance_std = np.std(resistances)
            
            # Set R0 and baseline
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
        
        # Save calibration data
        calib_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'temperature': self.current_temperature,
            'humidity': self.current_humidity,
            'sensors': calibration_results
        }
        
        with open('sensor_calibration.json', 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        self.logger.info("Calibration completed and saved")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)
            
            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']
            
            self.logger.info("Calibration data loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Please run calibration first.")
            return False

    def read_sensors(self):
        """Read all sensors with hybrid PPM calculation"""
        readings = {}
        
        for sensor_name, config in self.sensor_config.items():
            try:
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])
                
                # Calculate Rs/R0 ratio
                R0 = config.get('R0')
                rs_r0_ratio = resistance / R0 if R0 else None
                
                # Calculate PPM using hybrid method
                ppm = self.hybrid_ppm_calculation(sensor_name, resistance)
                
                readings[sensor_name] = {
                    'voltage': voltage,
                    'resistance': resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'R0': R0,
                    'mode': 'Extended' if not config['use_datasheet_mode'] else 'Datasheet',
                    'target_gases': config['target_gases']
                }
                
            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'resistance': 0, 'rs_r0_ratio': None,
                    'ppm': 0, 'R0': None, 'mode': 'Error', 'target_gases': []
                }
        
        return readings

    def set_sensor_mode(self, mode='extended'):
        """
        Set sensor calculation mode
        Args:
            mode: 'extended' for training (unlimited range) or 'datasheet' for accurate detection
        """
        use_datasheet = (mode == 'datasheet')
        
        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_datasheet_mode'] = use_datasheet
        
        mode_name = "Datasheet (Accurate)" if use_datasheet else "Extended (Training)"
        self.logger.info(f"Sensor mode set to: {mode_name}")

    def collect_training_data(self, gas_type, duration=60, samples_per_second=1):
        """Collect training data with extended range - UPDATED dengan normal class"""
        # Ensure we're in extended mode for training
        self.set_sensor_mode('extended')
        
        self.logger.info(f"Collecting training data for {gas_type} in EXTENDED mode")
        self.logger.info(f"Duration: {duration}s, Sampling rate: {samples_per_second} Hz")
        
        # UPDATE: Tambahkan 'normal' ke valid gases
        valid_gases = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']
        if gas_type not in valid_gases:
            self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
            return None
        
        # Special instructions untuk normal data collection
        if gas_type == 'normal':
            input(f"Ensure sensors are in CLEAN AIR (no gas). Press Enter to start...")
            self.logger.info("Collecting NORMAL/CLEAN AIR data...")
        else:
            input(f"Prepare to spray {gas_type}. Press Enter to start...")
        
        training_data = []
        start_time = time.time()
        sample_interval = 1.0 / samples_per_second
        
        while time.time() - start_time < duration:
            timestamp = datetime.now()
            readings = self.read_sensors()
            
            data_row = {
                'timestamp': timestamp,
                'gas_type': gas_type,  # Ini akan menjadi 'normal' untuk clean air
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
            
            training_data.append(data_row)
            
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            # Special display untuk normal data
            status = "CLEAN AIR" if gas_type == 'normal' else gas_type.upper()
            print(f"\rTime: {remaining:.1f}s | Status: {status} | "
                  f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                  f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                  f"2610: {readings['TGS2610']['ppm']:.0f}ppm | "
                  f"Mode: Extended", end="")
            
            time.sleep(sample_interval)
        
        print()
        
        # Save training data
        filename = f"data/training_{gas_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False)
        
        # Analysis
        self.analyze_training_data_quality(df, gas_type)
        
        self.logger.info(f"Training data saved to {filename}")
        self.logger.info(f"Collected {len(training_data)} samples for {gas_type}")
        
        return training_data

    def analyze_training_data_quality(self, df, gas_type):
        """Analyze training data quality with extended range"""
        self.logger.info(f"Training data analysis for {gas_type}:")
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            
            if ppm_col in df.columns:
                ppm_data = df[ppm_col]
                ppm_mean = ppm_data.mean()
                ppm_std = ppm_data.std()
                ppm_max = ppm_data.max()
                ppm_min = ppm_data.min()
                
                self.logger.info(f"  {sensor}: PPM {ppm_mean:.0f}±{ppm_std:.0f} (range: {ppm_min:.0f}-{ppm_max:.0f})")
                
                # Check for good dynamic range
                if gas_type == 'normal':
                    if ppm_max < 50:
                        self.logger.info(f"  ✅ {sensor}: Good normal/clean air baseline")
                    if ppm_std / ppm_mean < 0.5:
                        self.logger.info(f"  ✅ {sensor}: Stable normal readings")
                else:
                    if ppm_max > 100:
                        self.logger.info(f"  ✅ {sensor}: Good high-concentration response")
                    if (ppm_max - ppm_min) > 50:
                        self.logger.info(f"  ✅ {sensor}: Good dynamic range")
                    if ppm_std / ppm_mean < 0.3:
                        self.logger.info(f"  ✅ {sensor}: Stable readings")

    def train_model(self):
        """Train ML model with extended features including normal class"""
        self.logger.info("Training model with extended range data (including normal class)...")
        
        training_data = self.load_training_data()
        if training_data is None:
            return False
        
        feature_columns = [
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
            'temperature', 'humidity'
        ]
        
        available_columns = [col for col in feature_columns if col in training_data.columns]
        
        X = training_data[available_columns].values
        y = training_data['gas_type'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Check if we have normal class data
        unique_classes = np.unique(y)
        self.logger.info(f"Training classes found: {list(unique_classes)}")
        
        if 'normal' not in unique_classes:
            self.logger.warning("⚠️  No 'normal' class found in training data!")
            self.logger.warning("⚠️  Please collect normal/clean air data first!")
            return False
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
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
        
        self.logger.info(f"Model accuracy: {accuracy:.3f}")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model
        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save feature columns
        model_metadata = {
            'feature_columns': available_columns,
            'accuracy': accuracy,
            'classes': list(unique_classes),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        self.is_model_trained = True
        self.logger.info("Model trained and saved successfully")
        
        return True

    def load_training_data(self):
        """Load training data"""
        data_files = list(Path("data").glob("training_*.csv"))
        if not data_files:
            self.logger.error("No training data files found!")
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
        self.logger.info(f"Total training samples: {len(combined_data)}")
        
        # Show class distribution
        class_counts = combined_data['gas_type'].value_counts()
        self.logger.info("Class distribution:")
        for gas_type, count in class_counts.items():
            self.logger.info(f"  {gas_type}: {count} samples")
        
        return combined_data

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_model_trained = True
            self.logger.info("Model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Predict gas type"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0
        
        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            feature_columns = metadata['feature_columns']
        except:
            feature_columns = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage', 
                             'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']
        
        feature_vector = []
        for feature in feature_columns:
            if feature == 'temperature':
                feature_vector.append(self.current_temperature)
            elif feature == 'humidity':
                feature_vector.append(self.current_humidity)
            else:
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
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        return prediction, confidence

    def enhanced_predict_gas(self, readings, confidence_threshold=0.6):
        """Enhanced prediction dengan confidence threshold untuk kondisi normal"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0
        
        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            feature_columns = metadata['feature_columns']
        except:
            feature_columns = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage', 
                             'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']
        
        feature_vector = []
        for feature in feature_columns:
            if feature == 'temperature':
                feature_vector.append(self.current_temperature)
            elif feature == 'humidity':
                feature_vector.append(self.current_humidity)
            else:
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
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        # Enhanced logic: Jika confidence rendah dan prediction bukan 'normal',
        # kemungkinan kondisi tidak dikenali (might be normal)
        if confidence < confidence_threshold and prediction != 'normal':
            return "Uncertain - Possibly Normal", confidence
        
        return prediction, confidence

    def detect_with_threshold(self, readings):
        """
        Deteksi gas dengan threshold untuk kondisi normal
        Jika semua PPM di bawah threshold -> Normal
        Jika ada yang tinggi -> Prediksi gas
        """
        
        # Define thresholds untuk kondisi normal (adjust sesuai kalibrasi Anda)
        normal_thresholds = {
            'TGS2600': 15,  # ppm
            'TGS2602': 12,   # ppm  
            'TGS2610': 8    # ppm
        }
        
        # Check apakah semua sensor di bawah threshold
        all_normal = True
        max_ppm = 0
        
        for sensor_name, threshold in normal_thresholds.items():
            sensor_ppm = readings[sensor_name]['ppm']
            max_ppm = max(max_ppm, sensor_ppm)
            
            if sensor_ppm > threshold:
                all_normal = False
                break
        
        if all_normal:
            return "Normal/Clean Air", 0.95  # High confidence untuk normal
        else:
            # Jika tidak normal, gunakan ML prediction
            return self.predict_gas(readings)

    def continuous_monitoring(self, duration=None, monitoring_mode='datasheet'):
        """
        Continuous monitoring with selectable mode
        Args:
            monitoring_mode: 'datasheet' for accurate detection, 'extended' for full range
        """
        self.set_sensor_mode(monitoring_mode)
        
        self.logger.info(f"Starting monitoring in {monitoring_mode.upper()} mode...")
        self.is_collecting = True
        
        fieldnames = [
            'timestamp', 'temperature', 'humidity', 'sensor_mode',
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
            'predicted_gas', 'confidence'
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
                    predicted_gas, confidence = self.enhanced_predict_gas(readings)
                    
                    data_row = {
                        'timestamp': timestamp,
                        'temperature': self.current_temperature,
                        'humidity': self.current_humidity,
                        'sensor_mode': monitoring_mode,
                        
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
                    
                    print(f"\r{timestamp.strftime('%H:%M:%S')} | Mode: {monitoring_mode.title()} | "
                          f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                          f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                          f"2610: {readings['TGS2610']['ppm']:.0f}ppm | "
                          f"Predicted: {predicted_gas} ({confidence:.2f})", end="")
                    
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
            
        self.is_collecting = False
        self.logger.info(f"Monitoring data saved to {monitoring_file}")

def main():
    """Main function with hybrid mode options - UPDATED dengan normal class"""
    gas_sensor = HybridGasSensorArray()
    
    gas_sensor.load_calibration()
    gas_sensor.load_model()
    
    while True:
        print("\n" + "="*60)
        print("HYBRID Gas Sensor Array System - USV Air Pollution Detection")
        print("Solusi untuk masalah PPM stuck di nilai maksimum + Normal Detection")
        print("="*60)
        print("1. Calibrate sensors")
        print("2. Collect training data (Extended mode - unlimited range)")
        print("3. Train machine learning model")
        print("4. Start monitoring - Datasheet mode (accurate detection)")
        print("5. Start monitoring - Extended mode (full range)")
        print("6. Test single reading")
        print("7. Switch sensor mode (Extended ↔ Datasheet)")
        print("8. View current sensor modes")
        print("9. Collect NORMAL/Clean Air data")  # NEW OPTION
        print("10. Exit")  # UPDATED from 9 to 10
        print("-"*60)
        
        try:
            choice = input("Select option (1-10): ").strip()  # UPDATED from 1-9 to 1-10
            
            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                gas_sensor.calibrate_sensors(duration)
                
            elif choice == '2':
                gas_types = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']  # UPDATED
                print("Available gas types:", ', '.join(gas_types))
                print("⚠️  IMPORTANT: Collect 'normal' data first for baseline!")  # NEW WARNING
                gas_type = input("Enter gas type: ").strip().lower()
                
                if gas_type not in gas_types:
                    print("Invalid gas type!")
                    continue
                
                duration = int(input("Collection duration (seconds, default 60): ") or 60)
                print("⚠️  Training mode: Extended range (no PPM limits)")
                gas_sensor.collect_training_data(gas_type, duration)
                
            elif choice == '3':
                if gas_sensor.train_model():
                    print("✅ Model training completed successfully!")
                else:
                    print("❌ Model training failed!")
                
            elif choice == '4':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None
                print("🎯 Monitoring in DATASHEET mode (accurate detection)")
                gas_sensor.continuous_monitoring(duration, 'datasheet')
                
            elif choice == '5':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None
                print("📊 Monitoring in EXTENDED mode (full range)")
                gas_sensor.continuous_monitoring(duration, 'extended')
                
            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.enhanced_predict_gas(readings)
                
                print("\n" + "="*50)
                print("SENSOR READINGS")
                print("="*50)
                
                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Voltage: {data['voltage']:.3f}V")
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    print(f"  PPM: {data['ppm']:.0f}")
                
                print(f"\nPredicted Gas: {predicted_gas} (Confidence: {confidence:.3f})")
                
                # Additional threshold-based detection
                threshold_result, threshold_confidence = gas_sensor.detect_with_threshold(readings)
                print(f"Threshold Detection: {threshold_result} (Confidence: {threshold_confidence:.3f})")
                
            elif choice == '7':
                current_mode = 'Extended' if not gas_sensor.sensor_config['TGS2600']['use_datasheet_mode'] else 'Datasheet'
                print(f"Current mode: {current_mode}")
                print("1. Extended mode (unlimited range for training)")
                print("2. Datasheet mode (accurate detection)")
                
                mode_choice = input("Select mode (1-2): ").strip()
                if mode_choice == '1':
                    gas_sensor.set_sensor_mode('extended')
                elif mode_choice == '2':
                    gas_sensor.set_sensor_mode('datasheet')
                else:
                    print("Invalid choice!")
                
            elif choice == '8':
                print("\nCurrent Sensor Modes:")
                for sensor_name, config in gas_sensor.sensor_config.items():
                    mode = 'Datasheet' if config['use_datasheet_mode'] else 'Extended'
                    range_info = config['datasheet_range'] if config['use_datasheet_mode'] else config['extended_range']
                    print(f"  {sensor_name}: {mode} mode (Range: {range_info} ppm)")
                
            elif choice == '9':  # NEW: Shortcut untuk collect normal data
                print("🌬️  Collecting NORMAL/Clean Air baseline data")
                print("Ensure sensors are in clean environment (no gas sources)")
                print("💡 Tips: Best time is early morning or after rain")
                print("💡 Keep away from: cooking, perfume, alcohol, fuel, cleaning products")
                duration = int(input("Collection duration (seconds, default 120): ") or 120)
                gas_sensor.collect_training_data('normal', duration)
                
            elif choice == '10':  # UPDATED: Exit option moved from 9 to 10
                print("Exiting system...")
                break
                
            else:
                print("Invalid option!")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except ValueError:
            print("Invalid input!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
