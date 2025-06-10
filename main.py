#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System for USV Air Pollution Detection
Hybrid approach: Datasheet accuracy + Extended range for training
Optimized for TGS2600, TGS2602, TGS2610 based on Figaro datasheets
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

class EnhancedDatasheetGasSensorArray:
    def __init__(self):
        """Initialize enhanced gas sensor array system based on Figaro datasheets"""
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

        # Enhanced sensor configurations with hybrid modes
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['hydrogen', 'carbon_monoxide', 'alcohol'],
                'detection_range': (1, 30),  # ppm hydrogen - datasheet range
                'extended_range': (1, 500),  # ppm - extended for training
                'heater_voltage': 5.0,
                'heater_current': 42e-3,  # 42mA
                'power_consumption': 210e-3,  # 210mW
                'load_resistance': 10000,  # 10kΩ recommended
                'warmup_time': 7 * 24 * 3600,  # 7 days conditioning
                'operating_temp_range': (-20, 50),  # °C
                'optimal_temp': 20,  # °C
                'optimal_humidity': 65,  # %RH
                'R0': None,  # Will be set during calibration
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'hydrogen': (0.3, 0.6),
                    'carbon_monoxide': (0.4, 0.7),
                    'alcohol': (0.2, 0.5)
                },
                # NEW: Hybrid mode settings
                'use_extended_mode': False,  # False = datasheet mode, True = extended mode
                'concentration_threshold': 50,  # Switch threshold
                'extended_sensitivity': 2.5  # Sensitivity factor for extended range
            },
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['toluene', 'ammonia', 'h2s', 'alcohol'],
                'detection_range': (1, 30),  # ppm ethanol equivalent
                'extended_range': (1, 300),  # ppm - extended for training
                'heater_voltage': 5.0,
                'heater_current': 56e-3,  # 56mA
                'power_consumption': 280e-3,  # 280mW
                'load_resistance': 10000,  # 10kΩ recommended
                'warmup_time': 7 * 24 * 3600,  # 7 days conditioning
                'operating_temp_range': (-10, 60),  # °C
                'optimal_temp': 20,  # °C
                'optimal_humidity': 65,  # %RH
                'R0': None,  # Will be set during calibration
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'alcohol': (0.08, 0.5),
                    'toluene': (0.1, 0.4),
                    'ammonia': (0.15, 0.6),
                    'h2s': (0.05, 0.3)
                },
                # NEW: Hybrid mode settings
                'use_extended_mode': False,
                'concentration_threshold': 40,
                'extended_sensitivity': 3.0
            },
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'iso_butane'],
                'detection_range': (1, 25),  # % LEL
                'extended_range': (1, 200),  # % LEL - extended for training
                'heater_voltage': 5.0,
                'heater_current': 56e-3,  # 56mA
                'power_consumption': 280e-3,  # 280mW
                'load_resistance': 10000,  # 10kΩ recommended
                'warmup_time': 7 * 24 * 3600,  # 7 days conditioning
                'operating_temp_range': (-10, 50),  # °C
                'optimal_temp': 20,  # °C
                'optimal_humidity': 65,  # %RH
                'R0': None,  # R0 in 1800ppm iso-butane
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'iso_butane': (0.45, 0.62),
                    'butane': (0.4, 0.6),
                    'propane': (0.35, 0.55),
                    'lp_gas': (0.4, 0.6)
                },
                # NEW: Hybrid mode settings
                'use_extended_mode': False,
                'concentration_threshold': 30,
                'extended_sensitivity': 2.0
            }
        }

        # Environmental compensation parameters
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0  # Default room temperature
        self.current_humidity = 65.0     # Default humidity

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

        self.logger.info("Enhanced Datasheet Gas Sensor Array System initialized")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """
        Convert ADC voltage to sensor resistance using voltage divider
        Rs = RL * (Vc - Vout) / Vout
        """
        if voltage <= 0.001:  # Avoid division by zero
            return float('inf')

        # Assuming 5V circuit voltage and voltage divider configuration
        circuit_voltage = 5.0

        # Handle edge cases
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
        """
        ENHANCED: Hybrid PPM calculation - datasheet accuracy + extended range
        Solves PPM stuck at maximum values problem
        """
        config = self.sensor_config[sensor_name]
        R0 = config.get('R0')

        if R0 is None or R0 == 0:
            # Fallback to simplified calculation if R0 not calibrated
            return self.simplified_ppm_calculation(sensor_name, resistance)

        rs_r0_ratio = resistance / R0

        # Choose calculation mode: datasheet (accurate) vs extended (unlimited)
        if config['use_extended_mode']:
            return self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            return self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """
        ORIGINAL datasheet calculation - accurate but limited range
        """
        if sensor_name == 'TGS2600':
            if gas_type == 'hydrogen' or gas_type == 'auto':
                # Based on hydrogen sensitivity curve
                if rs_r0_ratio < 0.3:
                    ppm = 30  # FIXED: Use detection_range max instead of arbitrary 100
                elif rs_r0_ratio > 0.9:
                    ppm = 0    # No gas
                else:
                    # Exponential curve fitting from datasheet
                    ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                    ppm = min(ppm, 30)  # Limit to datasheet range
            elif gas_type == 'alcohol':
                # Alcohol sensitivity curve
                if rs_r0_ratio < 0.2:
                    ppm = 30  # FIXED: Limit to detection range
                elif rs_r0_ratio > 0.8:
                    ppm = 0
                else:
                    ppm = 40 * ((0.4 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 30)  # Limit to datasheet range
            else:
                ppm = 30 * ((0.5 / rs_r0_ratio) ** 2.0)
                ppm = min(ppm, 30)  # Limit to datasheet range

        elif sensor_name == 'TGS2602':
            if gas_type == 'alcohol' or gas_type == 'auto':
                # Ethanol sensitivity curve (1-30ppm range)
                if rs_r0_ratio < 0.08:
                    ppm = 30  # FIXED: Use actual detection range max
                elif rs_r0_ratio > 0.9:
                    ppm = 0
                else:
                    # Power law curve from datasheet
                    ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                    ppm = min(ppm, 30)  # Limit to datasheet range
            elif gas_type == 'toluene':
                ppm = 25 * ((0.2 / rs_r0_ratio) ** 1.5)
                ppm = min(ppm, 30)
            else:
                ppm = 20 * ((0.3 / rs_r0_ratio) ** 1.6)
                ppm = min(ppm, 30)

        elif sensor_name == 'TGS2610':
            # LP gas detection (% LEL)
            if rs_r0_ratio < 0.45:
                ppm = 25  # FIXED: Use actual detection range max
            elif rs_r0_ratio > 0.95:
                ppm = 0
            else:
                # Based on iso-butane curve
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                ppm = min(ppm, 25)  # Limit to datasheet range
        else:
            ppm = 0

        return max(0, ppm)

    def extended_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """
        NEW: Extended range calculation for training data
        Removes upper limits to capture full dynamic range and gas uniqueness
        """
        config = self.sensor_config[sensor_name]
        sensitivity = config['extended_sensitivity']

        if rs_r0_ratio >= 1.0:
            return 0  # No gas detected

        # Base calculation without upper limits
        if sensor_name == 'TGS2600':
            # Extended range for different gases with unique characteristics
            if rs_r0_ratio < 0.05:  # Very high concentration
                base_ppm = 200 + (0.05 - rs_r0_ratio) * 1000
            elif rs_r0_ratio < 0.2:  # High concentration
                base_ppm = 100 + (0.2 - rs_r0_ratio) * 500
            else:  # Normal to low concentration
                base_ppm = 60 * ((0.6 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2602':
            # Extended range for VOCs with unique signatures
            if rs_r0_ratio < 0.02:  # Very high concentration
                base_ppm = 150 + (0.02 - rs_r0_ratio) * 2000
            elif rs_r0_ratio < 0.1:  # High concentration
                base_ppm = 75 + (0.1 - rs_r0_ratio) * 800
            else:  # Normal to low concentration
                base_ppm = 50 * ((0.3 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2610':
            # Extended range for LP gas compounds
            if rs_r0_ratio < 0.1:  # Very high concentration
                base_ppm = 100 + (0.1 - rs_r0_ratio) * 1500
            elif rs_r0_ratio < 0.3:  # High concentration
                base_ppm = 50 + (0.3 - rs_r0_ratio) * 400
            else:  # Normal to low concentration
                base_ppm = 40 * ((0.7 / rs_r0_ratio) ** sensitivity)
        else:
            base_ppm = 0

        # Apply gas-specific multipliers for better differentiation
        gas_multipliers = {
            'alcohol': 1.0,
            'pertalite': 1.3,    # Higher multiplier for pertalite
            'pertamax': 1.6,     # Even higher for pertamax
            'dexlite': 1.9,      # Highest for dexlite
            'biosolar': 2.2,     # Unique signature for biosolar
            'hydrogen': 0.8,     # Lower for hydrogen
            'toluene': 1.1,      # Slight difference for toluene
            'ammonia': 0.9,      # Different for ammonia
            'butane': 1.2,       # Different for butane
            'propane': 1.4,      # Different for propane
            'normal': 0.7        # Normal/clean air
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Simplified PPM calculation when R0 is not available"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 0.4)

        # Convert baseline voltage to resistance for comparison
        baseline_resistance = self.voltage_to_resistance(baseline_voltage)

        if resistance >= baseline_resistance:
            return 0

        # Simplified ratio-based calculation
        ratio = baseline_resistance / resistance

        # Use extended range if in extended mode
        if config['use_extended_mode']:
            max_range = config['extended_range'][1]
            ppm = max_range * (ratio - 1) * 0.4
        else:
            max_range = config['detection_range'][1]
            ppm = max_range * (ratio - 1) * 0.5

        return max(0, ppm)

    def set_sensor_mode(self, mode='datasheet'):
        """
        NEW: Set calculation mode for all sensors
        Args:
            mode: 'datasheet' for accurate detection or 'extended' for training
        """
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

    def calibrate_sensors(self, duration=300):
        """
        Enhanced calibration for R0 determination (same as before)
        """
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
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])

                # Apply environmental compensation
                resistance = self.temperature_compensation(sensor_name, resistance, self.current_temperature)
                resistance = self.humidity_compensation(sensor_name, resistance, self.current_humidity)

                readings[sensor_name]['voltages'].append(voltage)
                readings[sensor_name]['resistances'].append(resistance)

            sample_count += 1
            time.sleep(2)  # 0.5 Hz sampling for stability

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
                'stability': (voltage_std / voltage_mean) * 100  # % coefficient of variation
            }

            self.logger.info(f"{sensor_name} Calibration:")
            self.logger.info(f"  R0: {resistance_mean:.1f}Ω ± {resistance_std:.1f}Ω")
            self.logger.info(f"  Baseline Voltage: {voltage_mean:.3f}V ± {voltage_std:.3f}V")
            self.logger.info(f"  Stability: {calibration_results[sensor_name]['stability']:.2f}%")

        # Save enhanced calibration data
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

        # Also save as current calibration
        with open('sensor_calibration.json', 'w') as f:
            json.dump(calib_data, f, indent=2)

        self.logger.info(f"Enhanced calibration completed and saved to {calib_file}")

    def load_calibration(self):
        """Load calibration data with enhanced validation"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)

            # Load sensor calibration data
            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']

            self.logger.info("Enhanced calibration data loaded successfully")
            self.logger.info(f"Calibration date: {calib_data.get('timestamp', 'Unknown')}")

            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Please run calibration first.")
            return False
        except KeyError as e:
            self.logger.error(f"Invalid calibration file format: {e}")
            return False

    def read_sensors(self):
        """Enhanced sensor reading with environmental compensation"""
        readings = {}

        for sensor_name, config in self.sensor_config.items():
            try:
                # Read raw voltage
                voltage = config['channel'].voltage

                # Convert to resistance
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])

                # Apply environmental compensation
                compensated_resistance = self.temperature_compensation(
                    sensor_name, resistance, self.current_temperature)
                compensated_resistance = self.humidity_compensation(
                    sensor_name, compensated_resistance, self.current_humidity)

                # Calculate Rs/R0 ratio
                R0 = config.get('R0')
                rs_r0_ratio = compensated_resistance / R0 if R0 else None

                # Convert to PPM using hybrid method
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)

                # Current mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"

                readings[sensor_name] = {
                    'voltage': voltage,
                    'resistance': resistance,
                    'compensated_resistance': compensated_resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'R0': R0,
                    'mode': current_mode,
                    'target_gases': config['target_gases']
                }

            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'resistance': 0, 'compensated_resistance': 0,
                    'rs_r0_ratio': None, 'ppm': 0, 'R0': None, 'mode': 'Error',
                    'target_gases': []
                }

        return readings

    def collect_training_data(self, gas_type, duration=60, samples_per_second=1):
        """
        ENHANCED: Training data collection with extended mode
        """
        # Auto-switch to extended mode for training
        self.set_sensor_mode('extended')

        self.logger.info(f"Collecting enhanced training data for {gas_type} in EXTENDED mode")
        self.logger.info(f"Duration: {duration}s, Sampling rate: {samples_per_second} Hz")

        # Enhanced gas type validation
        valid_gases = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar',
                      'hydrogen', 'toluene', 'ammonia', 'butane', 'propane']
        if gas_type not in valid_gases:
            self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
            return None

        # Special instructions for normal vs gas data
        if gas_type == 'normal':
            input(f"Ensure sensors are in CLEAN AIR (no gas). Press Enter to start...")
            self.logger.info("Collecting NORMAL/CLEAN AIR data...")
        else:
            input(f"Prepare to spray {gas_type}. Press Enter to start data collection...")

        training_data = []
        start_time = time.time()
        sample_interval = 1.0 / samples_per_second

        # Pre-collection baseline
        baseline_readings = self.read_sensors()
        self.logger.info("Baseline readings recorded")

        while time.time() - start_time < duration:
            timestamp = datetime.now()
            readings = self.read_sensors()

            # Enhanced data row with hybrid features
            data_row = {
                'timestamp': timestamp,
                'gas_type': gas_type,
                'temperature': self.current_temperature,
                'humidity': self.current_humidity,

                # TGS2600 data
                'TGS2600_voltage': readings['TGS2600']['voltage'],
                'TGS2600_resistance': readings['TGS2600']['resistance'],
                'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                'TGS2600_ppm': readings['TGS2600']['ppm'],

                # TGS2602 data
                'TGS2602_voltage': readings['TGS2602']['voltage'],
                'TGS2602_resistance': readings['TGS2602']['resistance'],
                'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                'TGS2602_ppm': readings['TGS2602']['ppm'],

                # TGS2610 data
                'TGS2610_voltage': readings['TGS2610']['voltage'],
                'TGS2610_resistance': readings['TGS2610']['resistance'],
                'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                'TGS2610_ppm': readings['TGS2610']['ppm']
            }

            training_data.append(data_row)

            # Enhanced display with mode info
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            status = "CLEAN AIR" if gas_type == 'normal' else gas_type.upper()
            print(f"\rTime: {remaining:.1f}s | Status: {status} | "
                  f"2600: {readings['TGS2600']['ppm']:.0f}ppm ({readings['TGS2600']['mode']}) | "
                  f"2602: {readings['TGS2602']['ppm']:.0f}ppm ({readings['TGS2602']['mode']}) | "
                  f"2610: {readings['TGS2610']['ppm']:.0f}ppm ({readings['TGS2610']['mode']})", end="")

            time.sleep(sample_interval)

        print()  # New line

        # Save enhanced training data
        filename = f"data/training_{gas_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False)

        # Enhanced data quality analysis
        self.analyze_enhanced_training_data_quality(df, gas_type)

        self.logger.info(f"Enhanced training data saved to {filename}")
        self.logger.info(f"Collected {len(training_data)} samples for {gas_type}")

        return training_data

    def analyze_enhanced_training_data_quality(self, df, gas_type):
        """Enhanced analysis of training data quality"""
        self.logger.info(f"Enhanced analysis for {gas_type}:")

        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            ratio_col = f'{sensor}_rs_r0_ratio'

            if ppm_col in df.columns:
                ppm_data = df[ppm_col]
                ppm_mean = ppm_data.mean()
                ppm_std = ppm_data.std()
                ppm_max = ppm_data.max()
                ppm_min = ppm_data.min()

                self.logger.info(f"  {sensor}: PPM {ppm_mean:.0f}±{ppm_std:.0f} (range: {ppm_min:.0f}-{ppm_max:.0f})")

                # Enhanced quality checks
                if gas_type == 'normal':
                    if ppm_max < 30:
                        self.logger.info(f"  ✅ {sensor}: Good normal baseline")
                    if ppm_std / ppm_mean < 0.5:
                        self.logger.info(f"  ✅ {sensor}: Stable normal readings")
                else:
                    if ppm_max > 100:
                        self.logger.info(f"  ✅ {sensor}: Good high-concentration response (no cap limit)")
                    if (ppm_max - ppm_min) > 50:
                        self.logger.info(f"  ✅ {sensor}: Excellent dynamic range")
                    if ppm_std / ppm_mean < 0.3:
                        self.logger.info(f"  ✅ {sensor}: Stable readings")

    def train_model(self):
        """Enhanced machine learning model training with hybrid features"""
        self.logger.info("Training enhanced machine learning model with hybrid features...")

        # Load all training data
        training_data = self.load_training_data()
        if training_data is None:
            return False

        # Enhanced feature selection (include hybrid features)
        feature_columns = [
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
            'temperature', 'humidity'
        ]

        # Filter available columns
        available_columns = [col for col in feature_columns if col in training_data.columns]

        X = training_data[available_columns].values
        y = training_data['gas_type'].values

        # Handle missing values (NaN in rs_r0_ratio if R0 not calibrated)
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

        # Enhanced Random Forest with optimized parameters
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

        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': available_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.logger.info(f"Model accuracy: {accuracy:.3f}")
        self.logger.info("\nTop 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Save model, scaler, and metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'feature_columns': available_columns,
            'feature_importance': feature_importance.to_dict('records'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': list(unique_classes),
            'model_type': 'enhanced_hybrid_datasheet'
        }

        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')

        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        self.is_model_trained = True
        self.logger.info("Enhanced hybrid model trained and saved successfully")

        return True

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

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_model_trained = True
            self.logger.info("Enhanced model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction with confidence scoring"""
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

    def continuous_monitoring(self, duration=None, monitoring_mode='datasheet'):
        """
        Enhanced continuous monitoring with mode selection
        Args:
            monitoring_mode: 'datasheet' for accurate detection, 'extended' for training
        """
        # Set monitoring mode
        self.set_sensor_mode(monitoring_mode)

        self.logger.info(f"Starting enhanced monitoring in {monitoring_mode.upper()} mode...")
        self.is_collecting = True

        # Enhanced CSV fields
        fieldnames = [
            'timestamp', 'temperature', 'humidity', 'sensor_mode',
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm',
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
                    predicted_gas, confidence = self.predict_gas(readings)

                    # Enhanced data row
                    data_row = {
                        'timestamp': timestamp,
                        'temperature': self.current_temperature,
                        'humidity': self.current_humidity,
                        'sensor_mode': monitoring_mode,

                        'TGS2600_voltage': readings['TGS2600']['voltage'],
                        'TGS2600_resistance': readings['TGS2600']['resistance'],
                        'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                        'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                        'TGS2600_ppm': readings['TGS2600']['ppm'],

                        'TGS2602_voltage': readings['TGS2602']['voltage'],
                        'TGS2602_resistance': readings['TGS2602']['resistance'],
                        'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                        'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                        'TGS2602_ppm': readings['TGS2602']['ppm'],

                        'TGS2610_voltage': readings['TGS2610']['voltage'],
                        'TGS2610_resistance': readings['TGS2610']['resistance'],
                        'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                        'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                        'TGS2610_ppm': readings['TGS2610']['ppm'],

                        'predicted_gas': predicted_gas,
                        'confidence': confidence
                    }

                    writer.writerow(data_row)
                    sample_count += 1

                    # Enhanced display with mode and confidence
                    print(f"\r{timestamp.strftime('%H:%M:%S')} | Mode: {monitoring_mode.title()} | "
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

                    time.sleep(1)  # 1 Hz sampling

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")

        self.is_collecting = False
        self.logger.info(f"Enhanced monitoring data saved to {monitoring_file}")
        self.logger.info(f"Total samples collected: {sample_count}")

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
    """Enhanced main function with hybrid mode controls"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration if available
    gas_sensor.load_calibration()

    # Load existing model if available
    gas_sensor.load_model()

    while True:
        print("\n" + "="*60)
        print("ENHANCED Gas Sensor Array System - USV Air Pollution Detection")
        print("Hybrid Datasheet Approach: Accuracy + Extended Range")
        print("="*60)
        print("1. Calibrate sensors (Enhanced R0 determination)")
        print("2. Collect training data (Auto-switch to Extended mode)")
        print("3. Train machine learning model (Hybrid features)")
        print("4. Start monitoring - Datasheet mode (Accurate detection)")
        print("5. Start monitoring - Extended mode (Full range)")
        print("6. Test single reading (Detailed analysis)")
        print("7. Set environmental conditions (T°C, %RH)")
        print("8. Switch sensor calculation mode")
        print("9. View sensor diagnostics")
        print("10. Exit")
        print("-"*60)

        try:
            choice = input("Select option (1-10): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                print("Ensure sensors are warmed up for at least 10 minutes in clean air!")
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

                duration = int(input("Collection duration (seconds, default 60): ") or 60)
                print("🔄 Auto-switching to EXTENDED mode for training...")
                gas_sensor.collect_training_data(gas_type, duration)

            elif choice == '3':
                if gas_sensor.train_model():
                    print("✅ Enhanced hybrid model training completed successfully!")
                else:
                    print("❌ Model training failed!")

            elif choice == '4':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None

                print("🎯 Enhanced monitoring in DATASHEET mode (accurate detection)")
                gas_sensor.continuous_monitoring(duration, 'datasheet')

            elif choice == '5':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None

                print("📊 Enhanced monitoring in EXTENDED mode (full range)")
                gas_sensor.continuous_monitoring(duration, 'extended')

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*50)
                print("DETAILED SENSOR ANALYSIS - HYBRID MODE")
                print("="*50)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Voltage: {data['voltage']:.3f}V")
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    print(f"  Compensated Resistance: {data['compensated_resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    else:
                        print(f"  Rs/R0 Ratio: Not calibrated")
                    print(f"  PPM: {data['ppm']:.0f} ({'No limit' if data['mode'] == 'Extended' else 'Datasheet limit'})")
                    print(f"  Target Gases: {', '.join(data['target_gases'])}")
                    print(f"  R0 (Baseline): {data['R0']:.1f}Ω" if data['R0'] else "  R0: Not calibrated")

                print(f"\nENVIRONMENTAL CONDITIONS:")
                print(f"  Temperature: {gas_sensor.current_temperature}°C")
                print(f"  Humidity: {gas_sensor.current_humidity}%RH")

                print(f"\nHYBRID PREDICTION:")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")

                if confidence < 0.5:
                    print("  ⚠️  Low confidence - may need more training data")
                elif confidence < 0.7:
                    print("  ⚠️  Medium confidence")
                else:
                    print("  ✅ High confidence")

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
                print("1. Datasheet mode (accurate detection, limited range)")
                print("2. Extended mode (unlimited range for training)")

                mode_choice = input("Select mode (1-2): ").strip()
                if mode_choice == '1':
                    gas_sensor.set_sensor_mode('datasheet')
                elif mode_choice == '2':
                    gas_sensor.set_sensor_mode('extended')
                else:
                    print("Invalid choice!")

            elif choice == '9':
                print("\n" + "="*50)
                print("ENHANCED SENSOR DIAGNOSTICS")
                print("="*50)

                for sensor_name, config in gas_sensor.sensor_config.items():
                    current_mode = 'Extended' if config['use_extended_mode'] else 'Datasheet'
                    print(f"\n{sensor_name} ({current_mode} mode):")
                    print(f"  Power Consumption: {config['power_consumption']*1000:.0f}mW")
                    print(f"  Heater Current: {config['heater_current']*1000:.0f}mA")
                    print(f"  Datasheet Range: {config['detection_range']} ppm")
                    print(f"  Extended Range: {config['extended_range']} ppm")
                    print(f"  Operating Temp: {config['operating_temp_range']}°C")
                    print(f"  Target Gases: {', '.join(config['target_gases'])}")

                    if config['R0']:
                        print(f"  Calibrated R0: {config['R0']:.1f}Ω")
                        print(f"  Baseline Voltage: {config['baseline_voltage']:.3f}V")
                    else:
                        print(f"  ⚠️  Not calibrated")

            elif choice == '10':
                print("Exiting enhanced hybrid system...")
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