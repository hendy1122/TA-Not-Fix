#!/usr/bin/env python3
"""
Gas Detection System for Raspberry Pi - FIXED VERSION
Deteksi gas menggunakan sensor TGS array (2600, 2602, 2610) dengan ADC ADS1115
Author: Fixed Version
Date: 2025
"""

import time
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gas_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simulation mode for testing without actual hardware
SIMULATION_MODE = True

if not SIMULATION_MODE:
    try:
        import board
        import busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn
        logger.info("Hardware libraries loaded successfully")
    except ImportError:
        logger.warning("Hardware libraries not available, switching to simulation mode")
        SIMULATION_MODE = True

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

class GasDetectionSystem:
    def __init__(self, config_file='config.json'):
        """Initialize gas detection system"""
        self.config = self.load_config(config_file)
        self.baseline_values = None
        self.model = None
        self.label_encoder = None

        # Setup components
        self.setup_baseline()
        self.load_or_train_model()

        if not SIMULATION_MODE:
            self.setup_adc()
        else:
            logger.info("Running in simulation mode - no hardware required")

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "adc_channels": [0, 1, 2],
            "adc_gain": 1,
            "sampling_rate": 2,
            "csv_output": "gas_detection_results.csv",
            "model_file": "gas_model.pkl",
            "label_encoder_file": "label_encoder.pkl",
            "dataset_file": "merged_dataset.csv.csv",  # Sesuai dengan nama file yang ada
            "sensor_names": ["TGS_2600", "TGS_2602", "TGS_2610"],
            "voltage_to_ppm_factors": [100, 80, 150]
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                # Merge with defaults to ensure all keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        logger.warning(f"Using default configuration")
        # Save default config
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Default configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Could not save default config: {e}")

        return default_config

    def setup_adc(self):
        """Setup ADC ADS1115"""
        if SIMULATION_MODE:
            return

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(i2c)
            self.ads.gain = self.config["adc_gain"]

            self.channels = []
            for ch in self.config["adc_channels"]:
                if ch == 0:
                    self.channels.append(AnalogIn(self.ads, ADS.P0))
                elif ch == 1:
                    self.channels.append(AnalogIn(self.ads, ADS.P1))
                elif ch == 2:
                    self.channels.append(AnalogIn(self.ads, ADS.P2))
                elif ch == 3:
                    self.channels.append(AnalogIn(self.ads, ADS.P3))

            logger.info("ADC ADS1115 initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ADC: {e}")
            logger.info("Switching to simulation mode")
            SIMULATION_MODE = True

    def setup_baseline(self):
        """Setup baseline values from dataset"""
        try:
            if not os.path.exists(self.config["dataset_file"]):
                logger.error(f"Dataset file {self.config['dataset_file']} not found!")
                # Use default baseline values
                self.baseline_values = {'TGS_2600': 148.5, 'TGS_2602': 51.5, 'TGS_2610': 205.0}
                return

            # Read CSV dengan delimiter yang benar (semicolon)
            df = pd.read_csv(self.config["dataset_file"], delimiter=';')

            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()

            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Labels in dataset: {df['label'].unique()}")

            # Find normal/baseline data
            normal_data = df[df['label'].str.lower().str.contains('normal', na=False)]

            if len(normal_data) > 0:
                self.baseline_values = {
                    'TGS_2600': normal_data['TGS 2600'].mean(),
                    'TGS_2602': normal_data['TGS 2602'].mean(),
                    'TGS_2610': normal_data['TGS 2610'].mean()
                }
                logger.info(f"Baseline values calculated from normal data: {self.baseline_values}")
            else:
                # Use overall mean as baseline
                self.baseline_values = {
                    'TGS_2600': df['TGS 2600'].mean(),
                    'TGS_2602': df['TGS 2602'].mean(),
                    'TGS_2610': df['TGS 2610'].mean()
                }
                logger.warning("No 'normal' data found, using overall mean as baseline")

        except Exception as e:
            logger.error(f"Failed to setup baseline: {e}")
            # Default baseline values based on your dataset
            self.baseline_values = {'TGS_2600': 148.5, 'TGS_2602': 51.5, 'TGS_2610': 205.0}

    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_exists = os.path.exists(self.config["model_file"])
        encoder_exists = os.path.exists(self.config["label_encoder_file"])

        if model_exists and encoder_exists:
            try:
                with open(self.config["model_file"], 'rb') as f:
                    self.model = pickle.load(f)

                with open(self.config["label_encoder_file"], 'rb') as f:
                    self.label_encoder = pickle.load(f)

                logger.info("Pre-trained model loaded successfully")
                return
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

        # Train new model
        logger.info("Training new model from dataset...")
        self.train_model()

    def train_model(self):
        """Train model from dataset"""
        try:
            if not os.path.exists(self.config["dataset_file"]):
                logger.error(f"Dataset file {self.config['dataset_file']} not found!")
                raise FileNotFoundError("Dataset file not found")

            # Load and prepare data dengan delimiter semicolon
            df = pd.read_csv(self.config["dataset_file"], delimiter=';')
            df.columns = df.columns.str.strip()  # Clean column names

            # Clean data - remove rows with spaces in numeric columns
            numeric_columns = ['TGS 2600', 'TGS 2602', 'TGS 2610']
            for col in numeric_columns:
                # Convert to string first, then strip spaces, then convert to numeric
                df[col] = df[col].astype(str).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values
            df = df.dropna()

            logger.info(f"Dataset shape after cleaning: {df.shape}")
            logger.info(f"Labels in dataset: {df['label'].unique()}")

            # Prepare features and labels
            X = df[['TGS 2600', 'TGS 2602', 'TGS 2610']].values
            y = df['label'].values

            logger.info(f"Training data shape: {X.shape}")

            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Model trained successfully!")
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"Classes: {self.label_encoder.classes_}")

            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

            # Save model and encoder
            with open(self.config["model_file"], 'wb') as f:
                pickle.dump(self.model, f)

            with open(self.config["label_encoder_file"], 'wb') as f:
                pickle.dump(self.label_encoder, f)

            logger.info("Model and encoder saved successfully")

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def read_sensors(self):
        """Read values from all gas sensors"""
        if SIMULATION_MODE:
            return self.simulate_sensor_data()

        try:
            sensor_data = {}

            for i, (channel, sensor_name, conversion_factor) in enumerate(
                zip(self.channels, self.config["sensor_names"], self.config["voltage_to_ppm_factors"])
            ):
                voltage = channel.voltage
                ppm = voltage * conversion_factor

                sensor_data[sensor_name] = {
                    'voltage': voltage,
                    'ppm': ppm,
                    'raw_value': channel.value
                }

            return sensor_data

        except Exception as e:
            logger.error(f"Failed to read sensors: {e}")
            return self.simulate_sensor_data()

    def simulate_sensor_data(self):
        """Simulate sensor data for testing"""
        # Simulate different gas types berdasarkan data asli
        gas_types = ['normal', 'alkohol', 'pertalite', 'pertamax']
        current_gas = random.choice(gas_types)

        if current_gas == 'normal':
            base_values = [148.5, 51.5, 205.0]
        elif current_gas == 'alkohol':
            base_values = [2500, 120, 2800]
        elif current_gas == 'pertalite':
            base_values = [800, 250, 1500]
        else:  # pertamax
            base_values = [1200, 220, 2500]

        # Add some noise
        noise_factor = 0.1
        sensor_data = {}
        sensor_names = ['TGS_2600', 'TGS_2602', 'TGS_2610']

        for i, (sensor_name, base_val) in enumerate(zip(sensor_names, base_values)):
            noise = base_val * noise_factor * (random.random() - 0.5)
            ppm = max(0, base_val + noise)  # Ensure positive values
            voltage = ppm / self.config["voltage_to_ppm_factors"][i]

            sensor_data[sensor_name] = {
                'voltage': voltage,
                'ppm': ppm,
                'raw_value': int(ppm * 10)  # Simulated raw ADC value
            }

        return sensor_data

    def predict_gas(self, sensor_data):
        """Predict gas type from sensor readings"""
        try:
            if self.model is None or self.label_encoder is None:
                return "Model not loaded", 0.0

            features = [
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['ppm'],
                sensor_data['TGS_2610']['ppm']
            ]

            features = np.array(features).reshape(1, -1)

            prediction_encoded = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]

            predicted_gas = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = max(prediction_proba) * 100

            return predicted_gas, confidence

        except Exception as e:
            logger.error(f"Failed to predict gas: {e}")
            return "Error", 0.0

    def save_to_csv(self, timestamp, sensor_data, predicted_gas, confidence):
        """Save data to CSV file"""
        try:
            file_exists = os.path.exists(self.config["csv_output"])

            with open(self.config["csv_output"], 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp',
                    'TGS_2600_ppm', 'TGS_2600_voltage',
                    'TGS_2602_ppm', 'TGS_2602_voltage',
                    'TGS_2610_ppm', 'TGS_2610_voltage',
                    'predicted_gas', 'confidence'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': timestamp,
                    'TGS_2600_ppm': round(sensor_data['TGS_2600']['ppm'], 2),
                    'TGS_2600_voltage': round(sensor_data['TGS_2600']['voltage'], 3),
                    'TGS_2602_ppm': round(sensor_data['TGS_2602']['ppm'], 2),
                    'TGS_2602_voltage': round(sensor_data['TGS_2602']['voltage'], 3),
                    'TGS_2610_ppm': round(sensor_data['TGS_2610']['ppm'], 2),
                    'TGS_2610_voltage': round(sensor_data['TGS_2610']['voltage'], 3),
                    'predicted_gas': predicted_gas,
                    'confidence': round(confidence, 1)
                })

        except Exception as e:
            logger.error(f"Failed to save to CSV: {e}")

    def run_detection(self):
        """Main detection loop"""
        mode_str = "SIMULATION" if SIMULATION_MODE else "HARDWARE"
        logger.info(f"Starting gas detection system in {mode_str} mode...")
        logger.info("Press Ctrl+C to stop")

        try:
            reading_count = 0
            while True:
                reading_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                sensor_data = self.read_sensors()

                if sensor_data:
                    predicted_gas, confidence = self.predict_gas(sensor_data)

                    # Display results
                    print(f"\n[Reading #{reading_count}] {timestamp}")
                    print("=" * 60)
                    if SIMULATION_MODE:
                        print("🔬 SIMULATION MODE")

                    for sensor_name, data in sensor_data.items():
                        print(f"{sensor_name:10}: {data['ppm']:8.2f} ppm ({data['voltage']:6.3f}V)")

                    print(f"{'Prediction':<10}: {predicted_gas} (Confidence: {confidence:.1f}%)")

                    # Determine alert level
                    if predicted_gas.lower() != 'normal' and confidence > 70:
                        print("⚠️  GAS DETECTED - HIGH CONFIDENCE")
                    elif predicted_gas.lower() != 'normal' and confidence > 50:
                        print("⚠️  Possible gas detection - Medium confidence")
                    else:
                        print("✅ Normal conditions")

                    self.save_to_csv(timestamp, sensor_data, predicted_gas, confidence)

                time.sleep(self.config["sampling_rate"])

        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            logger.info("Gas detection system stopped")

def main():
    """Main function"""
    try:
        print("🚀 Gas Detection System Starting...")
        print("=" * 50)

        gas_detector = GasDetectionSystem()
        gas_detector.run_detection()

    except Exception as e:
        logger.error(f"Failed to start gas detection system: {e}")
        print(f"Error: {e}")

        # Provide troubleshooting info
        print("\n🔧 Troubleshooting:")
        print("1. Make sure merged_dataset.csv.csv file exists")
        print("2. Check if all required Python packages are installed:")
        print("   pip install pandas scikit-learn numpy")
        print("3. For hardware mode, install Adafruit libraries:")
        print("   pip install adafruit-circuitpython-ads1x15")

if __name__ == "__main__":
    main()