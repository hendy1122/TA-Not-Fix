#!/usr/bin/env python3
"""
Gas Detection System for Raspberry Pi
Deteksi gas menggunakan sensor TGS array (2600, 2602, 2610) dengan ADC ADS1115
Author: [Your Name]
Date: 2025
"""

import time
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

class GasDetectionSystem:
    def __init__(self, config_file='config.json'):
        """Initialize gas detection system"""
        self.config = self.load_config(config_file)
        self.setup_adc()
        self.load_model()
        self.baseline_values = None
        self.setup_baseline()
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using default values")
            return self.get_default_config()
    
    def get_default_config(self):
        """Default configuration"""
        return {
            "adc_channels": [0, 1, 2],  # ADS1115 channels for TGS 2600, 2602, 2610
            "adc_gain": 1,  # ADS1115 gain
            "sampling_rate": 1,  # seconds between readings
            "csv_output": "gas_detection_results.csv",
            "model_file": "gas_model.pkl",
            "label_encoder_file": "label_encoder.pkl",
            "dataset_file": "merged_dataset.csv",
            "sensor_names": ["TGS_2600", "TGS_2602", "TGS_2610"],
            "voltage_to_ppm_factors": [100, 80, 150]  # Conversion factors (adjust based on your calibration)
        }
    
    def setup_adc(self):
        """Setup ADC ADS1115"""
        try:
            # Create I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Create ADS object
            self.ads = ADS.ADS1115(i2c)
            self.ads.gain = self.config["adc_gain"]
            
            # Create analog input channels
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
            raise
    
    def setup_baseline(self):
        """Setup baseline values from dataset"""
        try:
            # Load dataset to get baseline (normal) values
            df = pd.read_csv(self.config["dataset_file"])
            
            # Filter for normal/baseline readings
            baseline_data = df[df['label'].str.lower() == 'normal']
            
            if len(baseline_data) > 0:
                self.baseline_values = {
                    'TGS_2600': baseline_data['TGS 2600'].mean(),
                    'TGS_2602': baseline_data['TGS 2602'].mean(),
                    'TGS_2610': baseline_data['TGS 2610'].mean()
                }
                logger.info(f"Baseline values loaded: {self.baseline_values}")
            else:
                # If no normal data, use first few readings as baseline
                self.baseline_values = {
                    'TGS_2600': df['TGS 2600'].iloc[:10].mean(),
                    'TGS_2602': df['TGS 2602'].iloc[:10].mean(),
                    'TGS_2610': df['TGS 2610'].iloc[:10].mean()
                }
                logger.warning("No 'normal' labels found, using first 10 readings as baseline")
                
        except Exception as e:
            logger.error(f"Failed to setup baseline: {e}")
            # Default baseline values
            self.baseline_values = {'TGS_2600': 100, 'TGS_2602': 50, 'TGS_2610': 400}
    
    def load_model(self):
        """Load trained model and label encoder"""
        try:
            # Try to load existing model
            with open(self.config["model_file"], 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.config["label_encoder_file"], 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info("Pre-trained model loaded successfully")
            
        except FileNotFoundError:
            logger.warning("Pre-trained model not found, training new model from dataset")
            self.train_model()
    
    def train_model(self):
        """Train model from dataset"""
        try:
            # Load dataset
            df = pd.read_csv(self.config["dataset_file"])
            
            # Prepare features and labels
            feature_columns = ['TGS 2600', 'TGS 2602', 'TGS 2610']
            X = df[feature_columns].values
            y = df['label'].values
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y_encoded)
            
            # Save model and encoder
            with open(self.config["model_file"], 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.config["label_encoder_file"], 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            logger.info("Model trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise
    
    def read_sensors(self):
        """Read values from all gas sensors"""
        try:
            sensor_data = {}
            
            for i, (channel, sensor_name, conversion_factor) in enumerate(
                zip(self.channels, self.config["sensor_names"], self.config["voltage_to_ppm_factors"])
            ):
                # Read voltage
                voltage = channel.voltage
                
                # Convert voltage to PPM using conversion factor
                # This is a simplified conversion - adjust based on your sensor characteristics
                ppm = voltage * conversion_factor
                
                sensor_data[sensor_name] = {
                    'voltage': voltage,
                    'ppm': ppm,
                    'raw_value': channel.value
                }
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Failed to read sensors: {e}")
            return None
    
    def predict_gas(self, sensor_data):
        """Predict gas type from sensor readings"""
        try:
            # Prepare features for prediction
            features = [
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['ppm'],
                sensor_data['TGS_2610']['ppm']
            ]
            
            # Reshape for single prediction
            features = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction_encoded = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
            
            # Decode prediction
            predicted_gas = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = max(prediction_proba) * 100
            
            return predicted_gas, confidence
            
        except Exception as e:
            logger.error(f"Failed to predict gas: {e}")
            return "Unknown", 0.0
    
    def save_to_csv(self, timestamp, sensor_data, predicted_gas, confidence):
        """Save data to CSV file"""
        try:
            file_exists = False
            try:
                with open(self.config["csv_output"], 'r'):
                    file_exists = True
            except FileNotFoundError:
                pass
            
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
                    'TGS_2600_ppm': sensor_data['TGS_2600']['ppm'],
                    'TGS_2600_voltage': sensor_data['TGS_2600']['voltage'],
                    'TGS_2602_ppm': sensor_data['TGS_2602']['ppm'],
                    'TGS_2602_voltage': sensor_data['TGS_2602']['voltage'],
                    'TGS_2610_ppm': sensor_data['TGS_2610']['ppm'],
                    'TGS_2610_voltage': sensor_data['TGS_2610']['voltage'],
                    'predicted_gas': predicted_gas,
                    'confidence': confidence
                })
                
        except Exception as e:
            logger.error(f"Failed to save to CSV: {e}")
    
    def run_detection(self):
        """Main detection loop"""
        logger.info("Starting gas detection system...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Read sensor data
                sensor_data = self.read_sensors()
                
                if sensor_data:
                    # Predict gas type
                    predicted_gas, confidence = self.predict_gas(sensor_data)
                    
                    # Display results
                    print(f"\n{timestamp}")
                    print("-" * 50)
                    for sensor_name, data in sensor_data.items():
                        print(f"{sensor_name}: {data['ppm']:.2f} ppm ({data['voltage']:.3f}V)")
                    print(f"Predicted Gas: {predicted_gas} (Confidence: {confidence:.1f}%)")
                    
                    # Save to CSV
                    self.save_to_csv(timestamp, sensor_data, predicted_gas, confidence)
                
                # Wait before next reading
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
        # Create gas detection system
        gas_detector = GasDetectionSystem()
        
        # Run detection
        gas_detector.run_detection()
        
    except Exception as e:
        logger.error(f"Failed to start gas detection system: {e}")

if __name__ == "__main__":
    main()
