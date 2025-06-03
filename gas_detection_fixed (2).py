#!/usr/bin/env python3
"""
Gas Detection System for Raspberry Pi 4
Tugas Akhir: Implementasi Single Board Computer untuk Akuisisi Data dan Komputasi USV
Author: [Your Name]
"""

import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import logging
from pathlib import Path

# Import libraries untuk ADC dan Machine Learning
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except ImportError:
    print("Error: Adafruit libraries not found. Please install in virtual environment.")
    sys.exit(1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class GasDetectionSystem:
    def __init__(self, dataset_path="merged.csv", output_path="gas_detection_results.csv"):
        """
        Initialize Gas Detection System
        
        Args:
            dataset_path (str): Path to training dataset
            output_path (str): Path for output CSV file
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.model = None
        self.scaler = None
        
        # Circuit parameters for voltage to resistance conversion
        self.vc = 5.0  # Supply voltage (5V)
        self.rl = 10000  # Load resistance (10kΩ)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gas_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize ADC
        self.setup_adc()
        
        # Load and train model
        self.load_and_train_model()
        
        # Initialize CSV output file
        self.init_csv_output()
    
    def setup_adc(self):
        """Setup ADC ADS1115 connection"""
        try:
            # Create I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Create ADS1115 object
            self.ads = ADS.ADS1115(i2c)
            
            # Create analog input channels for 3 sensors
            # TGS 2600 -> A0, TGS 2602 -> A1, TGS 2610 -> A2
            self.chan0 = AnalogIn(self.ads, ADS.P0)  # TGS 2600
            self.chan1 = AnalogIn(self.ads, ADS.P1)  # TGS 2602
            self.chan2 = AnalogIn(self.ads, ADS.P2)  # TGS 2610
            
            self.logger.info("ADC ADS1115 initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ADC: {e}")
            sys.exit(1)
    
    def voltage_to_ppm(self, voltage, sensor_type):
        """
        Convert voltage to PPM based on sensor characteristics
        """
        # Calculate sensor resistance Rs from output voltage
        # Rs = (Vc/Vout - 1) × RL
        if voltage <= 0 or voltage >= self.vc:
            return 0  # Invalid voltage reading
        
        rs = ((self.vc / voltage) - 1) * self.rl
        
        if sensor_type == "TGS2600":
            # TGS2600 - Detection of Air Contaminants (H2, CO, Ethanol, etc.)
            # Target: Hydrogen detection (1-30 ppm range)
            # Rs/Ro characteristics from datasheet
            
            # Typical Rs in fresh air: 10kΩ - 90kΩ
            # Sensitivity: 0.3-0.6 for Rs(10ppm H2)/Rs(air)
            ro = 47000  # Typical sensor resistance in fresh air (47kΩ)
            rs_ro_ratio = rs / ro
            
            # Based on sensitivity curve for Hydrogen (most sensitive target gas)
            # Approximate inverse function from Rs/Ro vs ppm curve
            if rs_ro_ratio >= 1.0:
                ppm = 0  # Fresh air condition
            elif rs_ro_ratio >= 0.5:
                ppm = 1 + (1.0 - rs_ro_ratio) * 20  # 1-10 ppm range
            elif rs_ro_ratio >= 0.3:
                ppm = 10 + (0.5 - rs_ro_ratio) * 100  # 10-30 ppm range
            else:
                ppm = 30 + (0.3 - rs_ro_ratio) * 200  # Above 30 ppm
                
        elif sensor_type == "TGS2602":
            # TGS2602 - Detection of VOCs and Odorous Gases
            # Target: Ethanol, Ammonia, H2S, Toluene
            # Detection range: 1-30 ppm EtOH equivalent
            
            # Typical Rs in fresh air: 10kΩ - 100kΩ
            # Sensitivity: 0.08-0.5 for Rs(10ppm EtOH)/Rs(air)
            ro = 55000  # Typical sensor resistance in fresh air (55kΩ)
            rs_ro_ratio = rs / ro
            
            # Based on sensitivity curve for Ethanol
            if rs_ro_ratio >= 1.0:
                ppm = 0  # Fresh air condition
            elif rs_ro_ratio >= 0.3:
                ppm = 1 + (1.0 - rs_ro_ratio) * 14  # 1-10 ppm range
            elif rs_ro_ratio >= 0.08:
                ppm = 10 + (0.3 - rs_ro_ratio) * 91  # 10-30 ppm range
            else:
                ppm = 30 + (0.08 - rs_ro_ratio) * 250  # Above 30 ppm
                
        elif sensor_type == "TGS2610":
            # TGS2610 - Detection of LP Gas (Propane, Butane)
            # Detection range: 1-25% LEL (Lower Explosive Limit)
            # Reference gas: iso-butane at 1800 ppm
            
            # Rs in 1800ppm iso-butane: 1.0-10.0kΩ
            # Sensitivity: 0.45-0.62 for Rs(3000ppm)/Rs(1000ppm)
            ro = 1800  # Reference resistance in 1800ppm iso-butane
            rs_ro_ratio = rs / ro
            
            # Convert to % LEL (1% LEL ≈ 400-500 ppm for propane/butane)
            # Based on sensitivity curve for iso-butane/propane
            if rs_ro_ratio >= 10:
                ppm = 100  # Very low concentration
            elif rs_ro_ratio >= 1.0:
                # 100-1000 ppm range
                ppm = 100 + (10 - rs_ro_ratio) * 100
            elif rs_ro_ratio >= 0.1:
                # 1000-10000 ppm range  
                ppm = 1000 + (1.0 - rs_ro_ratio) * 10000
            else:
                # Above 10000 ppm
                ppm = 10000 + (0.1 - rs_ro_ratio) * 50000
                
            # Convert to % LEL (assuming propane LEL ≈ 21000 ppm)
            ppm_to_lel = ppm / 21000 * 100
            return max(0, min(25, ppm_to_lel))  # Limit to 0-25% LEL
            
        else:
            # Default conversion for unknown sensor types
            ro = 20000  # Default reference resistance
            rs_ro_ratio = rs / ro
            ppm = max(0, (1.0 - rs_ro_ratio) * 100)
        
        return max(0, ppm)  # Ensure non-negative values
    
    def read_sensors(self):
        """Read data from all three gas sensors"""
        try:
            # Read voltages
            voltage_tgs2600 = self.chan0.voltage
            voltage_tgs2602 = self.chan1.voltage
            voltage_tgs2610 = self.chan2.voltage
            
            # Convert to PPM
            ppm_tgs2600 = self.voltage_to_ppm(voltage_tgs2600, "TGS2600")
            ppm_tgs2602 = self.voltage_to_ppm(voltage_tgs2602, "TGS2602")
            ppm_tgs2610 = self.voltage_to_ppm(voltage_tgs2610, "TGS2610")
            
            return {
                'TGS2600': ppm_tgs2600,
                'TGS2602': ppm_tgs2602,
                'TGS2610': ppm_tgs2610,
                'voltages': {
                    'TGS2600': voltage_tgs2600,
                    'TGS2602': voltage_tgs2602,
                    'TGS2610': voltage_tgs2610
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error reading sensors: {e}")
            return None
    
    def load_and_train_model(self):
        """Load dataset and train machine learning model"""
        try:
            # Load dataset
            if not os.path.exists(self.dataset_path):
                self.logger.error(f"Dataset file {self.dataset_path} not found!")
                sys.exit(1)
            
            # Read CSV with proper handling of semicolon separator and quotes
            df = pd.read_csv(self.dataset_path, sep=';', quotechar='"', skipinitialspace=True)
            
            # Clean column names - remove extra spaces and quotes
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            # Print available columns for debugging
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Check if we have the expected columns with spaces
            expected_columns = ['TGS 2600', 'TGS 2602', 'TGS 2610']
            
            # Verify all sensor columns exist
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing columns: {missing_columns}")
                self.logger.error(f"Available columns: {df.columns.tolist()}")
                sys.exit(1)
            
            self.logger.info(f"Using sensor columns: {expected_columns}")
            
            # Prepare features - convert to numeric and handle any non-numeric values
            X = df[expected_columns].copy()
            
            # Convert to numeric, replacing any non-numeric values with NaN
            for col in expected_columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Fill NaN values with column means
            X = X.fillna(X.mean())
            
            # Convert to numpy array
            X = X.values
            
            # Handle labels
            if 'label' not in df.columns:
                self.logger.error(f"Label column not found. Available: {df.columns.tolist()}")
                sys.exit(1)
                
            y = df['label'].str.strip().values
            
            # Remove any rows with NaN values
            valid_indices = ~np.isnan(X).any(axis=1)
            X = X[valid_indices]
            y = y[valid_indices]
            
            self.logger.info(f"Dataset shape after cleaning: {X.shape}")
            self.logger.info(f"Classes found: {np.unique(y)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model trained successfully!")
            self.logger.info(f"Model accuracy: {accuracy:.3f}")
            self.logger.info(f"Training data shape: {X_train.shape}")
            
            # Save model and scaler
            joblib.dump(self.model, 'gas_detection_model.pkl')
            joblib.dump(self.scaler, 'gas_detection_scaler.pkl')
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)
    
    def predict_gas(self, sensor_data):
        """Predict gas type from sensor readings"""
        try:
            if self.model is None or self.scaler is None:
                return "Unknown", 0.0
            
            # Prepare input data
            features = np.array([[
                sensor_data['TGS2600'],
                sensor_data['TGS2602'],
                sensor_data['TGS2610']
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = np.max(self.model.predict_proba(features_scaled))
            
            return prediction, probability
            
        except Exception as e:
            self.logger.error(f"Error predicting gas: {e}")
            return "Error", 0.0
    
    def init_csv_output(self):
        """Initialize CSV output file with headers"""
        headers = [
            'timestamp',
            'TGS2600_ppm',
            'TGS2602_ppm', 
            'TGS2610_ppm',
            'TGS2600_voltage',
            'TGS2602_voltage',
            'TGS2610_voltage',
            'predicted_gas',
            'confidence'
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        
        self.logger.info(f"CSV output file initialized: {self.output_path}")
    
    def save_data(self, timestamp, sensor_data, prediction, confidence):
        """Save data to CSV file"""
        try:
            row = [
                timestamp,
                sensor_data['TGS2600'],
                sensor_data['TGS2602'],
                sensor_data['TGS2610'],
                sensor_data['voltages']['TGS2600'],
                sensor_data['voltages']['TGS2602'],
                sensor_data['voltages']['TGS2610'],
                prediction,
                confidence
            ]
            
            with open(self.output_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
                
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def run(self, sampling_interval=1.0):
        """Main execution loop"""
        self.logger.info("Starting gas detection system...")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Read sensor data
                sensor_data = self.read_sensors()
                
                if sensor_data is not None:
                    # Predict gas type
                    prediction, confidence = self.predict_gas(sensor_data)
                    
                    # Display results
                    print(f"\n{timestamp}")
                    print(f"TGS2600: {sensor_data['TGS2600']:.2f} ppm ({sensor_data['voltages']['TGS2600']:.3f}V)")
                    print(f"TGS2602: {sensor_data['TGS2602']:.2f} ppm ({sensor_data['voltages']['TGS2602']:.3f}V)")
                    print(f"TGS2610: {sensor_data['TGS2610']:.2f} ppm ({sensor_data['voltages']['TGS2610']:.3f}V)")
                    print(f"Predicted Gas: {prediction} (Confidence: {confidence:.3f})")
                    print("-" * 50)
                    
                    # Save to CSV
                    self.save_data(timestamp, sensor_data, prediction, confidence)
                
                # Wait for next sampling
                time.sleep(sampling_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Gas detection system stopped by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.logger.info(f"Data saved to: {self.output_path}")

def main():
    """Main function"""
    print("=" * 60)
    print("Gas Detection System for USV Air Pollution Mitigation")
    print("Tugas Akhir - Single Board Computer Implementation")
    print("=" * 60)
    
    # Initialize and run the system
    try:
        detector = GasDetectionSystem(
            dataset_path="merged.csv",
            output_path="gas_detection_results.csv"
        )
        
        # Run with 1 second sampling interval
        detector.run(sampling_interval=1.0)
        
    except Exception as e:
        print(f"Failed to start system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()