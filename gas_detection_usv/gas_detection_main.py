#!/usr/bin/env python3
"""
Gas Detection System for USV Air Pollution Monitoring
Raspberry Pi 4 + ADS1115 ADC + TGS Gas Sensors Array
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
import Adafruit_ADS1x15
import warnings
warnings.filterwarnings('ignore')

class GasDetectionSystem:
    def __init__(self):
        """Initialize the gas detection system"""
        # Initialize ADC
        self.adc = Adafruit_ADS1x15.ADS1115()
        
        # ADC Configuration
        self.GAIN = 1  # +/-4.096V range
        self.SAMPLES_PER_SECOND = 128
        
        # Sensor channels on ADS1115
        self.TGS_2600_CHANNEL = 0  # A0
        self.TGS_2602_CHANNEL = 1  # A1  
        self.TGS_2610_CHANNEL = 2  # A2
        
        # Voltage reference (3.3V for Raspberry Pi)
        self.VREF = 3.3
        
        # Load machine learning model and data
        self.model = None
        self.label_encoder = None
        self.training_data = None
        self.scaler = None
        
        # Data storage
        self.csv_filename = f"gas_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print("Gas Detection System Initialized")
        print(f"Data will be saved to: {self.csv_filename}")
        
    def load_model_and_data(self):
        """Load ML model, label encoder, and training data"""
        try:
            # Load training data (merged.csv)
            self.training_data = pd.read_csv('merged.csv')
            print(f"Training data loaded: {len(self.training_data)} samples")
            print(f"Gas types in dataset: {self.training_data['label'].unique()}")
            
            # Create and train a simple model from the data
            self.create_model_from_data()
            
        except Exception as e:
            print(f"Error loading model/data: {e}")
            print("Creating default classification model...")
            self.create_default_model()
    
    def create_model_from_data(self):
        """Create and train model from the training data"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = self.training_data[['TGS 2600', 'TGS 2602', 'TGS 2610']].values
        y = self.training_data['label'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_encoded)
        
        print("Model trained successfully!")
        print(f"Classes: {self.label_encoder.classes_}")
        
    def create_default_model(self):
        """Create a default rule-based classification model"""
        from sklearn.preprocessing import LabelEncoder
        
        # Create default label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['normal', 'alkohol', 'pertalite', 'pertamax'])
        
        print("Default rule-based model created")
        
    def read_sensor_voltage(self, channel):
        """Read voltage from specific ADC channel"""
        try:
            # Read ADC value
            adc_value = self.adc.read_adc(channel, gain=self.GAIN, data_rate=self.SAMPLES_PER_SECOND)
            
            # Convert to voltage (ADS1115 is 16-bit)
            voltage = (adc_value * 4.096) / 32767.0  # 4.096V range for gain=1
            
            return max(0, voltage)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error reading sensor {channel}: {e}")
            return 0.0
    
    def voltage_to_ppm(self, voltage, sensor_type):
        """Convert voltage to PPM based on sensor characteristics"""
        # These are approximation formulas - adjust based on your sensor datasheets
        # and calibration data from your Arduino experiments
        
 # Calculate sensor resistance Rs from output voltage
    # Rs = (Vc/Vout - 1) × RL
    if voltage <= 0 or voltage >= vc:
        return 0  # Invalid voltage reading
    
    rs = ((vc / voltage) - 1) * rl
    
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

    
    def read_all_sensors(self):
        """Read all gas sensors and return PPM values"""
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
    
    def predict_gas_type(self, sensor_data):
        """Predict gas type based on sensor readings"""
        try:
            if not sensor_data:
                return "unknown", 0.0
                
            # Prepare input features
            features = np.array([[
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['ppm'], 
                sensor_data['TGS_2610']['ppm']
            ]])
            
            if self.model is not None and self.scaler is not None:
                # Use trained ML model
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
                gas_type = self.label_encoder.inverse_transform([prediction])[0]
                
            else:
                # Use rule-based classification
                gas_type, confidence = self.rule_based_classification(features[0])
            
            return gas_type, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "unknown", 0.0
    
    def rule_based_classification(self, features):
        """Simple rule-based gas classification"""
        tgs2600, tgs2602, tgs2610 = features
        
        # Define thresholds based on your training data analysis
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
    
    def save_to_csv(self, timestamp, sensor_data, predicted_gas, confidence):
        """Save data to CSV file"""
        try:
            # Prepare row data
            row = [
                timestamp,
                sensor_data['TGS_2600']['voltage'],
                sensor_data['TGS_2600']['ppm'],
                sensor_data['TGS_2602']['voltage'], 
                sensor_data['TGS_2602']['ppm'],
                sensor_data['TGS_2610']['voltage'],
                sensor_data['TGS_2610']['ppm'],
                predicted_gas,
                confidence
            ]
            
            # Check if file exists to write header
            import os
            write_header = not os.path.exists(self.csv_filename)
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                
                if write_header:
                    header = [
                        'timestamp',
                        'TGS2600_voltage', 'TGS2600_ppm',
                        'TGS2602_voltage', 'TGS2602_ppm', 
                        'TGS2610_voltage', 'TGS2610_ppm',
                        'predicted_gas', 'confidence'
                    ]
                    writer.writerow(header)
                
                writer.writerow(row)
                
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def run_detection(self, interval=2.0):
        """Main detection loop"""
        print("\n=== Starting Gas Detection System ===")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Read sensors
                sensor_data = self.read_all_sensors()
                
                if sensor_data:
                    # Predict gas type
                    predicted_gas, confidence = self.predict_gas_type(sensor_data)
                    
                    # Display results
                    print(f"\n[{timestamp}]")
                    print(f"TGS2600: {sensor_data['TGS_2600']['ppm']:.2f} ppm ({sensor_data['TGS_2600']['voltage']:.3f}V)")
                    print(f"TGS2602: {sensor_data['TGS_2602']['ppm']:.2f} ppm ({sensor_data['TGS_2602']['voltage']:.3f}V)")
                    print(f"TGS2610: {sensor_data['TGS_2610']['ppm']:.2f} ppm ({sensor_data['TGS_2610']['voltage']:.3f}V)")
                    print(f"Predicted Gas: {predicted_gas.upper()} (Confidence: {confidence:.2f})")
                    
                    # Save to CSV
                    self.save_to_csv(timestamp, sensor_data, predicted_gas, confidence)
                
                else:
                    print(f"[{timestamp}] Error reading sensors")
                
                # Wait for next reading
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nSystem stopped by user")
            print(f"Data saved to: {self.csv_filename}")
        except Exception as e:
            print(f"\nError in main loop: {e}")
        finally:
            print("Gas detection system shutdown complete")

def main():
    """Main function"""
    print("=== USV Gas Detection System ===")
    print("Raspberry Pi 4 + ADS1115 + TGS Sensor Array")
    print("-" * 50)
    
    # Initialize system
    detector = GasDetectionSystem()
    
    # Load model and training data
    detector.load_model_and_data()
    
    # Start detection
    detector.run_detection(interval=2.0)  # Read every 2 seconds

if __name__ == "__main__":
    main()