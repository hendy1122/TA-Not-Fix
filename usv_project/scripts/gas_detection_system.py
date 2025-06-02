import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import csv
import json
import os

class GasDetectionSystem:
    def __init__(self, model_path, label_encoder_path, csv_filename="gas_detection_data.csv"):
        # Initialize I2C and ADS1115
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c)
        
        # Setup analog channels for 3 sensors
        self.chan0 = AnalogIn(self.ads, ADS.P0)  # TGS 2600
        self.chan1 = AnalogIn(self.ads, ADS.P1)  # TGS 2602
        self.chan2 = AnalogIn(self.ads, ADS.P2)  # TGS 2610
        
        # Load machine learning model and label encoder
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            print("Model dan label encoder berhasil dimuat!")
        except Exception as e:
            print(f"Error loading model atau label encoder: {e}")
            self.model = None
            self.label_encoder = None
        
        self.csv_filename = csv_filename
        self.setup_csv_file()
        
        # Voltage reference (5V for ADS1115)
        self.vref = 5.0
        
    def setup_csv_file(self):
        """Setup CSV file dengan header jika belum ada"""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", 
                    "TGS2600_ppm", "TGS2600_voltage",
                    "TGS2602_ppm", "TGS2602_voltage", 
                    "TGS2610_ppm", "TGS2610_voltage",
                    "predicted_gas", "confidence"
                ])
    
    def voltage_to_ppm(self, voltage, sensor_type):
        """
        Konversi voltage ke PPM berdasarkan karakteristik sensor
        Sesuaikan formula ini dengan kalibrasi yang sudah dilakukan di Arduino
        """
        if sensor_type == "TGS2600":
            # Formula untuk TGS2600 (sesuaikan dengan kalibrasi Anda)
            # Contoh: ppm = (voltage / vref) * scale_factor
            ppm = (voltage / self.vref) * 1000  # Sesuaikan scale factor
        elif sensor_type == "TGS2602":
            # Formula untuk TGS2602
            ppm = (voltage / self.vref) * 1000  # Sesuaikan scale factor
        elif sensor_type == "TGS2610":
            # Formula untuk TGS2610
            ppm = (voltage / self.vref) * 1000  # Sesuaikan scale factor
        else:
            ppm = 0
        
        return max(0, ppm)  # Pastikan tidak negatif
    
    def read_sensors(self):
        """Baca data dari ketiga sensor"""
        try:
            # Baca voltage dari setiap channel
            voltage_0 = self.chan0.voltage
            voltage_1 = self.chan1.voltage
            voltage_2 = self.chan2.voltage
            
            # Konversi ke PPM
            ppm_2600 = self.voltage_to_ppm(voltage_0, "TGS2600")
            ppm_2602 = self.voltage_to_ppm(voltage_1, "TGS2602")
            ppm_2610 = self.voltage_to_ppm(voltage_2, "TGS2610")
            
            return {
                "TGS2600": {"ppm": ppm_2600, "voltage": voltage_0},
                "TGS2602": {"ppm": ppm_2602, "voltage": voltage_1},
                "TGS2610": {"ppm": ppm_2610, "voltage": voltage_2}
            }
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return None
    
    def predict_gas(self, sensor_data):
        """Prediksi jenis gas menggunakan model ML"""
        if self.model is None or self.label_encoder is None:
            return "Unknown", 0.0
        
        try:
            # Buat array fitur dari data sensor
            # Sesuaikan urutan fitur dengan yang digunakan saat training
            features = np.array([[
                sensor_data["TGS2600"]["ppm"],
                sensor_data["TGS2602"]["ppm"],
                sensor_data["TGS2610"]["ppm"]
            ]])
            
            # Prediksi
            prediction = self.model.predict(features)
            prediction_proba = self.model.predict_proba(features)
            
            # Decode label
            gas_type = self.label_encoder.inverse_transform(prediction)[0]
            confidence = np.max(prediction_proba)
            
            return gas_type, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0
    
    def save_data(self, sensor_data, predicted_gas, confidence):
        """Simpan data ke CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    sensor_data["TGS2600"]["ppm"], sensor_data["TGS2600"]["voltage"],
                    sensor_data["TGS2602"]["ppm"], sensor_data["TGS2602"]["voltage"],
                    sensor_data["TGS2610"]["ppm"], sensor_data["TGS2610"]["voltage"],
                    predicted_gas, confidence
                ])
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def run_detection(self, interval=2, duration=None):
        """
        Jalankan sistem deteksi gas
        interval: waktu delay antar pembacaan (detik)
        duration: durasi total pengukuran (detik), None untuk unlimited
        """
        print("=== Sistem Deteksi Gas Dimulai ===")
        print("Tekan Ctrl+C untuk menghentikan")
        
        start_time = time.time()
        
        try:
            while True:
                # Baca sensor
                sensor_data = self.read_sensors()
                
                if sensor_data:
                    # Prediksi gas
                    predicted_gas, confidence = self.predict_gas(sensor_data)
                    
                    # Tampilkan hasil
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{current_time}]")
                    print(f"TGS2600: {sensor_data['TGS2600']['ppm']:.2f} ppm ({sensor_data['TGS2600']['voltage']:.3f}V)")
                    print(f"TGS2602: {sensor_data['TGS2602']['ppm']:.2f} ppm ({sensor_data['TGS2602']['voltage']:.3f}V)")
                    print(f"TGS2610: {sensor_data['TGS2610']['ppm']:.2f} ppm ({sensor_data['TGS2610']['voltage']:.3f}V)")
                    print(f"Prediksi: {predicted_gas} (Confidence: {confidence:.2%})")
                    
                    # Simpan data
                    self.save_data(sensor_data, predicted_gas, confidence)
                    
                    # Alert jika confidence tinggi dan gas terdeteksi
                    if confidence > 0.8 and predicted_gas != "No_Gas":
                        print(f"⚠️  ALERT: Gas {predicted_gas} terdeteksi dengan confidence tinggi!")
                
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n=== Sistem Deteksi Dihentikan ===")
        except Exception as e:
            print(f"Error in detection loop: {e}")

def main():
    # Konfigurasi path file
    MODEL_PATH = "gas_detection_model.pkl"  # Sesuaikan path model Anda
    LABEL_ENCODER_PATH = "label_encoder.pkl"  # Sesuaikan path label encoder Anda
    CSV_FILENAME = "gas_detection_results.csv"
    
    # Buat instance sistem deteksi
    detector = GasDetectionSystem(MODEL_PATH, LABEL_ENCODER_PATH, CSV_FILENAME)
    
    # Jalankan deteksi
    # detector.run_detection(interval=2)  # Unlimited duration
    detector.run_detection(interval=2, duration=300)  # 5 menit
    
if __name__ == "__main__":
    main()