# gas_detection_system.py
# Tujuan Penelitian 1: Implementasi SBC dan sensor untuk deteksi profil gas dan membedakan berbagai gas

import smbus
import time
import csv
import pickle
import numpy as np
from datetime import datetime
import threading
import os

class ADS1115:
    """Driver untuk ADC ADS1115 tanpa library Adafruit"""
    
    def __init__(self, i2c_address=0x48):
        self.i2c_address = i2c_address
        self.bus = smbus.SMBus(1)  # I2C bus 1 pada Raspberry Pi
        
        # Register addresses
        self.REG_CONVERSION = 0x00
        self.REG_CONFIG = 0x01
        
        # Config register values
        self.CONFIG_OS_SINGLE = 0x8000  # Start single conversion
        self.CONFIG_MUX_DIFF_0_1 = 0x0000  # Differential P = AIN0, N = AIN1
        self.CONFIG_MUX_DIFF_0_3 = 0x1000  # Differential P = AIN0, N = AIN3
        self.CONFIG_MUX_DIFF_1_3 = 0x2000  # Differential P = AIN1, N = AIN3
        self.CONFIG_MUX_DIFF_2_3 = 0x3000  # Differential P = AIN2, N = AIN3
        self.CONFIG_MUX_SINGLE_0 = 0x4000  # Single-ended AIN0
        self.CONFIG_MUX_SINGLE_1 = 0x5000  # Single-ended AIN1
        self.CONFIG_MUX_SINGLE_2 = 0x6000  # Single-ended AIN2
        self.CONFIG_MUX_SINGLE_3 = 0x7000  # Single-ended AIN3
        
        self.CONFIG_PGA_6_144V = 0x0000  # +/-6.144V range
        self.CONFIG_PGA_4_096V = 0x0200  # +/-4.096V range
        self.CONFIG_PGA_2_048V = 0x0400  # +/-2.048V range (default)
        self.CONFIG_PGA_1_024V = 0x0600  # +/-1.024V range
        self.CONFIG_PGA_0_512V = 0x0800  # +/-0.512V range
        self.CONFIG_PGA_0_256V = 0x0A00  # +/-0.256V range
        
        self.CONFIG_MODE_CONTIN = 0x0000  # Continuous conversion mode
        self.CONFIG_MODE_SINGLE = 0x0100  # Single-shot mode (default)
        
        self.CONFIG_DR_128SPS = 0x0000   # 128 samples per second
        self.CONFIG_DR_250SPS = 0x0020   # 250 samples per second
        self.CONFIG_DR_490SPS = 0x0040   # 490 samples per second
        self.CONFIG_DR_920SPS = 0x0060   # 920 samples per second
        self.CONFIG_DR_1600SPS = 0x0080  # 1600 samples per second (default)
        self.CONFIG_DR_2400SPS = 0x00A0  # 2400 samples per second
        self.CONFIG_DR_3300SPS = 0x00C0  # 3300 samples per second
        
        self.CONFIG_CMODE_TRAD = 0x0000   # Traditional comparator (default)
        self.CONFIG_CMODE_WINDOW = 0x0010 # Window comparator
        
        self.CONFIG_CPOL_ACTVLOW = 0x0000 # ALERT/RDY pin is low when active (default)
        self.CONFIG_CPOL_ACTVHI = 0x0008  # ALERT/RDY pin is high when active
        
        self.CONFIG_CLAT_NONLAT = 0x0000  # Non-latching comparator (default)
        self.CONFIG_CLAT_LATCH = 0x0004   # Latching comparator
        
        self.CONFIG_CQUE_1CONV = 0x0000   # Assert after one conversion
        self.CONFIG_CQUE_2CONV = 0x0001   # Assert after two conversions
        self.CONFIG_CQUE_4CONV = 0x0002   # Assert after four conversions
        self.CONFIG_CQUE_NONE = 0x0003    # Disable comparator (default)
        
    def read_adc(self, channel=0, gain=1):
        """Membaca nilai ADC dari channel tertentu"""
        # Mapping channel ke config
        mux_config = {
            0: self.CONFIG_MUX_SINGLE_0,
            1: self.CONFIG_MUX_SINGLE_1,
            2: self.CONFIG_MUX_SINGLE_2,
            3: self.CONFIG_MUX_SINGLE_3
        }
        
        # Mapping gain ke config
        gain_config = {
            2/3: self.CONFIG_PGA_6_144V,
            1: self.CONFIG_PGA_4_096V,
            2: self.CONFIG_PGA_2_048V,
            4: self.CONFIG_PGA_1_024V,
            8: self.CONFIG_PGA_0_512V,
            16: self.CONFIG_PGA_0_256V
        }
        
        # Build config register
        config = (self.CONFIG_OS_SINGLE |
                 mux_config[channel] |
                 gain_config[gain] |
                 self.CONFIG_MODE_SINGLE |
                 self.CONFIG_DR_1600SPS |
                 self.CONFIG_CQUE_NONE)
        
        # Write config register
        config_bytes = [(config >> 8) & 0xFF, config & 0xFF]
        self.bus.write_i2c_block_data(self.i2c_address, self.REG_CONFIG, config_bytes)
        
        # Wait for conversion
        time.sleep(0.001)
        
        # Read conversion register
        data = self.bus.read_i2c_block_data(self.i2c_address, self.REG_CONVERSION, 2)
        
        # Convert to signed 16-bit value
        raw_value = (data[0] << 8) | data[1]
        if raw_value > 32767:
            raw_value -= 65536
            
        return raw_value
    
    def read_voltage(self, channel=0, gain=1):
        """Membaca voltage dari channel tertentu"""
        raw_value = self.read_adc(channel, gain)
        
        # Voltage range mapping
        voltage_range = {
            2/3: 6.144,
            1: 4.096,
            2: 2.048,
            4: 1.024,
            8: 0.512,
            16: 0.256
        }
        
        # Convert to voltage
        voltage = raw_value * voltage_range[gain] / 32767.0
        return voltage

class GasDetectionSystem:
    def __init__(self):
        self.adc = ADS1115()
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.is_running = False
        self.data_buffer = []
        
        # Load model dan preprocessing objects
        self.load_model_and_preprocessing()
        
    def load_model_and_preprocessing(self):
        """Load model machine learning dan preprocessing objects"""
        try:
            # Load trained model
            with open('gas_detection_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Model berhasil dimuat")
            
            # Load label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Label encoder berhasil dimuat")
            
            # Load scaler jika ada
            if os.path.exists('scaler.pkl'):
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Scaler berhasil dimuat")
            
        except FileNotFoundError as e:
            print(f"❌ Error: File tidak ditemukan - {e}")
            print("Pastikan file model, label encoder, dan scaler ada di direktori yang sama")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def read_sensors(self):
        """Membaca data dari 3 sensor gas TGS"""
        try:
            # Baca voltage dari 3 channel ADC (sensor TGS 2600, 2602, 2610)
            voltage_ch0 = self.adc.read_voltage(channel=0, gain=1)  # TGS 2600
            voltage_ch1 = self.adc.read_voltage(channel=1, gain=1)  # TGS 2602
            voltage_ch2 = self.adc.read_voltage(channel=2, gain=1)  # TGS 2610
            
            # Konversi voltage ke PPM (sesuaikan dengan kalibrasi Anda)
            # Formula ini harus disesuaikan dengan baseline yang sudah ditentukan saat pengambilan dataset
            ppm_sensor1 = self.voltage_to_ppm(voltage_ch0, sensor_type="TGS2600")
            ppm_sensor2 = self.voltage_to_ppm(voltage_ch1, sensor_type="TGS2602")
            ppm_sensor3 = self.voltage_to_ppm(voltage_ch2, sensor_type="TGS2610")
            
            return {
                'sensor1_ppm': ppm_sensor1,
                'sensor1_vrl': voltage_ch0,
                'sensor2_ppm': ppm_sensor2,
                'sensor2_vrl': voltage_ch1,
                'sensor3_ppm': ppm_sensor3,
                'sensor3_vrl': voltage_ch2,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"❌ Error membaca sensor: {e}")
            return None
    
    def voltage_to_ppm(self, voltage, sensor_type):
        """
        Konversi voltage ke PPM berdasarkan baseline yang sudah ditentukan
        Sesuaikan fungsi ini dengan karakteristik masing-masing sensor dan baseline Anda
        """
        # Konstanta kalibrasi (sesuaikan dengan data baseline Anda)
        if sensor_type == "TGS2600":
            # Contoh konversi untuk TGS2600 - sesuaikan dengan baseline Anda
            ppm = max(0, (voltage - 0.1) * 1000)  # Baseline 0.1V
        elif sensor_type == "TGS2602":
            # Contoh konversi untuk TGS2602 - sesuaikan dengan baseline Anda
            ppm = max(0, (voltage - 0.15) * 800)   # Baseline 0.15V
        elif sensor_type == "TGS2610":
            # Contoh konversi untuk TGS2610 - sesuaikan dengan baseline Anda
            ppm = max(0, (voltage - 0.12) * 900)   # Baseline 0.12V
        else:
            ppm = voltage * 100  # Default konversi
            
        return round(ppm, 2)
    
    def predict_gas_type(self, sensor_data):
        """Prediksi jenis gas berdasarkan data sensor"""
        if self.model is None or self.label_encoder is None:
            return "Model_Not_Loaded"
        
        try:
            # Siapkan feature vector [sensor1_ppm, sensor1_vrl, sensor2_ppm, sensor2_vrl, sensor3_ppm, sensor3_vrl]
            features = np.array([[
                sensor_data['sensor1_ppm'],
                sensor_data['sensor1_vrl'],
                sensor_data['sensor2_ppm'],
                sensor_data['sensor2_vrl'],
                sensor_data['sensor3_ppm'],
                sensor_data['sensor3_vrl']
            ]])
            
            # Scaling jika scaler tersedia
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Prediksi
            prediction = self.model.predict(features)[0]
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Prediksi probabilitas jika model mendukung
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = max(probabilities)
                return predicted_label, confidence
            
            return predicted_label, 1.0
            
        except Exception as e:
            print(f"❌ Error prediksi: {e}")
            return "Prediction_Error", 0.0
    
    def save_data_to_csv(self, sensor_data, predicted_gas, confidence, filename="gas_detection_log.csv"):
        """Simpan data ke CSV"""
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header jika file baru
            if not file_exists:
                writer.writerow([
                    "timestamp", "sensor1_ppm", "sensor1_vrl", 
                    "sensor2_ppm", "sensor2_vrl", "sensor3_ppm", 
                    "sensor3_vrl", "predicted_gas", "confidence"
                ])
            
            # Write data
            writer.writerow([
                sensor_data['timestamp'],
                sensor_data['sensor1_ppm'],
                sensor_data['sensor1_vrl'],
                sensor_data['sensor2_ppm'],
                sensor_data['sensor2_vrl'],
                sensor_data['sensor3_ppm'],
                sensor_data['sensor3_vrl'],
                predicted_gas,
                confidence
            ])
    
    def display_results(self, sensor_data, predicted_gas, confidence):
        """Tampilkan hasil deteksi"""
        print(f"\n{'='*60}")
        print(f"🕐 Timestamp: {sensor_data['timestamp']}")
        print(f"📊 Sensor Readings:")
        print(f"   TGS2600 - PPM: {sensor_data['sensor1_ppm']:6.2f} | Voltage: {sensor_data['sensor1_vrl']:.3f}V")
        print(f"   TGS2602 - PPM: {sensor_data['sensor2_ppm']:6.2f} | Voltage: {sensor_data['sensor2_vrl']:.3f}V")
        print(f"   TGS2610 - PPM: {sensor_data['sensor3_ppm']:6.2f} | Voltage: {sensor_data['sensor3_vrl']:.3f}V")
        print(f"🔍 Detected Gas: {predicted_gas}")
        print(f"📈 Confidence: {confidence:.2%}")
        print(f"{'='*60}")
    
    def run_detection(self, sampling_interval=2.0, save_to_csv=True):
        """Menjalankan sistem deteksi gas secara kontinyu"""
        print("🚀 Memulai sistem deteksi gas...")
        print("📍 Tekan Ctrl+C untuk menghentikan")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Baca data sensor
                sensor_data = self.read_sensors()
                
                if sensor_data:
                    # Prediksi jenis gas
                    result = self.predict_gas_type(sensor_data)
                    
                    if isinstance(result, tuple):
                        predicted_gas, confidence = result
                    else:
                        predicted_gas, confidence = result, 1.0
                    
                    # Tampilkan hasil
                    self.display_results(sensor_data, predicted_gas, confidence)
                    
                    # Simpan ke CSV jika diminta
                    if save_to_csv:
                        self.save_data_to_csv(sensor_data, predicted_gas, confidence)
                
                # Tunggu sebelum pembacaan berikutnya
                time.sleep(sampling_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Sistem dihentikan oleh user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.is_running = False
            print("✅ Sistem deteksi gas telah dihentikan")

def main():
    """Fungsi utama"""
    print("🌟 Gas Detection System for USV")
    print("🎯 Tujuan: Implementasi SBC untuk deteksi dan identifikasi gas")
    print("="*60)
    
    # Inisialisasi sistem
    gas_detector = GasDetectionSystem()
    
    # Cek apakah model berhasil dimuat
    if gas_detector.model is None:
        print("❌ Sistem tidak dapat dijalankan karena model tidak dimuat")
        return
    
    print("\n⚙️  Konfigurasi:")
    print(f"   - ADC: ADS1115 (I2C Address: 0x48)")
    print(f"   - Sensors: TGS2600, TGS2602, TGS2610")
    print(f"   - Sampling Interval: 2 detik")
    print(f"   - Output: Console + CSV file")
    
    # Jalankan deteksi
    gas_detector.run_detection(sampling_interval=2.0, save_to_csv=True)

if __name__ == "__main__":
    main()
