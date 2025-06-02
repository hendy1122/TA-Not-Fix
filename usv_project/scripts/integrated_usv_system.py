import time
import threading
import queue
from datetime import datetime
import json
import os

# Import dari kedua sistem sebelumnya
from gas_detection_system import GasDetectionSystem
from gps_mapping_system import GPSMappingSystem

class IntegratedUSVSystem:
    def __init__(self, config_file="usv_config.json"):
        # Load konfigurasi
        self.config = self.load_config(config_file)
        
        # Initialize komponen sistem
        self.gas_detector = GasDetectionSystem(
            model_path=self.config["gas_detection"]["model_path"],
            label_encoder_path=self.config["gas_detection"]["label_encoder_path"],
            csv_filename=self.config["gas_detection"]["csv_filename"]
        )
        
        self.gps_mapper = GPSMappingSystem(
            gps_port=self.config["gps"]["port"],
            baud_rate=self.config["gps"]["baud_rate"],
            csv_filename=self.config["gps"]["csv_filename"]
        )
        
        # Queue untuk komunikasi antar thread
        self.data_queue = queue.Queue()
        self.running = False
        
        # Thread untuk masing-masing sistem
        self.gas_thread = None
        self.gps_thread = None
        self.integration_thread = None
        
    def load_config(self, config_file):
        """Load konfigurasi dari file JSON"""
        default_config = {
            "gas_detection": {
                "model_path": "gas_detection_model.pkl",
                "label_encoder_path": "label_encoder.pkl",
                "csv_filename": "usv_gas_data.csv",
                "sampling_interval": 2
            },
            "gps": {
                "port": "/dev/ttyAMA0",
                "baud_rate": 9600,
                "csv_filename": "usv_gps_data.csv",
                "sampling_interval": 5
            },
            "integration": {
                "data_fusion_interval": 10,
                "alert_threshold": 0.8,
                "high_pollution_threshold": 300
            },
            "usv": {
                "mission_duration": 3600,  # 1 jam
                "auto_return_on_high_pollution": True,
                "max_pollution_exposure_time": 300  # 5 menit
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge dengan default config
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}, using default config")
        else:
            # Buat file config default
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Default config created: {config_file}")
        
        return default_config
    
    def gas_detection_worker(self):
        """Worker thread untuk deteksi gas"""
        print("Gas detection thread started")
        last_reading_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - last_reading_time >= self.config["gas_detection"]["sampling_interval"]:
                    # Baca sensor gas
                    sensor_data = self.gas_detector.read_sensors()
                    
                    if sensor_data:
                        # Prediksi gas
                        predicted_gas, confidence = self.gas_detector.predict_gas(sensor_data)
                        
                        # Hitung konsentrasi rata-rata
                        avg_concentration = (
                            sensor_data["TGS2600"]["ppm"] +
                            sensor_data["TGS2602"]["ppm"] +
                            sensor_data["TGS2610"]["ppm"]
                        ) / 3
                        
                        # Kirim data ke queue
                        gas_data = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "gas_detection",
                            "sensor_data": sensor_data,
                            "predicted_gas": predicted_gas,
                            "confidence": confidence,
                            "avg_concentration": avg_concentration
                        }
                        
                        self.data_queue.put(gas_data)
                        last_reading_time = current_time
                
                time.sleep(0.1)  # Small delay untuk CPU
                
            except Exception as e:
                print(f"Error in gas detection worker: {e}")
                time.sleep(1)
    
    def gps_mapping_worker(self):
        """Worker thread untuk GPS mapping"""
        print("GPS mapping thread started")
        last_reading_time = 0
        gps_position = {}
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - last_reading_time >= self.config["gps"]["sampling_interval"]:
                    # Baca data GPS
                    gps_data = self.gps_mapper.read_gps_data()
                    
                    if gps_data:
                        if "latitude" in gps_data:
                            gps_position.update(gps_data)
                        else:
                            gps_position.update(gps_data)
                    
                    # Jika ada posisi GPS yang valid
                    if "latitude" in gps_position and "longitude" in gps_position:
                        gps_msg = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "gps_data",
                            "position": gps_position.copy()
                        }
                        
                        self.data_queue.put(gps_msg)
                        last_reading_time = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in GPS mapping worker: {e}")
                time.sleep(1)
    
    def data_integration_worker(self):
        """Worker thread untuk integrasi dan analisis data"""
        print("Data integration thread started")
        
        latest_gas_data = None
        latest_gps_data = None
        high_pollution_start_time = None
        
        while self.running:
            try:
                # Ambil data dari queue
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    
                    if data["type"] == "gas_detection":
                        latest_gas_data = data
                        
                        # Simpan data gas
                        self.gas_detector.save_data(
                            data["sensor_data"],
                            data["predicted_gas"],
                            data["confidence"]
                        )
                        
                    elif data["type"] == "gps_data":
                        latest_gps_data = data
                
                # Jika ada data gas dan GPS yang valid
                if latest_gas_data and latest_gps_data:
                    # Gabungkan data untuk mapping
                    gas_info = {
                        "gas_type": latest_gas_data["predicted_gas"],
                        "concentration": latest_gas_data["avg_concentration"],
                        "confidence": latest_gas_data["confidence"]
                    }
                    
                    self.gps_mapper.save_location_data(
                        latest_gps_data["position"],
                        gas_info
                    )
                    
                    # Tampilkan status terintegrasi
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n=== USV Status [{current_time}] ===")
                    print(f"Position: {latest_gps_data['position']['latitude']:.6f}, {latest_gps_data['position']['longitude']:.6f}")
                    print(f"Gas: {latest_gas_data['predicted_gas']} ({latest_gas_data['avg_concentration']:.2f} ppm)")
                    print(f"Confidence: {latest_gas_data['confidence']:.2%}")
                    
                    pollution_level = self.gps_mapper.get_pollution_level(latest_gas_data['avg_concentration'])
                    print(f"Pollution Level: {pollution_level}")
                    
                    # Alert system
                    if (latest_gas_data['confidence'] > self.config["integration"]["alert_threshold"] and 
                        latest_gas_data['avg_concentration'] > self.config["integration"]["high_pollution_threshold"]):
                        
                        if high_pollution_start_time is None:
                            high_pollution_start_time = time.time()
                            print("⚠️  HIGH POLLUTION ALERT! Monitoring exposure time...")
                        
                        exposure_time = time.time() - high_pollution_start_time
                        print(f"⚠️  High pollution exposure: {exposure_time:.0f}s")
                        
                        # Auto return jika exposure terlalu lama
                        if (self.config["usv"]["auto_return_on_high_pollution"] and 
                            exposure_time > self.config["usv"]["max_pollution_exposure_time"]):
                            print("🚨 CRITICAL: Maximum pollution exposure reached! Initiating return protocol...")
                            # Di sini bisa ditambahkan logika untuk mengarahkan USV kembali
                    
                    else:
                        high_pollution_start_time = None
                    
                    # Reset data untuk menghindari duplikasi
                    latest_gas_data = None
                    latest_gps_data = None
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in data integration worker: {e}")
                time.sleep(1)
    
    def start_mission(self, duration=None):
        """Memulai misi USV terintegrasi"""
        print("🚢 === USV POLLUTION MONITORING MISSION STARTED ===")
        print("Press Ctrl+C to stop the mission")
        
        if duration is None:
            duration = self.config["usv"]["mission_duration"]
        
        self.running = True
        
        # Start worker threads
        self.gas_thread = threading.Thread(target=self.gas_detection_worker)
        self.gps_thread = threading.Thread(target=self.gps_mapping_worker)
        self.integration_thread = threading.Thread(target=self.data_integration_worker)
        
        self.gas_thread.daemon = True
        self.gps_thread.daemon = True
        self.integration_thread.daemon = True
        
        self.gas_thread.start()
        self.gps_thread.start()
        self.integration_thread.start()
        
        try:
            # Main mission loop
            start_time = time.time()
            
            while self.running:
                elapsed_time = time.time() - start_time
                
                if elapsed_time >= duration:
                    print(f"\n🎯 Mission completed after {duration}s")
                    break
                
                # Status update setiap 30 detik
                if int(elapsed_time) % 30 == 0:
                    remaining_time = duration - elapsed_time
                    print(f"\n⏱️  Mission time remaining: {remaining_time:.0f}s")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Mission interrupted by user")
        
        finally:
            self.stop_mission()
    
    def stop_mission(self):
        """Menghentikan misi dan membuat laporan"""
        print("\n🔄 Stopping USV mission...")
        self.running = False
        
        # Wait for threads to finish
        if self.gas_thread and self.gas_thread.is_alive():
            self.gas_thread.join(timeout=5)
        if self.gps_thread and self.gps_thread.is_alive():
            self.gps_thread.join(timeout=5)
        if self.integration_thread and self.integration_thread.is_alive():
            self.integration_thread.join(timeout=5)
        
        # Generate mission report
        self.generate_mission_report()
        
        print("🏁 USV mission completed successfully!")
    
    def generate_mission_report(self):
        """Generate laporan misi lengkap"""
        try:
            # Buat peta pencemaran
            self.gps_mapper.create_pollution_map("usv_mission_map.html")
            
            # Analisis area pencemaran
            analysis = self.gps_mapper.analyze_pollution_areas()
            
            # Buat laporan text
            report_filename = f"usv_mission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_filename, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("USV POLLUTION MONITORING MISSION REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("MISSION SUMMARY:\n")
                f.write(f"- Total measurement points: {analysis['total_points']}\n")
                f.write(f"- High pollution points: {analysis['high_pollution_points']}\n")
                f.write(f"- Medium pollution points: {analysis['medium_pollution_points']}\n")
                f.write(f"- High pollution percentage: {analysis['high_pollution_percentage']:.1f}%\n")
                f.write(f"- Pollution hotspots identified: {len(analysis['hotspots'])}\n\n")
                
                if analysis['hotspots']:
                    f.write("POLLUTION HOTSPOTS:\n")
                    for i, hotspot in enumerate(analysis['hotspots'], 1):
                        f.write(f"Hotspot {i}:\n")
                        f.write(f"  - Center coordinates: {hotspot['center'][0]:.6f}, {hotspot['center'][1]:.6f}\n")
                        f.write(f"  - Number of high-pollution points: {hotspot['nearby_points']}\n")
                        f.write(f"  - Average concentration: {hotspot['avg_concentration']:.2f} ppm\n\n")
                
                f.write("FILES GENERATED:\n")
                f.write(f"- Gas detection data: {self.config['gas_detection']['csv_filename']}\n")
                f.write(f"- GPS mapping data: {self.config['gps']['csv_filename']}\n")
                f.write("- Interactive map: usv_mission_map.html\n")
                f.write(f"- Mission report: {report_filename}\n")
            
            print(f"📊 Mission report generated: {report_filename}")
            print("🗺️  Interactive pollution map: usv_mission_map.html")
            
        except Exception as e:
            print(f"Error generating mission report: {e}")

def main():
    # Buat instance sistem USV terintegrasi
    usv_system = IntegratedUSVSystem("usv_config.json")
    
    # Mulai misi (durasi dalam detik)
    usv_system.start_mission(duration=600)  # 10 menit untuk testing
    
if __name__ == "__main__":
    main()
                