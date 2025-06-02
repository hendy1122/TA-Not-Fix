import time
import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from datetime import datetime
import csv
import json
import os
import re
from geopy.distance import geodesic

class GPSMappingSystem:
    def __init__(self, gps_port="/dev/ttyAMA0", baud_rate=9600, csv_filename="gps_pollution_data.csv"):
        # GPS Serial connection
        try:
            self.gps_serial = serial.Serial(gps_port, baud_rate, timeout=1)
            print(f"GPS Serial connection established on {gps_port}")
        except Exception as e:
            print(f"Error connecting to GPS: {e}")
            self.gps_serial = None
        
        self.csv_filename = csv_filename
        self.setup_csv_file()
        
        # Data storage untuk tracking
        self.location_data = []
        self.pollution_threshold = {
            "low": 100,
            "medium": 300,
            "high": 500
        }
    
    def setup_csv_file(self):
        """Setup CSV file dengan header jika belum ada"""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "latitude", "longitude", "altitude",
                    "speed_kmh", "course", "gas_type", "gas_concentration",
                    "pollution_level", "confidence"
                ])
    
    def parse_nmea_gga(self, sentence):
        """Parse NMEA GGA sentence untuk mendapatkan koordinat"""
        try:
            parts = sentence.split(',')
            if len(parts) < 15 or parts[0] != '$GPGGA':
                return None
            
            # Time
            time_str = parts[1]
            
            # Latitude
            lat_raw = parts[2]
            lat_dir = parts[3]
            if lat_raw and lat_dir:
                lat_deg = float(lat_raw[:2])
                lat_min = float(lat_raw[2:])
                latitude = lat_deg + lat_min/60
                if lat_dir == 'S':
                    latitude = -latitude
            else:
                return None
            
            # Longitude
            lon_raw = parts[4]
            lon_dir = parts[5]
            if lon_raw and lon_dir:
                lon_deg = float(lon_raw[:3])
                lon_min = float(lon_raw[3:])
                longitude = lon_deg + lon_min/60
                if lon_dir == 'W':
                    longitude = -longitude
            else:
                return None
            
            # Altitude
            altitude = float(parts[9]) if parts[9] else 0.0
            
            return {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "time": time_str
            }
        except Exception as e:
            print(f"Error parsing NMEA: {e}")
            return None
    
    def parse_nmea_rmc(self, sentence):
        """Parse NMEA RMC sentence untuk mendapatkan kecepatan dan arah"""
        try:
            parts = sentence.split(',')
            if len(parts) < 12 or parts[0] != '$GPRMC':
                return None
            
            # Speed in knots
            speed_knots = float(parts[7]) if parts[7] else 0.0
            speed_kmh = speed_knots * 1.852  # Convert to km/h
            
            # Course
            course = float(parts[8]) if parts[8] else 0.0
            
            return {
                "speed_kmh": speed_kmh,
                "course": course
            }
        except Exception as e:
            print(f"Error parsing RMC: {e}")
            return None
    
    def read_gps_data(self):
        """Baca data GPS dari serial port"""
        if not self.gps_serial:
            return None
        
        try:
            line = self.gps_serial.readline().decode('ascii', errors='replace').strip()
            
            if line.startswith('$GPGGA'):
                return self.parse_nmea_gga(line)
            elif line.startswith('$GPRMC'):
                return self.parse_nmea_rmc(line)
            
            return None
        except Exception as e:
            print(f"Error reading GPS: {e}")
            return None
    
    def get_pollution_level(self, gas_concentration):
        """Tentukan level pencemaran berdasarkan konsentrasi gas"""
        if gas_concentration >= self.pollution_threshold["high"]:
            return "High"
        elif gas_concentration >= self.pollution_threshold["medium"]:
            return "Medium"
        elif gas_concentration >= self.pollution_threshold["low"]:
            return "Low"
        else:
            return "Clean"
    
    def save_location_data(self, gps_data, gas_data=None):
        """Simpan data lokasi dan polusi ke CSV"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Default values jika tidak ada data gas
            gas_type = gas_data.get("gas_type", "No_Gas") if gas_data else "No_Gas"
            gas_concentration = gas_data.get("concentration", 0) if gas_data else 0
            confidence = gas_data.get("confidence", 0) if gas_data else 0
            
            pollution_level = self.get_pollution_level(gas_concentration)
            
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    gps_data.get("latitude", 0),
                    gps_data.get("longitude", 0),
                    gps_data.get("altitude", 0),
                    gps_data.get("speed_kmh", 0),
                    gps_data.get("course", 0),
                    gas_type,
                    gas_concentration,
                    pollution_level,
                    confidence
                ])
            
            # Simpan ke memory untuk mapping
            self.location_data.append({
                "timestamp": timestamp,
                "lat": gps_data.get("latitude", 0),
                "lon": gps_data.get("longitude", 0),
                "pollution_level": pollution_level,
                "gas_type": gas_type,
                "concentration": gas_concentration
            })
            
        except Exception as e:
            print(f"Error saving location data: {e}")
    
    def create_pollution_map(self, output_filename="pollution_map.html"):
        """Buat peta interaktif menunjukkan area pencemaran"""
        if not self.location_data:
            print("Tidak ada data lokasi untuk dipetakan")
            return
        
        # Hitung center map
        lats = [point["lat"] for point in self.location_data if point["lat"] != 0]
        lons = [point["lon"] for point in self.location_data if point["lon"] != 0]
        
        if not lats or not lons:
            print("Tidak ada koordinat GPS yang valid")
            return
        
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Buat peta
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        
        # Color mapping untuk level pencemaran
        color_map = {
            "Clean": "green",
            "Low": "yellow", 
            "Medium": "orange",
            "High": "red"
        }
        
        # Tambahkan marker untuk setiap titik
        for point in self.location_data:
            if point["lat"] != 0 and point["lon"] != 0:
                color = color_map.get(point["pollution_level"], "blue")
                
                popup_text = f"""
                <b>Waktu:</b> {point["timestamp"]}<br>
                <b>Gas:</b> {point["gas_type"]}<br>
                <b>Konsentrasi:</b> {point["concentration"]:.2f} ppm<br>
                <b>Level Pencemaran:</b> {point["pollution_level"]}
                """
                
                folium.CircleMarker(
                    location=[point["lat"], point["lon"]],
                    radius=8,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Tambahkan legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Level Pencemaran</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Clean</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Low</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
        <p><i class="fa fa-circle" style="color:red"></i> High</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Simpan peta
        m.save(output_filename)
        print(f"Peta pencemaran disimpan sebagai {output_filename}")
    
    def analyze_pollution_areas(self):
        """Analisis area dengan tingkat pencemaran tinggi"""
        if not self.location_data:
            print("Tidak ada data untuk dianalisis")
            return {}
        
        # Filter data berdasarkan level pencemaran
        high_pollution = [p for p in self.location_data if p["pollution_level"] == "High"]
        medium_pollution = [p for p in self.location_data if p["pollution_level"] == "Medium"]
        
        analysis = {
            "total_points": len(self.location_data),
            "high_pollution_points": len(high_pollution),
            "medium_pollution_points": len(medium_pollution),
            "high_pollution_percentage": (len(high_pollution) / len(self.location_data)) * 100,
            "hotspots": []
        }
        
        # Identifikasi hotspot (area dengan pencemaran tinggi yang berdekatan)
        for point in high_pollution:
            nearby_high = []
            for other_point in high_pollution:
                if point != other_point:
                    distance = geodesic(
                        (point["lat"], point["lon"]), 
                        (other_point["lat"], other_point["lon"])
                    ).meters
                    if distance < 100:  # Dalam radius 100 meter
                        nearby_high.append(other_point)
            
            if len(nearby_high) >= 2:  # Minimal 3 titik berdekatan
                analysis["hotspots"].append({
                    "center": (point["lat"], point["lon"]),
                    "nearby_points": len(nearby_high) + 1,
                    "avg_concentration": np.mean([point["concentration"]] + 
                                               [p["concentration"] for p in nearby_high])
                })
        
        return analysis
    
    def run_mapping(self, interval=5, duration=None, gas_detector=None):
        """
        Jalankan sistem pemetaan GPS
        interval: waktu delay antar pembacaan (detik)
        duration: durasi total (detik), None untuk unlimited
        gas_detector: instance dari GasDetectionSystem untuk integrasi
        """
        print("=== Sistem Pemetaan GPS Dimulai ===")
        print("Tekan Ctrl+C untuk menghentikan")
        
        start_time = time.time()
        gps_position = {}
        
        try:
            while True:
                # Baca data GPS
                gps_data = self.read_gps_data()
                
                if gps_data:
                    if "latitude" in gps_data:
                        gps_position.update(gps_data)
                    else:
                        gps_position.update(gps_data)
                
                # Jika ada posisi GPS yang valid
                if "latitude" in gps_position and "longitude" in gps_position:
                    # Baca data gas jika detector tersedia
                    gas_data = None
                    if gas_detector:
                        sensor_reading = gas_detector.read_sensors()
                        if sensor_reading:
                            predicted_gas, confidence = gas_detector.predict_gas(sensor_reading)
                            # Hitung konsentrasi total
                            total_concentration = (
                                sensor_reading["TGS2600"]["ppm"] +
                                sensor_reading["TGS2602"]["ppm"] +
                                sensor_reading["TGS2610"]["ppm"]
                            ) / 3
                            
                            gas_data = {
                                "gas_type": predicted_gas,
                                "concentration": total_concentration,
                                "confidence": confidence
                            }
                    
                    # Simpan data
                    self.save_location_data(gps_position, gas_data)
                    
                    # Tampilkan status
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{current_time}]")
                    print(f"GPS: {gps_position['latitude']:.6f}, {gps_position['longitude']:.6f}")
                    if gas_data:
                        print(f"Gas: {gas_data['gas_type']} ({gas_data['concentration']:.2f} ppm)")
                        print(f"Level Pencemaran: {self.get_pollution_level(gas_data['concentration'])}")
                
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n=== Sistem Pemetaan Dihentikan ===")
        except Exception as e:
            print(f"Error in mapping loop: {e}")
        
        # Buat peta dan analisis setelah selesai
        if self.location_data:
            self.create_pollution_map()
            analysis = self.analyze_pollution_areas()
            print("\n=== Hasil Analisis ===")
            print(f"Total titik pengukuran: {analysis['total_points']}")
            print(f"Titik pencemaran tinggi: {analysis['high_pollution_points']}")
            print(f"Persentase pencemaran tinggi: {analysis['high_pollution_percentage']:.1f}%")
            print(f"Hotspot teridentifikasi: {len(analysis['hotspots'])}")

def main():
    # Konfigurasi GPS
    GPS_PORT = "/dev/ttyAMA0"  # Port GPS default Raspberry Pi
    BAUD_RATE = 9600
    CSV_FILENAME = "gps_pollution_mapping.csv"
    
    # Buat instance sistem pemetaan
    mapper = GPSMappingSystem(GPS_PORT, BAUD_RATE, CSV_FILENAME)
    
    # Jalankan pemetaan
    mapper.run_mapping(interval=5, duration=600)  # 10 menit
    
if __name__ == "__main__":
    main()