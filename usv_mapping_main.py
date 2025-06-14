#!/usr/bin/env python3
"""
USV Gas Mapping System - Main Implementation
Integrates gas detection from main.py with GPS coordinates from Pixhawk
Creates real-time contamination maps
"""

import time
import json
import threading
import queue
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# MAVLink imports
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobalRelative

# Import gas detection from your existing main.py
try:
    from main import *  # Import all your existing gas detection functions
    print("✅ Successfully imported gas detection system from main.py")
except ImportError as e:
    print(f"❌ Error importing main.py: {e}")
    print("Make sure main.py is in the same directory")
    exit(1)

@dataclass
class GPSCoordinate:
    lat: float
    lon: float
    alt: float
    timestamp: datetime
    hdg: Optional[float] = None

@dataclass 
class GasReading:
    gas_type: str
    concentration: float
    confidence: float
    sensor_array: Dict[str, float]  # TGS2600, TGS2602, TGS2610 readings
    timestamp: datetime

@dataclass
class ContaminationPoint:
    coordinate: GPSCoordinate
    gas_reading: GasReading
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

class USVGasMapper:
    def __init__(self, config_file='config/mapping_config.json'):
        """Initialize USV Gas Mapping System"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Data storage
        self.contamination_points: List[ContaminationPoint] = []
        self.gps_data_queue = queue.Queue()
        self.gas_data_queue = queue.Queue()
        
        # Connection objects
        self.mavlink_connection = None
        self.vehicle = None
        
        # Threading control
        self.running = False
        self.threads = []
        
        # Gas detection thresholds for mapping (based on your main.py gas types)
        self.mapping_thresholds = {
            'normal': {'LOW': 1, 'MEDIUM': 3, 'HIGH': 5, 'CRITICAL': 10},        # Clean air baseline
            'alcohol': {'LOW': 10, 'MEDIUM': 25, 'HIGH': 50, 'CRITICAL': 100},   # Alcohol detection
            'pertalite': {'LOW': 8, 'MEDIUM': 20, 'HIGH': 40, 'CRITICAL': 80},   # Pertalite fuel  
            'pertamax': {'LOW': 8, 'MEDIUM': 20, 'HIGH': 40, 'CRITICAL': 80},    # Pertamax fuel
            'dexlite': {'LOW': 6, 'MEDIUM': 15, 'HIGH': 30, 'CRITICAL': 60},     # Dexlite fuel
            'biosolar': {'LOW': 7, 'MEDIUM': 18, 'HIGH': 35, 'CRITICAL': 70},    # Biosolar fuel
            'hydrogen': {'LOW': 5, 'MEDIUM': 12, 'HIGH': 25, 'CRITICAL': 50},    # Hydrogen gas
            'toluene': {'LOW': 5, 'MEDIUM': 15, 'HIGH': 30, 'CRITICAL': 60},     # Toluene
            'ammonia': {'LOW': 5, 'MEDIUM': 15, 'HIGH': 30, 'CRITICAL': 60},     # Ammonia
            'butane': {'LOW': 10, 'MEDIUM': 25, 'HIGH': 50, 'CRITICAL': 100},    # Butane
            'propane': {'LOW': 10, 'MEDIUM': 25, 'HIGH': 50, 'CRITICAL': 100},   # Propane
            'unknown': {'LOW': 5, 'MEDIUM': 15, 'HIGH': 30, 'CRITICAL': 60},     # Unknown gas
            'error': {'LOW': 999, 'MEDIUM': 999, 'HIGH': 999, 'CRITICAL': 999}  # Error state
        }
        
        # Initialize your existing gas detection system
        self.init_gas_detection_system()
        
        print("🚢 USV Gas Mapping System initialized")
    
    def load_config(self, config_file):
        """Load configuration from file"""
        default_config = {
            'mavlink_connection': 'udp:0.0.0.0:14550',
            'update_interval': 2.0,  # seconds
            'gps_timeout': 10.0,
            'gas_sample_rate': 1.0,
            'map_update_interval': 5.0,
            'log_level': 'INFO',
            'output_dir': 'maps',
            'data_dir': 'data'
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            return default_config
    
    def setup_logging(self):
        """Setup logging system"""
        log_level = getattr(logging, self.config['log_level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/usv_mapping_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def init_gas_detection_system(self):
        """Initialize gas detection system using your EnhancedDatasheetGasSensorArray"""
        try:
            # Import your main.py class
            from main import EnhancedDatasheetGasSensorArray
            
            # Initialize your gas sensor array
            self.gas_detector = EnhancedDatasheetGasSensorArray()
            
            # Load existing calibration if available
            if not self.gas_detector.load_calibration():
                print("⚠️  Warning: No calibration found. Sensors may need calibration.")
                print("   Run calibration from main.py first, or continue with simplified readings.")
            
            # Load existing model if available
            if not self.gas_detector.load_model():
                print("⚠️  Warning: No trained model found. Gas classification will be basic.")
                print("   Train model from main.py first for better gas identification.")
            
            # Set to datasheet mode for accurate mapping
            self.gas_detector.set_sensor_mode('datasheet')
            
            print("✅ Gas detection system initialized with your EnhancedDatasheetGasSensorArray")
            self.logger.info("Gas detection system initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize gas detection: {e}")
            self.logger.error(f"Gas detection initialization failed: {e}")
            raise
    
    def connect_mavlink(self, connection_string=None):
        """Connect to Pixhawk via MAVLink"""
        if connection_string is None:
            connection_string = self.config['mavlink_connection']
        
        try:
            print(f"🔌 Connecting to MAVLink: {connection_string}")
            self.logger.info(f"Connecting to MAVLink: {connection_string}")
            
            # Try pymavlink first
            self.mavlink_connection = mavutil.mavlink_connection(
                connection_string,
                baud=57600,
                timeout=self.config['gps_timeout']
            )
            
            # Wait for heartbeat
            heartbeat = self.mavlink_connection.wait_heartbeat(timeout=15)
            if heartbeat:
                print(f"✅ MAVLink connected - System ID: {heartbeat.get_srcSystem()}")
                self.logger.info(f"MAVLink heartbeat received from system {heartbeat.get_srcSystem()}")
                
                # Request GPS data stream
                self.mavlink_connection.mav.request_data_stream_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                    1,  # 1 Hz
                    1   # Start streaming
                )
                
                return True
            else:
                print("❌ No heartbeat received")
                return False
                
        except Exception as e:
            print(f"❌ MAVLink connection failed: {e}")
            self.logger.error(f"MAVLink connection failed: {e}")
            return False
    
    def gps_data_thread(self):
        """Thread to continuously read GPS data from Pixhawk"""
        self.logger.info("GPS data thread started")
        
        while self.running:
            try:
                if self.mavlink_connection:
                    msg = self.mavlink_connection.recv_match(
                        type='GLOBAL_POSITION_INT', 
                        blocking=False
                    )
                    
                    if msg:
                        gps_coord = GPSCoordinate(
                            lat=msg.lat / 1e7,
                            lon=msg.lon / 1e7,
                            alt=msg.alt / 1000.0,
                            hdg=msg.hdg / 100.0 if msg.hdg != 65535 else None,
                            timestamp=datetime.now()
                        )
                        
                        # Put in queue (keep only latest)
                        if not self.gps_data_queue.empty():
                            try:
                                self.gps_data_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.gps_data_queue.put(gps_coord)
                        
                        # Log every 10th reading to avoid spam
                        if hasattr(self, '_gps_log_counter'):
                            self._gps_log_counter += 1
                        else:
                            self._gps_log_counter = 1
                            
                        if self._gps_log_counter % 10 == 0:
                            self.logger.debug(f"GPS: {gps_coord.lat:.6f}, {gps_coord.lon:.6f}")
                
                time.sleep(0.5)  # 2 Hz GPS reading
                
            except Exception as e:
                self.logger.error(f"GPS thread error: {e}")
                time.sleep(1)
        
        self.logger.info("GPS data thread stopped")
    
    def gas_detection_thread(self):
        """Thread to continuously read gas sensor data"""
        self.logger.info("Gas detection thread started")
        
        while self.running:
            try:
                # Get gas reading using your existing main.py functions
                # Adapt this call based on your actual main.py structure
                gas_data = self.get_current_gas_reading()
                
                if gas_data:
                    gas_reading = GasReading(
                        gas_type=gas_data['gas_type'],
                        concentration=gas_data['concentration'],
                        confidence=gas_data['confidence'],
                        sensor_array=gas_data.get('sensor_readings', {}),
                        timestamp=datetime.now()
                    )
                    
                    # Put in queue (keep only latest)
                    if not self.gas_data_queue.empty():
                        try:
                            self.gas_data_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.gas_data_queue.put(gas_reading)
                    
                    # Log significant detections
                    if gas_reading.concentration > 5:  # Threshold for logging
                        self.logger.info(f"Gas detected: {gas_reading.gas_type} "
                                       f"at {gas_reading.concentration:.1f} "
                                       f"(confidence: {gas_reading.confidence:.2f})")
                
                time.sleep(self.config['gas_sample_rate'])
                
            except Exception as e:
                self.logger.error(f"Gas detection thread error: {e}")
                time.sleep(1)
        
        self.logger.info("Gas detection thread stopped")
    
    def get_current_gas_reading(self):
        """Get current gas reading using your EnhancedDatasheetGasSensorArray"""
        try:
            # Use your actual gas sensor system
            sensor_readings = self.gas_detector.read_sensors()
            predicted_gas, confidence = self.gas_detector.predict_gas(sensor_readings)
            
            # Calculate average concentration from all sensors for mapping threshold
            concentrations = []
            sensor_data = {}
            
            for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
                if sensor_name in sensor_readings:
                    ppm = sensor_readings[sensor_name]['ppm']
                    concentrations.append(ppm)
                    sensor_data[sensor_name] = ppm
                else:
                    sensor_data[sensor_name] = 0.0
            
            # Use maximum concentration for mapping decision
            max_concentration = max(concentrations) if concentrations else 0.0
            
            # Format data for mapping system
            gas_data = {
                'gas_type': predicted_gas if confidence > 0.5 else 'unknown',
                'concentration': max_concentration,
                'confidence': confidence,
                'sensor_readings': sensor_data,
                'raw_sensor_data': sensor_readings  # Keep full sensor data for debugging
            }
            
            return gas_data
            
        except Exception as e:
            self.logger.error(f"Error getting gas reading: {e}")
            # Return safe fallback data
            return {
                'gas_type': 'error',
                'concentration': 0.0,
                'confidence': 0.0,
                'sensor_readings': {'TGS2600': 0.0, 'TGS2602': 0.0, 'TGS2610': 0.0}
            }
    
    def data_fusion_thread(self):
        """Thread to combine GPS and gas data"""
        self.logger.info("Data fusion thread started")
        
        while self.running:
            try:
                # Get latest GPS and gas data
                gps_coord = None
                gas_reading = None
                
                if not self.gps_data_queue.empty():
                    gps_coord = self.gps_data_queue.get_nowait()
                
                if not self.gas_data_queue.empty():
                    gas_reading = self.gas_data_queue.get_nowait()
                
                # Combine data if both available and recent
                if gps_coord and gas_reading:
                    time_diff = abs((gps_coord.timestamp - gas_reading.timestamp).total_seconds())
                    
                    if time_diff <= 5.0:  # Within 5 seconds
                        # Determine severity level
                        severity = self.determine_severity(gas_reading)
                        
                        # Only map if significant detection
                        if severity != 'NONE':
                            contamination_point = ContaminationPoint(
                                coordinate=gps_coord,
                                gas_reading=gas_reading,
                                severity=severity
                            )
                            
                            self.contamination_points.append(contamination_point)
                            
                            self.logger.info(f"Contamination point added: {gas_reading.gas_type} "
                                           f"({severity}) at {gps_coord.lat:.6f}, {gps_coord.lon:.6f}")
                            
                            # Save data point
                            self.save_contamination_point(contamination_point)
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Data fusion thread error: {e}")
                time.sleep(1)
        
        self.logger.info("Data fusion thread stopped")
    
    def determine_severity(self, gas_reading: GasReading) -> str:
        """Determine contamination severity level"""
        gas_type = gas_reading.gas_type
        concentration = gas_reading.concentration
        
        if gas_type not in self.mapping_thresholds:
            return 'NONE'
        
        thresholds = self.mapping_thresholds[gas_type]
        
        if concentration >= thresholds['CRITICAL']:
            return 'CRITICAL'
        elif concentration >= thresholds['HIGH']:
            return 'HIGH'
        elif concentration >= thresholds['MEDIUM']:
            return 'MEDIUM'
        elif concentration >= thresholds['LOW']:
            return 'LOW'
        else:
            return 'NONE'
    
    def save_contamination_point(self, point: ContaminationPoint):
        """Save contamination point to file"""
        try:
            data = {
                'timestamp': point.coordinate.timestamp.isoformat(),
                'lat': point.coordinate.lat,
                'lon': point.coordinate.lon,
                'alt': point.coordinate.alt,
                'hdg': point.coordinate.hdg,
                'gas_type': point.gas_reading.gas_type,
                'concentration': point.gas_reading.concentration,
                'confidence': point.gas_reading.confidence,
                'severity': point.severity,
                'sensor_array': point.gas_reading.sensor_array
            }
            
            # Append to daily data file
            filename = f"data/contamination_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(filename, 'a') as f:
                f.write(json.dumps(data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error saving contamination point: {e}")
    
    def generate_realtime_map(self):
        """Generate real-time contamination map"""
        if not self.contamination_points:
            self.logger.info("No contamination points to map")
            return None
        
        try:
            # Get center point for map
            center_lat = np.mean([p.coordinate.lat for p in self.contamination_points])
            center_lon = np.mean([p.coordinate.lon for p in self.contamination_points])
            
            # Create folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Color mapping for severity
            severity_colors = {
                'LOW': 'yellow',
                'MEDIUM': 'orange', 
                'HIGH': 'red',
                'CRITICAL': 'darkred'
            }
            
            # Add contamination points
            for point in self.contamination_points:
                color = severity_colors.get(point.severity, 'gray')
                
                # Create popup text
                popup_text = f"""
                <b>Gas Type:</b> {point.gas_reading.gas_type}<br>
                <b>Concentration:</b> {point.gas_reading.concentration:.2f}<br>
                <b>Severity:</b> {point.severity}<br>
                <b>Confidence:</b> {point.gas_reading.confidence:.2f}<br>
                <b>Time:</b> {point.coordinate.timestamp.strftime('%H:%M:%S')}<br>
                <b>Coordinates:</b> {point.coordinate.lat:.6f}, {point.coordinate.lon:.6f}
                """
                
                folium.CircleMarker(
                    location=[point.coordinate.lat, point.coordinate.lon],
                    radius=10 + (2 * ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(point.severity)),
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
            
            # Add legend
            legend_html = """
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Contamination Level</b></p>
            <p><span style="color:yellow;">●</span> Low</p>
            <p><span style="color:orange;">●</span> Medium</p>
            <p><span style="color:red;">●</span> High</p>
            <p><span style="color:darkred;">●</span> Critical</p>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save map
            map_filename = f"maps/contamination_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            m.save(map_filename)
            
            self.logger.info(f"Map saved: {map_filename}")
            return map_filename
            
        except Exception as e:
            self.logger.error(f"Error generating map: {e}")
            return None
    
    def generate_3d_visualization(self):
        """Generate 3D visualization of contamination data"""
        if not self.contamination_points:
            return None
        
        try:
            # Prepare data
            lats = [p.coordinate.lat for p in self.contamination_points]
            lons = [p.coordinate.lon for p in self.contamination_points]
            concentrations = [p.gas_reading.concentration for p in self.contamination_points]
            gas_types = [p.gas_reading.gas_type for p in self.contamination_points]
            severities = [p.severity for p in self.contamination_points]
            
            # Create 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=lons,
                y=lats,
                z=concentrations,
                mode='markers',
                marker=dict(
                    size=8,
                    color=concentrations,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Concentration")
                ),
                text=[f"Gas: {g}<br>Severity: {s}<br>Conc: {c:.2f}" 
                      for g, s, c in zip(gas_types, severities, concentrations)],
                hovertemplate='<b>%{text}</b><br>Lat: %{y:.6f}<br>Lon: %{x:.6f}<extra></extra>'
            )])
            
            fig.update_layout(
                title='USV Gas Contamination 3D Map',
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude', 
                    zaxis_title='Concentration'
                )
            )
            
            # Save 3D plot
            plot_filename = f"maps/contamination_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_filename)
            
            self.logger.info(f"3D visualization saved: {plot_filename}")
            return plot_filename
            
        except Exception as e:
            self.logger.error(f"Error generating 3D visualization: {e}")
            return None
    
    def generate_summary_report(self):
        """Generate summary report of contamination mapping"""
        if not self.contamination_points:
            return None
        
        try:
            # Statistics
            total_points = len(self.contamination_points)
            gas_types = {}
            severity_counts = {}
            
            for point in self.contamination_points:
                # Count by gas type
                gas_type = point.gas_reading.gas_type
                if gas_type not in gas_types:
                    gas_types[gas_type] = {'count': 0, 'avg_concentration': 0, 'max_concentration': 0}
                
                gas_types[gas_type]['count'] += 1
                gas_types[gas_type]['max_concentration'] = max(
                    gas_types[gas_type]['max_concentration'],
                    point.gas_reading.concentration
                )
                
                # Count by severity
                severity = point.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Calculate averages
            for gas_type in gas_types:
                concentrations = [p.gas_reading.concentration for p in self.contamination_points 
                                if p.gas_reading.gas_type == gas_type]
                gas_types[gas_type]['avg_concentration'] = np.mean(concentrations)
            
            # Create report
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_contamination_points': total_points,
                    'mapping_duration': None,  # Calculate if needed
                    'area_covered': None       # Calculate if needed
                },
                'gas_detection_summary': gas_types,
                'severity_distribution': severity_counts,
                'high_risk_areas': [
                    {
                        'lat': p.coordinate.lat,
                        'lon': p.coordinate.lon,
                        'gas_type': p.gas_reading.gas_type,
                        'concentration': p.gas_reading.concentration,
                        'severity': p.severity
                    }
                    for p in self.contamination_points 
                    if p.severity in ['HIGH', 'CRITICAL']
                ]
            }
            
            # Save report
            report_filename = f"data/summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Summary report saved: {report_filename}")
            return report_filename
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return None
    
    def start_mapping(self):
        """Start the USV gas mapping system"""
        print("🚀 Starting USV Gas Mapping System...")
        
        # Connect to MAVLink
        if not self.connect_mavlink():
            print("❌ Failed to connect to MAVLink. Exiting.")
            return False
        
        # Start threads
        self.running = True
        
        # GPS data thread
        gps_thread = threading.Thread(target=self.gps_data_thread, daemon=True)
        gps_thread.start()
        self.threads.append(gps_thread)
        
        # Gas detection thread
        gas_thread = threading.Thread(target=self.gas_detection_thread, daemon=True)
        gas_thread.start()
        self.threads.append(gas_thread)
        
        # Data fusion thread
        fusion_thread = threading.Thread(target=self.data_fusion_thread, daemon=True)
        fusion_thread.start()
        self.threads.append(fusion_thread)
        
        print("✅ All threads started successfully")
        self.logger.info("USV Gas Mapping System started")
        
        return True
    
    def stop_mapping(self):
        """Stop the USV gas mapping system"""
        print("🛑 Stopping USV Gas Mapping System...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Close connections
        if self.mavlink_connection:
            self.mavlink_connection.close()
        
        print("✅ System stopped")
        self.logger.info("USV Gas Mapping System stopped")
    
    def interactive_mode(self):
        """Run system in interactive mode"""
        print("\n🎮 USV Gas Mapping - Interactive Mode")
        print("=" * 50)
        
        try:
            while self.running:
                print(f"\nContamination points collected: {len(self.contamination_points)}")
                print("\nOptions:")
                print("1. Generate real-time map")
                print("2. Generate 3D visualization") 
                print("3. Generate summary report")
                print("4. View latest readings")
                print("5. Stop mapping")
                
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    map_file = self.generate_realtime_map()
                    if map_file:
                        print(f"✅ Map generated: {map_file}")
                
                elif choice == '2':
                    plot_file = self.generate_3d_visualization()
                    if plot_file:
                        print(f"✅ 3D visualization generated: {plot_file}")
                
                elif choice == '3':
                    report_file = self.generate_summary_report()
                    if report_file:
                        print(f"✅ Summary report generated: {report_file}")
                
                elif choice == '4':
                    self.show_latest_readings()
                
                elif choice == '5':
                    break
                
                else:
                    print("Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            self.stop_mapping()
    
    def show_latest_readings(self):
        """Show latest GPS and gas readings"""
        print("\n📊 Latest Readings:")
        print("-" * 30)
        
        # Latest GPS
        if not self.gps_data_queue.empty():
            gps = self.gps_data_queue.queue[-1]
            print(f"GPS: {gps.lat:.6f}, {gps.lon:.6f} @ {gps.timestamp.strftime('%H:%M:%S')}")
        else:
            print("GPS: No data")
        
        # Latest Gas
        if not self.gas_data_queue.empty():
            gas = self.gas_data_queue.queue[-1]
            print(f"Gas: {gas.gas_type} - {gas.concentration:.2f} (conf: {gas.confidence:.2f}) @ {gas.timestamp.strftime('%H:%M:%S')}")
        else:
            print("Gas: No data")
        
        # Recent contamination points
        if self.contamination_points:
            recent = self.contamination_points[-5:]  # Last 5 points
            print(f"\nRecent contamination points ({len(recent)}):")
            for i, point in enumerate(recent, 1):
                print(f"{i}. {point.gas_reading.gas_type} ({point.severity}) @ "
                      f"{point.coordinate.lat:.6f}, {point.coordinate.lon:.6f}")

def main():
    """Main function"""
    print("🚢 USV Gas Mapping System v1.0")
    print("=" * 50)
    
    # Create directories
    import os
    os.makedirs('logs', exist_ok=True)
    os.makedirs('maps', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Initialize system
    mapper = USVGasMapper()
    
    # Start mapping
    if mapper.start_mapping():
        print("\n🎯 System ready! Starting interactive mode...")
        print("Press Ctrl+C to stop at any time")
        time.sleep(2)
        
        # Run interactive mode
        mapper.interactive_mode()
    else:
        print("❌ Failed to start mapping system")

if __name__ == "__main__":
    main()
