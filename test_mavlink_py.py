#!/usr/bin/env python3
"""
Test MAVLink Connection Script untuk USV Gas Mapping System
Test koneksi UDP MAVLink dengan Pixhawk via MAVProxy bridge
"""

import time
import sys
from datetime import datetime

def test_mavlink_import():
    """Test import MAVLink library"""
    print("="*50)
    print("Testing MAVLink Library Import...")
    print("="*50)
    
    try:
        from pymavlink import mavutil
        from pymavlink.dialects.v20 import ardupilotmega as mavlink2
        print("✅ MAVLink library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ MAVLink import failed: {e}")
        print("Install with: pip install pymavlink")
        return False

def test_connection_types():
    """Test different connection types"""
    print("\n" + "="*50)
    print("Available Connection Types...")
    print("="*50)
    
    connection_types = {
        'UDP MAVProxy Bridge': 'udp:0.0.0.0:14550',
        'TCP MAVProxy': 'tcp:127.0.0.1:5760',
        'Serial USB': '/dev/ttyUSB0',
        'Serial ACM': '/dev/ttyACM0'
    }
    
    print("Supported connection strings:")
    for name, conn_str in connection_types.items():
        print(f"  {name}: {conn_str}")
    
    return connection_types

def test_udp_connection(connection_string='udp:0.0.0.0:14550', timeout=10):
    """Test UDP MAVLink connection"""
    print(f"\n" + "="*50)
    print(f"Testing UDP MAVLink Connection...")
    print(f"Connection: {connection_string}")
    print(f"Timeout: {timeout} seconds")
    print("="*50)
    
    try:
        from pymavlink import mavutil
        
        print("Creating MAVLink connection...")
        connection = mavutil.mavlink_connection(
            connection_string,
            source_system=254,  # Different from Mission Planner
            source_component=190
        )
        
        print("Waiting for heartbeat...")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            msg = connection.recv_match(type='HEARTBEAT', blocking=False, timeout=1)
            if msg:
                print(f"✅ Heartbeat received!")
                print(f"  System ID: {msg.get_srcSystem()}")
                print(f"  Component ID: {msg.get_srcComponent()}")
                print(f"  Vehicle Type: {msg.type}")
                print(f"  Autopilot: {msg.autopilot}")
                print(f"  Base Mode: {msg.base_mode}")
                print(f"  System Status: {msg.system_status}")
                
                connection.close()
                return True
            
            remaining = timeout - (time.time() - start_time)
            print(f"  Waiting... {remaining:.1f}s remaining", end='\r')
            time.sleep(0.5)
        
        print(f"\n❌ No heartbeat received within {timeout} seconds")
        connection.close()
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_gps_messages(connection_string='udp:0.0.0.0:14550', duration=15):
    """Test GPS message reception"""
    print(f"\n" + "="*50)
    print(f"Testing GPS Message Reception...")
    print(f"Duration: {duration} seconds")
    print("="*50)
    
    try:
        from pymavlink import mavutil
        
        print("Connecting to MAVLink...")
        connection = mavutil.mavlink_connection(
            connection_string,
            source_system=254,
            source_component=190
        )
        
        print("Waiting for heartbeat...")
        msg = connection.wait_heartbeat(timeout=10)
        if not msg:
            print("❌ No heartbeat - aborting GPS test")
            return False
        
        print("✅ Connected! Testing GPS messages...")
        
        # Request GPS data stream
        try:
            connection.mav.request_data_stream_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                2,  # 2 Hz
                1   # Start
            )
            print("GPS data stream requested")
        except Exception as e:
            print(f"Warning: Could not request GPS stream: {e}")
        
        # Monitor GPS messages
        start_time = time.time()
        gps_raw_count = 0
        global_pos_count = 0
        last_gps_data = {}
        
        print("\nGPS Message Log:")
        print("Time     | Type           | Lat        | Lon        | Alt    | Fix | Sats")
        print("-" * 75)
        
        while (time.time() - start_time) < duration:
            msg = connection.recv_match(
                type=['GPS_RAW_INT', 'GLOBAL_POSITION_INT'], 
                blocking=False, 
                timeout=1
            )
            
            if msg:
                timestamp = datetime.now().strftime('%H:%M:%S')
                msg_type = msg.get_type()
                
                if msg_type == 'GPS_RAW_INT':
                    gps_raw_count += 1
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                    alt = msg.alt / 1000.0
                    fix_type = msg.fix_type
                    satellites = msg.satellites_visible
                    
                    print(f"{timestamp} | GPS_RAW_INT    | {lat:10.6f} | {lon:10.6f} | {alt:6.1f} | {fix_type}   | {satellites}")
                    
                    last_gps_data = {
                        'lat': lat, 'lon': lon, 'alt': alt,
                        'fix': fix_type, 'sats': satellites
                    }
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    global_pos_count += 1
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                    alt = msg.alt / 1000.0
                    
                    print(f"{timestamp} | GLOBAL_POS_INT | {lat:10.6f} | {lon:10.6f} | {alt:6.1f} | -   | -")
            
            else:
                # No message received
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"Waiting for GPS messages... {remaining:.1f}s remaining", end='\r')
                time.sleep(0.5)
        
        print(f"\n\nGPS Test Results:")
        print(f"  GPS_RAW_INT messages: {gps_raw_count}")
        print(f"  GLOBAL_POSITION_INT messages: {global_pos_count}")
        print(f"  Total GPS messages: {gps_raw_count + global_pos_count}")
        
        if last_gps_data:
            print(f"\nLast GPS Reading:")
            print(f"  Position: ({last_gps_data['lat']:.6f}, {last_gps_data['lon']:.6f})")
            print(f"  Altitude: {last_gps_data['alt']:.1f}m")
            print(f"  GPS Fix Type: {last_gps_data['fix']} (3=3D Fix)")
            print(f"  Satellites: {last_gps_data['sats']}")
            
            # Evaluate GPS quality
            if last_gps_data['fix'] >= 3 and last_gps_data['sats'] >= 6:
                print("  ✅ GPS Quality: GOOD")
            elif last_gps_data['fix'] >= 2:
                print("  ⚠️  GPS Quality: FAIR")
            else:
                print("  ❌ GPS Quality: POOR")
        
        connection.close()
        
        if gps_raw_count > 0 or global_pos_count > 0:
            print("✅ GPS message test PASSED")
            return True
        else:
            print("❌ No GPS messages received")
            return False
            
    except Exception as e:
        print(f"❌ GPS test failed: {e}")
        return False

def test_mavproxy_bridge_status():
    """Test if MAVProxy bridge is running"""
    print(f"\n" + "="*50)
    print("Testing MAVProxy Bridge Status...")
    print("="*50)
    
    try:
        import socket
        
        # Test UDP port 14550
        print("Testing UDP port 14550...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        
        try:
            # Try to bind to the port
            sock.bind(('0.0.0.0', 14550))
            print("❌ Port 14550 is available (MAVProxy bridge not running)")
            sock.close()
            return False
        except OSError:
            print("✅ Port 14550 is in use (MAVProxy bridge likely running)")
            sock.close()
            return True
            
    except Exception as e:
        print(f"❌ Bridge status test failed: {e}")
        return False

def test_network_connectivity():
    """Test network connectivity to common MAVProxy sources"""
    print(f"\n" + "="*50)
    print("Testing Network Connectivity...")
    print("="*50)
    
    import socket
    
    # Test localhost connectivity
    print("Testing localhost connectivity...")
    try:
        sock = socket.create_connection(('127.0.0.1', 80), timeout=2)
        sock.close()
        print("✅ Localhost connectivity: OK")
        localhost_ok = True
    except:
        print("❌ Localhost connectivity: FAILED")
        localhost_ok = False
    
    # Test external connectivity (for WiFi telemetry)
    print("Testing external network connectivity...")
    try:
        sock = socket.create_connection(('8.8.8.8', 53), timeout=3)
        sock.close()
        print("✅ External connectivity: OK")
        external_ok = True
    except:
        print("❌ External connectivity: FAILED")
        external_ok = False
    
    return localhost_ok and external_ok

def main():
    """Main test function"""
    print("="*60)
    print("USV GAS MAPPING - MAVLINK CONNECTION TEST")
    print("="*60)
    print("This script will test MAVLink connectivity with Pixhawk")
    print("via MAVProxy bridge or direct connection.")
    print()
    
    # Configuration
    print("Configuration:")
    print("- Expected MAVProxy bridge at udp:0.0.0.0:14550")
    print("- System ID: 254 (different from Mission Planner)")
    print("- Component ID: 190")
    print()
    
    # Test sequence
    success_count = 0
    total_tests = 6
    
    # Test 1: MAVLink Import
    if test_mavlink_import():
        success_count += 1
    
    # Test 2: Connection Types
    connection_types = test_connection_types()
    success_count += 1  # Always succeeds
    
    # Test 3: Network Connectivity
    if test_network_connectivity():
        success_count += 1
    
    # Test 4: MAVProxy Bridge Status
    bridge_running = test_mavproxy_bridge_status()
    if bridge_running:
        success_count += 1
    
    # Test 5: UDP Connection
    connection_string = 'udp:0.0.0.0:14550'
    if test_udp_connection(connection_string, timeout=10):
        success_count += 1
        
        # Test 6: GPS Messages (only if connected)
        if test_gps_messages(connection_string, duration=15):
            success_count += 1
    else:
        print("\n⚠️  Skipping GPS test due to connection failure")
    
    # Summary
    print("\n" + "="*60)
    print("MAVLINK TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED - MAVLink connection is ready!")
        print("\n✅ You can now run the gas mapping system")
        exit_code = 0
    elif success_count >= 4:
        print("⚠️  PARTIAL SUCCESS - Connection working but some issues")
        exit_code = 1
    else:
        print("❌ MULTIPLE FAILURES - Check MAVProxy bridge and connections")
        exit_code = 2
    
    print("\nTroubleshooting tips:")
    if not bridge_running:
        print("- Start MAVProxy bridge on Desktop:")
        print("  mavproxy.py --master=COM3 --out=udp:RASPBERRY_PI_IP:14550")
    
    print("- Verify Mission Planner is NOT directly connected to Pixhawk")
    print("- Check network connectivity between Desktop and Raspberry Pi")
    print("- Ensure firewall allows UDP port 14550")
    print("- Try different connection strings if UDP fails")
    
    print(f"\nConnection strings to try:")
    for name, conn_str in connection_types.items():
        print(f"  {name}: {conn_str}")
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)