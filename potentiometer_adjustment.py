#!/usr/bin/env python3
"""
Real-time Voltage Monitoring untuk Adjustment Potensiometer
Gunakan script ini sambil memutar potensiometer
"""

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

def live_voltage_monitor():
    """Monitor voltage real-time untuk adjustment potensiometer"""
    
    print("="*60)
    print("LIVE VOLTAGE MONITORING - POTENTIOMETER ADJUSTMENT")
    print("="*60)
    print("TARGET:")
    print("  Clean Air: 1.0 - 2.0V")
    print("  Stability: <2% variation")
    print("  NO 0.000V atau 5.000V")
    print("="*60)
    
    try:
        # Initialize ADC
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        
        # Setup channels
        channels = {
            'TGS2600': AnalogIn(ads, ADS.P0),
            'TGS2602': AnalogIn(ads, ADS.P1), 
            'TGS2610': AnalogIn(ads, ADS.P2)
        }
        
        print("Monitoring started. Press Ctrl+C to stop")
        print("Putar potensiometer sambil monitor voltage...")
        print()
        
        sample_count = 0
        voltage_history = {sensor: [] for sensor in channels.keys()}
        
        while True:
            sample_count += 1
            current_voltages = {}
            
            # Read all channels
            for sensor_name, channel in channels.items():
                voltage = channel.voltage
                current_voltages[sensor_name] = voltage
                voltage_history[sensor_name].append(voltage)
                
                # Keep only last 10 samples for stability calculation
                if len(voltage_history[sensor_name]) > 10:
                    voltage_history[sensor_name].pop(0)
            
            # Display current readings
            print(f"\rSample {sample_count:4d} | ", end="")
            
            for sensor_name, voltage in current_voltages.items():
                # Calculate stability if enough samples
                if len(voltage_history[sensor_name]) >= 5:
                    recent = voltage_history[sensor_name][-5:]  # Last 5 samples
                    mean_v = sum(recent) / len(recent)
                    var = sum((v - mean_v)**2 for v in recent) / len(recent)
                    std_v = var ** 0.5
                    stability = (std_v / mean_v * 100) if mean_v > 0 else 999
                    
                    # Status indicator
                    if voltage < 0.1:
                        status = "❌LOW"
                    elif voltage > 4.8:
                        status = "❌HIGH" 
                    elif 1.0 <= voltage <= 2.0 and stability < 2:
                        status = "✅GOOD"
                    elif 1.0 <= voltage <= 2.0:
                        status = "⚠️UNST"
                    else:
                        status = "⚠️ADJ"
                        
                    print(f"{sensor_name}: {voltage:.3f}V ({stability:.1f}%) {status} | ", end="")
                else:
                    print(f"{sensor_name}: {voltage:.3f}V | ", end="")
            
            time.sleep(0.5)  # Update 2x per second
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("="*60)
        print("FINAL ANALYSIS:")
        
        for sensor_name in channels.keys():
            if len(voltage_history[sensor_name]) >= 5:
                recent = voltage_history[sensor_name][-10:]  # Last 10 samples
                mean_v = sum(recent) / len(recent)
                var = sum((v - mean_v)**2 for v in recent) / len(recent)
                std_v = var ** 0.5
                stability = (std_v / mean_v * 100) if mean_v > 0 else 999
                
                print(f"{sensor_name}:")
                print(f"  Final Voltage: {mean_v:.3f}V ± {std_v:.3f}V")
                print(f"  Stability: {stability:.2f}%")
                
                # Recommendation
                if voltage < 0.5:
                    print(f"  📝 ACTION: Increase GAIN (putar clockwise)")
                elif voltage > 4.0:
                    print(f"  📝 ACTION: Decrease GAIN (putar counter-clockwise)")
                elif stability > 5:
                    print(f"  📝 ACTION: Check connections, reduce interference")
                elif 1.0 <= mean_v <= 2.0 and stability < 2:
                    print(f"  ✅ STATUS: OPTIMAL - Ready for calibration!")
                else:
                    print(f"  📝 ACTION: Adjust to 1.0-2.0V range")
                print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check hardware connections!")

if __name__ == "__main__":
    live_voltage_monitor()
