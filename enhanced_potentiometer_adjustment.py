#!/usr/bin/env python3
"""
Enhanced Potentiometer Adjustment untuk Gas Sensor Troubleshooting
Khusus untuk mengatasi masalah Pertamax & Dexlite detection
"""

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import json
from datetime import datetime

class EnhancedPotentiometerAdjustment:
    def __init__(self):
        """Initialize enhanced potentiometer adjustment system"""
        try:
            # Initialize ADC
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)
            
            # Setup channels
            self.channels = {
                'TGS2600': AnalogIn(self.ads, ADS.P0),
                'TGS2602': AnalogIn(self.ads, ADS.P1), 
                'TGS2610': AnalogIn(self.ads, ADS.P2)
            }
            
            # Target voltages berdasarkan analisis troubleshooting
            self.target_voltages = {
                'TGS2600': {'target': 1.5, 'current': 2.557, 'status': 'NEEDS_ADJUSTMENT'},
                'TGS2602': {'target': 1.5, 'current': 2.388, 'status': 'NEEDS_ADJUSTMENT'},
                'TGS2610': {'target': 1.1, 'current': 1.102, 'status': 'OK'}
            }
            
            # Voltage history untuk stability tracking
            self.voltage_history = {sensor: [] for sensor in self.channels.keys()}
            
            print("✅ Enhanced Potentiometer Adjustment System initialized")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise

    def show_problem_analysis(self):
        """Tampilkan analisis masalah berdasarkan troubleshooting"""
        print("="*80)
        print("🚨 MASALAH TERIDENTIFIKASI - PERTAMAX & DEXLITE DETECTION")
        print("="*80)
        print("\n📊 CURRENT STATUS vs TARGET:")
        print("-"*50)
        
        for sensor, data in self.target_voltages.items():
            current = data['current']
            target = data['target']
            diff = current - target
            status = data['status']
            
            if status == 'NEEDS_ADJUSTMENT':
                action = "⬇️ DECREASE GAIN" if diff > 0 else "⬆️ INCREASE GAIN"
                print(f"{sensor}:")
                print(f"  Current: {current:.3f}V | Target: {target:.3f}V | Diff: {diff:+.3f}V")
                print(f"  Status: ❌ {status}")
                print(f"  Action: {action} (putar counter-clockwise)")
            else:
                print(f"{sensor}:")
                print(f"  Current: {current:.3f}V | Target: {target:.3f}V | Diff: {diff:+.3f}V")
                print(f"  Status: ✅ {status} - No adjustment needed")
            print()
        
        print("🎯 EXPECTED RESULTS AFTER ADJUSTMENT:")
        print("• Pertamax detection: TGS2602 akan menghasilkan PPM > 1")
        print("• Dexlite detection: TGS2600 & TGS2602 akan menghasilkan PPM > 1")
        print("• Model accuracy akan meningkat untuk kedua gas ini")
        print("\n" + "="*80)

    def live_adjustment_monitor(self):
        """Enhanced real-time monitoring dengan troubleshooting guidance"""
        print("\n🔧 LIVE ADJUSTMENT MONITORING - ENHANCED MODE")
        print("="*80)
        
        self.show_problem_analysis()
        
        print("\n📋 ADJUSTMENT INSTRUCTIONS:")
        print("1. TGS2600: Putar potentiometer COUNTER-CLOCKWISE ~40% (2.557V → 1.5V)")
        print("2. TGS2602: Putar potentiometer COUNTER-CLOCKWISE ~37% (2.388V → 1.5V)")
        print("3. TGS2610: JANGAN DIUBAH (sudah optimal di 1.1V)")
        print("\n⚠️  PENTING: Putar PERLAHAN, monitor perubahan real-time")
        print("⚠️  TARGET: Stabilitas <2%, tidak boleh 0.000V atau 5.000V")
        print("\nPress Ctrl+C to stop monitoring")
        print("-"*80)
        
        sample_count = 0
        adjustment_log = []
        
        try:
            while True:
                sample_count += 1
                current_voltages = {}
                current_time = datetime.now()
                
                # Read all channels
                for sensor_name, channel in self.channels.items():
                    voltage = channel.voltage
                    current_voltages[sensor_name] = voltage
                    self.voltage_history[sensor_name].append(voltage)
                    
                    # Keep only last 20 samples for stability calculation
                    if len(self.voltage_history[sensor_name]) > 20:
                        self.voltage_history[sensor_name].pop(0)
                
                # Display current readings with enhanced analysis
                print(f"\r⏰ {current_time.strftime('%H:%M:%S')} | Sample {sample_count:4d} | ", end="")
                
                all_sensors_ok = True
                
                for sensor_name, voltage in current_voltages.items():
                    target_data = self.target_voltages[sensor_name]
                    target = target_data['target']
                    
                    # Calculate stability if enough samples
                    if len(self.voltage_history[sensor_name]) >= 10:
                        recent = self.voltage_history[sensor_name][-10:]  # Last 10 samples
                        mean_v = sum(recent) / len(recent)
                        var = sum((v - mean_v)**2 for v in recent) / len(recent)
                        std_v = var ** 0.5
                        stability = (std_v / mean_v * 100) if mean_v > 0 else 999
                        
                        # Enhanced status with troubleshooting guidance
                        voltage_diff = voltage - target
                        
                        if voltage < 0.1:
                            status = "❌LOW"
                            all_sensors_ok = False
                        elif voltage > 4.8:
                            status = "❌HIGH"
                            all_sensors_ok = False
                        elif abs(voltage_diff) <= 0.05 and stability < 2:
                            status = "✅PERFECT"
                        elif abs(voltage_diff) <= 0.1 and stability < 3:
                            status = "✅GOOD"
                        elif abs(voltage_diff) <= 0.2:
                            if voltage_diff > 0:
                                status = "⬇️DECREASE"
                                all_sensors_ok = False
                            else:
                                status = "⬆️INCREASE"
                                all_sensors_ok = False
                        else:
                            status = "⚠️ADJUST"
                            all_sensors_ok = False
                        
                        print(f"{sensor_name}: {voltage:.3f}V({voltage_diff:+.3f}) {stability:.1f}% {status} | ", end="")
                    else:
                        print(f"{sensor_name}: {voltage:.3f}V | ", end="")
                
                # Log significant changes
                if sample_count % 20 == 0:  # Every 20 samples
                    log_entry = {
                        'timestamp': current_time.isoformat(),
                        'sample': sample_count,
                        'voltages': current_voltages.copy(),
                        'all_sensors_ok': all_sensors_ok
                    }
                    adjustment_log.append(log_entry)
                
                # Alert when all sensors reach target
                if all_sensors_ok and sample_count > 50:
                    print(f"\n\n🎉 SUCCESS! All sensors reached target voltages!")
                    print("✅ Ready for calibration and data collection")
                    break
                
                time.sleep(0.5)  # 2Hz update rate
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            self.show_final_analysis(adjustment_log)

    def show_final_analysis(self, adjustment_log):
        """Enhanced final analysis dengan troubleshooting recommendations"""
        print("\n" + "="*80)
        print("📊 FINAL ANALYSIS & TROUBLESHOOTING RECOMMENDATIONS")
        print("="*80)
        
        for sensor_name in self.channels.keys():
            if len(self.voltage_history[sensor_name]) >= 10:
                recent = self.voltage_history[sensor_name][-20:]  # Last 20 samples
                mean_v = sum(recent) / len(recent)
                var = sum((v - mean_v)**2 for v in recent) / len(recent)
                std_v = var ** 0.5
                stability = (std_v / mean_v * 100) if mean_v > 0 else 999
                
                target = self.target_voltages[sensor_name]['target']
                voltage_diff = mean_v - target
                
                print(f"\n🔬 {sensor_name} ANALYSIS:")
                print(f"  Final Voltage: {mean_v:.3f}V ± {std_v:.3f}V")
                print(f"  Target: {target:.3f}V (Diff: {voltage_diff:+.3f}V)")
                print(f"  Stability: {stability:.2f}%")
                
                # Detailed recommendations
                if abs(voltage_diff) <= 0.05 and stability < 2:
                    print(f"  ✅ STATUS: OPTIMAL - Perfect for {sensor_name}")
                    print(f"  📝 ACTION: Ready for calibration!")
                elif abs(voltage_diff) <= 0.1 and stability < 3:
                    print(f"  ✅ STATUS: GOOD - Acceptable for {sensor_name}")
                    print(f"  📝 ACTION: Can proceed to calibration")
                elif voltage_diff > 0.2:
                    print(f"  ⚠️  STATUS: TOO HIGH - Need more adjustment")
                    print(f"  📝 ACTION: Continue counter-clockwise rotation")
                elif voltage_diff < -0.2:
                    print(f"  ⚠️  STATUS: TOO LOW - Over-adjusted")
                    print(f"  📝 ACTION: Rotate clockwise slightly")
                elif stability > 5:
                    print(f"  ⚠️  STATUS: UNSTABLE - Check connections")
                    print(f"  📝 ACTION: Check wiring, reduce interference")
                else:
                    print(f"  ⚠️  STATUS: NEEDS FINE-TUNING")
                    print(f"  📝 ACTION: Small adjustments needed")
        
        print(f"\n🎯 NEXT STEPS:")
        
        # Check if ready for next phase
        ready_sensors = 0
        total_sensors = len(self.channels)
        
        for sensor_name in self.channels.keys():
            if len(self.voltage_history[sensor_name]) >= 10:
                recent = self.voltage_history[sensor_name][-10:]
                mean_v = sum(recent) / len(recent)
                target = self.target_voltages[sensor_name]['target']
                voltage_diff = mean_v - target
                
                if abs(voltage_diff) <= 0.1:
                    ready_sensors += 1
        
        if ready_sensors == total_sensors:
            print("✅ 1. Hardware adjustment COMPLETE!")
            print("✅ 2. Run calibration: python3 enhanced_datasheet_gas_sensor.py → option 1")
            print("✅ 3. Collect training data: focus on pertamax & dexlite")
            print("✅ 4. Re-train model with new data")
            print("\n🚀 Expected improvement: Pertamax & Dexlite detection should work!")
        else:
            print(f"⚠️  1. Only {ready_sensors}/{total_sensors} sensors ready")
            print("⚠️  2. Continue adjustment for remaining sensors")
            print("⚠️  3. Re-run this script after adjustments")
        
        # Save adjustment log
        try:
            log_filename = f"adjustment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'sensor_targets': self.target_voltages,
                    'final_voltages': {sensor: self.voltage_history[sensor][-1] 
                                     for sensor in self.channels.keys()},
                    'adjustment_log': adjustment_log
                }, f, indent=2)
            print(f"\n💾 Adjustment log saved to: {log_filename}")
        except Exception as e:
            print(f"⚠️  Could not save log: {e}")

    def quick_voltage_check(self):
        """Quick voltage check untuk verify current state"""
        print("\n⚡ QUICK VOLTAGE CHECK")
        print("-"*40)
        
        current_voltages = {}
        for sensor_name, channel in self.channels.items():
            voltage = channel.voltage
            current_voltages[sensor_name] = voltage
            target = self.target_voltages[sensor_name]['target']
            diff = voltage - target
            
            if abs(diff) <= 0.05:
                status = "✅ OPTIMAL"
            elif abs(diff) <= 0.1:
                status = "✅ GOOD"
            elif diff > 0:
                status = "⬇️ TOO HIGH"
            else:
                status = "⬆️ TOO LOW"
            
            print(f"{sensor_name}: {voltage:.3f}V (target: {target:.3f}V) {status}")
        
        return current_voltages

def main():
    """Enhanced main function dengan troubleshooting workflow"""
    print("="*80)
    print("🔧 ENHANCED POTENTIOMETER ADJUSTMENT - TROUBLESHOOTING MODE")
    print("🎯 Khusus untuk mengatasi masalah Pertamax & Dexlite detection")
    print("="*80)
    
    try:
        adjuster = EnhancedPotentiometerAdjustment()
        
        while True:
            print("\n📋 MENU:")
            print("1. Show problem analysis (why pertamax/dexlite fail)")
            print("2. Quick voltage check")
            print("3. Live adjustment monitoring (MAIN FEATURE)")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                adjuster.show_problem_analysis()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                adjuster.quick_voltage_check()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                print("\n🚀 Starting live adjustment monitoring...")
                print("⚠️  Prepare your screwdriver for potentiometer adjustment!")
                input("Press Enter when ready...")
                adjuster.live_adjustment_monitor()
                
            elif choice == '4':
                print("👋 Exiting enhanced adjustment system")
                break
                
            else:
                print("❌ Invalid option!")
                
    except KeyboardInterrupt:
        print("\n\n👋 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Check hardware connections and try again")

if __name__ == "__main__":
    main()
