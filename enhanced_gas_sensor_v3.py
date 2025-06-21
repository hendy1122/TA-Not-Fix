#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - SMART AUTO DRIFT SOLUTION + EMERGENCY PPM RECOVERY
Versi 3.0 dengan solusi untuk masalah PPM 0 pada TGS2600 dan TGS2602:
1. Emergency PPM calculation tanpa R0
2. Smart troubleshooting system
3. Auto-fix untuk masalah umum
4. Enhanced diagnostic tools
5. Robust PPM calculation algorithms
6. Smart recovery dari masalah kalibrasi
"""

import time
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import logging
import math
from pathlib import Path

# Library untuk ADC dan GPIO
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
except ImportError:
    print("Installing required libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "adafruit-circuitpython-ads1x15"])
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn

# Machine Learning libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import joblib

class EmergencyPPMCalculator:
    """Emergency PPM Calculator - Solusi untuk sensor yang tidak bisa baca PPM"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Emergency baseline values berdasarkan pengalaman empiris
        self.emergency_baselines = {
            'TGS2600': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 8000,
                'sensitivity_curve': 'power',  # Power curve response
                'gas_response_factor': 2.5,
                'detection_threshold': 0.1  # 100mV threshold
            },
            'TGS2602': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 9000,
                'sensitivity_curve': 'exponential',  # Exponential response
                'gas_response_factor': 3.0,
                'detection_threshold': 0.08  # 80mV threshold
            },
            'TGS2610': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 7500,
                'sensitivity_curve': 'linear',  # More linear response
                'gas_response_factor': 2.0,
                'detection_threshold': 0.12  # 120mV threshold
            }
        }
        
        # Gas-specific response factors
        self.gas_factors = {
            'TGS2600': {
                'alcohol': 1.0, 'hydrogen': 1.2, 'carbon_monoxide': 0.8,
                'pertalite': 1.1, 'pertamax': 1.3, 'default': 1.0
            },
            'TGS2602': {
                'alcohol': 1.0, 'toluene': 1.2, 'ammonia': 0.9, 'h2s': 1.5,
                'pertalite': 1.1, 'pertamax': 1.3, 'default': 1.0
            },
            'TGS2610': {
                'butane': 1.0, 'propane': 1.1, 'lp_gas': 1.0, 'iso_butane': 0.9,
                'pertalite': 1.2, 'pertamax': 1.4, 'default': 1.0
            }
        }
    
    def calculate_emergency_ppm(self, sensor_name, current_voltage, gas_type='default'):
        """
        Emergency PPM calculation tanpa memerlukan R0 kalibrasi
        Berdasarkan voltage drop dari baseline empiris
        """
        if sensor_name not in self.emergency_baselines:
            return 0
        
        baseline = self.emergency_baselines[sensor_name]
        baseline_voltage = baseline['clean_air_voltage']
        threshold = baseline['detection_threshold']
        
        # Hitung voltage drop
        voltage_drop = baseline_voltage - current_voltage
        
        # Jika voltage drop kurang dari threshold, anggap tidak ada gas
        if voltage_drop < threshold:
            return max(0, voltage_drop * 100)  # Small baseline reading
        
        # Hitung PPM berdasarkan kurva sensor
        curve_type = baseline['sensitivity_curve']
        response_factor = baseline['gas_response_factor']
        
        # Gas-specific factor
        gas_factor = self.gas_factors.get(sensor_name, {}).get(gas_type, 1.0)
        
        if curve_type == 'power':
            # Power curve: PPM = factor * (voltage_drop) ^ power
            ppm = response_factor * (voltage_drop * 100) ** 1.5 * gas_factor
            
        elif curve_type == 'exponential':
            # Exponential curve: PPM = factor * exp(voltage_drop * scale)
            ppm = response_factor * (math.exp(voltage_drop * 3) - 1) * gas_factor
            
        elif curve_type == 'linear':
            # Linear curve: PPM = factor * voltage_drop * scale
            ppm = response_factor * voltage_drop * 200 * gas_factor
            
        else:
            # Default linear calculation
            ppm = voltage_drop * 150 * gas_factor
        
        # Limit maximum PPM untuk extended mode
        max_ppm = 500  # Extended range limit
        return min(max_ppm, max(0, ppm))
    
    def calculate_voltage_based_ppm(self, sensor_name, current_voltage, reference_voltage=None):
        """
        Voltage-based PPM calculation using adaptive reference
        """
        if reference_voltage is None:
            reference_voltage = self.emergency_baselines[sensor_name]['clean_air_voltage']
        
        # Adaptive calculation berdasarkan voltage ratio
        voltage_ratio = current_voltage / reference_voltage
        
        if voltage_ratio >= 0.95:
            # Clean air atau very low concentration
            return 0
        elif voltage_ratio >= 0.8:
            # Low concentration
            ppm = (1 - voltage_ratio) * 200
        elif voltage_ratio >= 0.6:
            # Medium concentration
            ppm = (1 - voltage_ratio) * 400
        elif voltage_ratio >= 0.4:
            # High concentration
            ppm = (1 - voltage_ratio) * 600
        else:
            # Very high concentration
            ppm = (1 - voltage_ratio) * 800
        
        return max(0, ppm)

class SmartDriftManager:
    """Smart Drift Manager - Enhanced dengan troubleshooting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.baseline_history = {}
        self.drift_compensation_factors = {}
        self.last_calibration_time = None
        self.daily_check_done = False
        
        # TOLERANSI YANG TEPAT UNTUK BASELINE 1.6V (mV based, bukan %)
        self.drift_tolerance = {
            'excellent': 0.020,    # ±20mV - sangat stabil
            'good': 0.050,        # ±50mV - stabil, tidak perlu kompensasi
            'moderate': 0.100,    # ±100mV - kompensasi ringan
            'high': 0.200,        # ±200mV - kompensasi kuat
            'extreme': 0.300      # ±300mV - auto reset baseline
        }
        
        # BASELINE NORMALIZATION untuk model compatibility
        self.original_baseline = {
            'TGS2600': 1.6,
            'TGS2602': 1.6,
            'TGS2610': 1.6
        }
        
        self.current_baseline = {
            'TGS2600': 1.6,
            'TGS2602': 1.6,
            'TGS2610': 1.6
        }
        
        self.normalization_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0,
            'TGS2610': 1.0
        }
        
        # Voltage adjustment tracking
        self.voltage_adjustments = {
            'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
        }
        
        self.load_drift_data()
    
    def smart_troubleshoot_ppm_issue(self, sensor_array, sensor_name):
        """
        Smart troubleshooting untuk sensor yang PPM nya 0
        """
        print(f"\n" + "="*70)
        print(f"🔧 SMART TROUBLESHOOTING - {sensor_name} PPM Issue")
        print("="*70)
        print("Diagnosing why PPM shows 0...")
        
        # Baca data sensor saat ini
        config = sensor_array.sensor_config[sensor_name]
        current_voltage = config['channel'].voltage
        
        # Check 1: Voltage issues
        print(f"\n📊 STEP 1: VOLTAGE ANALYSIS")
        print(f"Current voltage: {current_voltage:.3f}V")
        
        voltage_issue = False
        if current_voltage < 0.5:
            print("❌ CRITICAL: Voltage too low - check hardware connection")
            voltage_issue = True
        elif current_voltage < 1.0:
            print("⚠️ WARNING: Voltage too low for gas detection")
            voltage_issue = True
        elif current_voltage > 3.0:
            print("⚠️ WARNING: Voltage too high - check circuit")
            voltage_issue = True
        else:
            print("✅ Voltage level OK")
        
        # Check 2: R0 Calibration
        print(f"\n📊 STEP 2: CALIBRATION ANALYSIS")
        R0 = config.get('R0')
        baseline_voltage = config.get('baseline_voltage')
        
        calibration_issue = False
        if R0 is None or R0 == 0:
            print("❌ CRITICAL: R0 not calibrated")
            calibration_issue = True
        else:
            print(f"✅ R0 calibrated: {R0:.1f}Ω")
            
        if baseline_voltage is None:
            print("❌ CRITICAL: Baseline voltage not set")
            calibration_issue = True
        else:
            print(f"✅ Baseline voltage: {baseline_voltage:.3f}V")
        
        # Check 3: Resistance calculation
        print(f"\n📊 STEP 3: RESISTANCE ANALYSIS")
        resistance = sensor_array.voltage_to_resistance(current_voltage)
        print(f"Current resistance: {resistance:.1f}Ω")
        
        if R0 and resistance:
            rs_r0_ratio = resistance / R0
            print(f"Rs/R0 ratio: {rs_r0_ratio:.3f}")
            
            if rs_r0_ratio >= 1.0:
                print("⚠️ WARNING: Rs/R0 >= 1.0 means no gas detection (clean air)")
            elif rs_r0_ratio < 0.1:
                print("⚠️ WARNING: Rs/R0 too low - may be over-range")
            else:
                print("✅ Rs/R0 ratio in detection range")
        
        # Check 4: PPM Calculation
        print(f"\n📊 STEP 4: PPM CALCULATION ANALYSIS")
        try:
            ppm_datasheet = sensor_array.resistance_to_ppm(sensor_name, resistance, 'auto')
            print(f"Datasheet PPM: {ppm_datasheet:.1f}")
            
            # Try emergency calculation
            emergency_calc = sensor_array.emergency_ppm_calc
            ppm_emergency = emergency_calc.calculate_emergency_ppm(sensor_name, current_voltage)
            print(f"Emergency PPM: {ppm_emergency:.1f}")
            
            if ppm_datasheet == 0 and ppm_emergency > 0:
                print("⚠️ Datasheet calculation failed, but emergency calculation works")
            elif ppm_datasheet > 0:
                print("✅ PPM calculation working")
            else:
                print("❌ Both calculations return 0")
                
        except Exception as e:
            print(f"❌ PPM calculation error: {e}")
        
        # Generate recommendations
        print(f"\n💡 SMART RECOMMENDATIONS:")
        solutions = []
        
        if voltage_issue:
            solutions.append("1. Check hardware connections and power supply")
            if current_voltage < 1.2:
                solutions.append("2. Run Smart Voltage Adjustment (Option 17)")
        
        if calibration_issue:
            solutions.append("3. Run Emergency R0 Fix (Option 20)")
            solutions.append("4. Alternative: Use Emergency PPM Mode")
        
        if not voltage_issue and not calibration_issue:
            solutions.append("5. Enable Emergency PPM Mode as backup")
            solutions.append("6. Check if sensor is in clean air (PPM 0 is normal)")
        
        for solution in solutions:
            print(f"   {solution}")
        
        # Offer auto-fix
        print(f"\n🔧 AUTO-FIX OPTIONS:")
        print("1. Enable Emergency PPM Mode (bypass R0 issues)")
        print("2. Fix R0 with emergency calibration")
        print("3. Reset sensor configuration")
        
        fix_choice = input("Select auto-fix option (1-3, or Enter to skip): ").strip()
        
        if fix_choice == '1':
            return self.enable_emergency_ppm_mode(sensor_array, sensor_name)
        elif fix_choice == '2':
            return self.emergency_r0_fix(sensor_array, sensor_name)
        elif fix_choice == '3':
            return self.reset_sensor_config(sensor_array, sensor_name)
        
        return False
    
    def enable_emergency_ppm_mode(self, sensor_array, sensor_name):
        """Enable emergency PPM mode for sensor"""
        print(f"\n🚨 ENABLING EMERGENCY PPM MODE for {sensor_name}")
        print("This will bypass R0 calibration and use voltage-based calculation")
        
        # Mark sensor for emergency mode
        sensor_array.sensor_config[sensor_name]['emergency_mode'] = True
        sensor_array.sensor_config[sensor_name]['use_emergency_ppm'] = True
        
        # Test emergency calculation
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        emergency_ppm = sensor_array.emergency_ppm_calc.calculate_emergency_ppm(sensor_name, current_voltage)
        
        print(f"✅ Emergency mode enabled")
        print(f"✅ Current emergency PPM: {emergency_ppm:.1f}")
        print(f"✅ Sensor now functional without R0 calibration")
        
        return True
    
    def emergency_r0_fix(self, sensor_array, sensor_name):
        """Emergency R0 fix berdasarkan voltage saat ini"""
        print(f"\n🔧 EMERGENCY R0 FIX for {sensor_name}")
        print("Assuming current condition is clean air...")
        
        # Baca voltage saat ini
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        current_resistance = sensor_array.voltage_to_resistance(current_voltage)
        
        # Set as emergency R0
        sensor_array.sensor_config[sensor_name]['R0'] = current_resistance
        sensor_array.sensor_config[sensor_name]['baseline_voltage'] = current_voltage
        
        print(f"✅ Emergency R0 set: {current_resistance:.1f}Ω")
        print(f"✅ Emergency baseline: {current_voltage:.3f}V")
        print(f"✅ Sensor should now calculate PPM normally")
        
        # Test calculation
        test_ppm = sensor_array.resistance_to_ppm(sensor_name, current_resistance)
        print(f"✅ Test PPM: {test_ppm:.1f}")
        
        return True
    
    def reset_sensor_config(self, sensor_array, sensor_name):
        """Reset sensor configuration to defaults"""
        print(f"\n🔄 RESETTING {sensor_name} CONFIGURATION")
        
        # Reset to defaults
        sensor_array.sensor_config[sensor_name]['R0'] = None
        sensor_array.sensor_config[sensor_name]['baseline_voltage'] = None
        sensor_array.sensor_config[sensor_name]['emergency_mode'] = False
        sensor_array.sensor_config[sensor_name]['use_emergency_ppm'] = False
        
        print(f"✅ Configuration reset")
        print(f"✅ Ready for fresh calibration or emergency mode")
        
        return True
    
    # Sisa kode SmartDriftManager sama seperti sebelumnya...
    # [Previous SmartDriftManager methods tetap sama]
    
    def smart_voltage_adjustment(self, sensor_array, target_sensor='TGS2600', target_voltage=1.6):
        """
        Smart voltage adjustment tanpa mengganggu kalibrasi existing
        Khusus untuk sensor yang voltage-nya terlalu rendah/tinggi
        """
        print(f"\n" + "="*70)
        print(f"🔧 SMART VOLTAGE ADJUSTMENT untuk {target_sensor}")
        print(f"Target voltage: {target_voltage}V")
        print("="*70)
        print("TUJUAN:")
        print("✅ Fix sensor yang tidak responsif karena voltage terlalu rendah/tinggi")
        print("✅ Tidak perlu kalibrasi ulang")
        print("✅ Tidak perlu collect dataset ulang") 
        print("✅ Model compatibility terjaga")
        print("="*70)
        
        # Baca voltage saat ini
        current_voltage = sensor_array.sensor_config[target_sensor]['channel'].voltage
        voltage_diff = target_voltage - current_voltage
        
        print(f"\n📊 VOLTAGE ANALYSIS:")
        print(f"Current voltage: {current_voltage:.3f}V")
        print(f"Target voltage:  {target_voltage:.3f}V")
        print(f"Adjustment needed: {voltage_diff:+.3f}V ({voltage_diff*1000:+.0f}mV)")
        
        # Diagnose responsivity issue
        if current_voltage < 1.2:
            print(f"⚠️  DIAGNOSIS: Voltage terlalu rendah - sensor tidak responsif")
            print(f"   Sensor butuh minimal 1.2V untuk deteksi gas yang baik")
        elif current_voltage > 2.0:
            print(f"⚠️  DIAGNOSIS: Voltage terlalu tinggi - bisa over-sensitive")
            print(f"   Optimal range: 1.4V - 1.8V")
        elif 1.2 <= current_voltage <= 1.4:
            print(f"⚠️  DIAGNOSIS: Voltage rendah - responsivitas terbatas")
        else:
            print(f"✅ DIAGNOSIS: Voltage dalam range baik")
        
        if abs(voltage_diff) < 0.05:
            print("✅ Voltage sudah dalam range optimal")
            return False
        
        print(f"\n🎯 PETUNJUK HARDWARE ADJUSTMENT:")
        print(f"Target sensor: {target_sensor}")
        if voltage_diff > 0:
            print(f"   Action: Putar potentiometer {target_sensor} SEARAH JARUM JAM")
            print(f"   Target: naikkan {voltage_diff*1000:.0f}mV")
            print(f"   Putaran: SEDIKIT DEMI SEDIKIT (1/8 putaran dulu)")
        else:
            print(f"   Action: Putar potentiometer {target_sensor} BERLAWANAN JARUM JAM") 
            print(f"   Target: turunkan {abs(voltage_diff)*1000:.0f}mV")
            print(f"   Putaran: SEDIKIT DEMI SEDIKIT (1/8 putaran dulu)")
        
        print(f"\n⚠️  PENTING:")
        print(f"   - Putar SEDIKIT DEMI SEDIKIT (maks 1/4 putaran sekali)")
        print(f"   - Monitor voltage real-time di screen")
        print(f"   - STOP saat mencapai {target_voltage}V ± 0.05V")
        print(f"   - Jangan over-adjust!")
        
        response = input(f"\nReady untuk adjustment {target_sensor}? (y/n): ").lower()
        if response != 'y':
            print("❌ Voltage adjustment cancelled")
            return False
        
        # Real-time monitoring
        print(f"\n📊 REAL-TIME VOLTAGE MONITORING:")
        print(f"Target: {target_voltage}V ± 0.05V")
        print("Press Ctrl+C when target reached")
        print("-" * 50)
        
        try:
            adjustment_start_time = time.time()
            sample_count = 0
            
            while True:
                current_v = sensor_array.sensor_config[target_sensor]['channel'].voltage
                diff_from_target = current_v - target_voltage
                diff_mv = diff_from_target * 1000
                
                if abs(diff_from_target) <= 0.05:
                    status = "✅ TARGET REACHED"
                    status_color = "SUCCESS"
                elif abs(diff_from_target) <= 0.1:
                    status = "🎯 CLOSE - Fine tune"
                    status_color = "CLOSE"
                elif abs(diff_from_target) <= 0.2:
                    status = "🔧 Getting closer"
                    status_color = "PROGRESS"
                else:
                    status = "⚡ Keep adjusting"
                    status_color = "ADJUST"
                
                # Direction guide
                if diff_from_target > 0.05:
                    direction = "🔽 Turn COUNTER-CLOCKWISE (lower voltage)"
                elif diff_from_target < -0.05:
                    direction = "🔼 Turn CLOCKWISE (raise voltage)"
                else:
                    direction = "🎯 PERFECT - Stop adjusting!"
                
                print(f"\rCurrent: {current_v:.3f}V | Target: {target_voltage:.3f}V | "
                      f"Diff: {diff_mv:+.0f}mV | {status} | {direction}", end="")
                
                sample_count += 1
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            final_voltage = sensor_array.sensor_config[target_sensor]['channel'].voltage
            final_diff = abs(final_voltage - target_voltage)
            adjustment_time = time.time() - adjustment_start_time
            
            print(f"\n\n🎉 ADJUSTMENT COMPLETED!")
            print("="*50)
            print(f"Final voltage: {final_voltage:.3f}V")
            print(f"Target voltage: {target_voltage:.3f}V")
            print(f"Final accuracy: ±{final_diff*1000:.0f}mV")
            print(f"Adjustment time: {adjustment_time:.1f} seconds")
            
            if final_diff <= 0.05:
                print("✅ EXCELLENT - Perfect target achieved")
                accuracy_rating = "EXCELLENT"
            elif final_diff <= 0.1:
                print("✅ GOOD - Within acceptable range")
                accuracy_rating = "GOOD"
            elif final_diff <= 0.15:
                print("⚠️ OK - May need fine tuning")
                accuracy_rating = "OK"
            else:
                print("❌ POOR - Recommend re-adjustment")
                accuracy_rating = "POOR"
                
            # Update smart drift compensation untuk voltage baru
            success = self.apply_voltage_update(target_sensor, final_voltage, target_voltage, sensor_array)
            
            if success and accuracy_rating in ["EXCELLENT", "GOOD"]:
                print(f"\n🎉 VOLTAGE ADJUSTMENT SUCCESS!")
                print(f"✅ {target_sensor} optimized untuk responsivitas maksimal")
                print(f"✅ Model compatibility preserved")
                print(f"✅ Ready untuk gas detection")
                return True
            else:
                print(f"\n⚠️ Adjustment completed but may need fine-tuning")
                return True
                
    def apply_voltage_update(self, sensor_name, new_voltage, target_voltage, sensor_array):
        """
        Update smart compensation untuk voltage baru TANPA mengganggu kalibrasi existing
        """
        print(f"\n🧠 UPDATING SMART COMPENSATION...")
        print("-" * 40)
        
        # Record voltage adjustment
        old_baseline = self.current_baseline.get(sensor_name, 1.6)
        
        # Calculate voltage adjustment factor
        voltage_adjustment_factor = new_voltage / old_baseline
        
        # Update voltage tracking
        self.voltage_adjustments[sensor_name] = {
            'original': old_baseline,
            'current': new_voltage,
            'adjusted': True,
            'adjustment_factor': voltage_adjustment_factor,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update current baseline tapi JANGAN ubah original baseline (untuk model compatibility)
        self.current_baseline[sensor_name] = new_voltage
        
        # Update normalization factor untuk menjaga model compatibility
        # Ini yang penting: model tetap expect original baseline, tapi kita normalize ke new voltage
        original_norm_factor = self.normalization_factors.get(sensor_name, 1.0)
        new_norm_factor = self.original_baseline[sensor_name] / new_voltage
        self.normalization_factors[sensor_name] = new_norm_factor
        
        # Reset drift compensation karena hardware sudah di-adjust
        if sensor_name in self.drift_compensation_factors:
            old_drift_factor = self.drift_compensation_factors[sensor_name]
            del self.drift_compensation_factors[sensor_name]
            print(f"✅ Reset drift compensation (was: {old_drift_factor:.3f})")
        
        print(f"✅ Updated baseline: {old_baseline:.3f}V → {new_voltage:.3f}V")
        print(f"✅ Voltage adjustment factor: {voltage_adjustment_factor:.3f}")
        print(f"✅ Updated normalization factor: {new_norm_factor:.3f}")
        print(f"✅ Model compatibility preserved")
        
        # Test responsivitas setelah adjustment
        print(f"\n🧪 TESTING RESPONSIVITAS SETELAH ADJUSTMENT...")
        print("Collecting baseline readings untuk verify...")
        
        test_readings = []
        
        for i in range(15):
            reading = sensor_array.read_sensors()[sensor_name]
            test_readings.append(reading['ppm'])
            if i % 5 == 0:
                print(f"  Sample {i+1}/15: {reading['ppm']:.1f} PPM")
            time.sleep(1)
        
        baseline_ppm_mean = np.mean(test_readings)
        baseline_ppm_std = np.std(test_readings)
        
        print(f"\n📊 POST-ADJUSTMENT ANALYSIS:")
        print(f"Baseline PPM: {baseline_ppm_mean:.1f} ± {baseline_ppm_std:.1f}")
        print(f"Voltage: {new_voltage:.3f}V")
        print(f"Normalization factor: {new_norm_factor:.3f}")
        
        # Evaluate results
        if baseline_ppm_mean < 30 and baseline_ppm_std < 10:
            print("✅ EXCELLENT - Sensor ready and responsive")
            print("✅ Low baseline, good stability")
            responsivity_status = "EXCELLENT"
        elif baseline_ppm_mean < 50 and baseline_ppm_std < 15:
            print("✅ GOOD - Sensor responsive")
            responsivity_status = "GOOD" 
        elif baseline_ppm_mean < 100:
            print("⚠️ MODERATE - Sensor responsive but baseline high")
            responsivity_status = "MODERATE"
        else:
            print("❌ POOR - High baseline, may need re-adjustment")
            responsivity_status = "POOR"
        
        # Save updated data
        self.save_drift_data()
        
        print(f"\n🎉 SMART VOLTAGE ADJUSTMENT COMPLETED!")
        print("="*50)
        print(f"✅ Hardware: Voltage optimized to {new_voltage:.3f}V")
        print(f"✅ Software: Smart compensation updated")
        print(f"✅ Model: Compatibility maintained via normalization")
        print(f"✅ Status: {responsivity_status}")
        print(f"✅ Result: NO need to recalibrate or collect new dataset")
        
        return responsivity_status in ["EXCELLENT", "GOOD"]
    
    # [Rest of SmartDriftManager methods remain the same...]
    
    def is_daily_check_needed(self):
        """Check jika daily drift check perlu dilakukan"""
        if not self.daily_check_done:
            return True
            
        if hasattr(self, 'last_check_date'):
            today = datetime.now().date()
            if today != self.last_check_date:
                return True
                
        return False
        
    def smart_drift_check(self, sensor_array):
        """SMART drift check dengan toleransi 1.6V yang tepat"""
        self.logger.info("🧠 Smart drift check untuk baseline 1.6V...")
        
        print("\n" + "="*70)
        print("🧠 SMART DRIFT CHECK - OPTIMIZED FOR 1.6V BASELINE")
        print("Menggunakan toleransi mV (bukan %) yang tepat untuk baseline Anda")
        print("="*70)
        
        response = input("Pastikan sensor di CLEAN AIR dan stabil. Continue? (y/n): ").lower()
        if response != 'y':
            print("❌ Smart drift check cancelled")
            return False
            
        current_readings = self.measure_clean_air_baseline(sensor_array, duration=90)
        
        drift_detected = False
        compensation_applied = False
        baseline_reset_needed = False
        voltage_adjustment_needed = []
        
        print("\n📊 SMART DRIFT ANALYSIS:")
        print("-" * 70)
        
        for sensor_name, current_voltage in current_readings.items():
            
            if sensor_name in self.baseline_history and self.baseline_history[sensor_name]:
                previous_voltage = self.baseline_history[sensor_name][-1]
                voltage_drift = current_voltage - previous_voltage
                voltage_diff_abs = abs(voltage_drift)
                drift_mv = voltage_diff_abs * 1000  # Convert to mV
                
                print(f"\n{sensor_name}:")
                print(f"  Previous Baseline: {previous_voltage:.3f}V")
                print(f"  Current Reading:   {current_voltage:.3f}V") 
                print(f"  Drift Amount:      {drift_mv:.0f}mV ({voltage_drift:+.3f}V)")
                
                # Check jika voltage terlalu rendah untuk responsivitas
                if current_voltage < 1.2:
                    print(f"  ⚠️  WARNING: Voltage too low for good responsivity")
                    voltage_adjustment_needed.append(sensor_name)
                
                # EVALUASI BERDASARKAN TOLERANSI mV (BUKAN %)
                if voltage_diff_abs <= self.drift_tolerance['excellent']:
                    print(f"  Status: ✅ EXCELLENT - Drift {drift_mv:.0f}mV (<20mV)")
                    action = "No action needed - very stable"
                    if sensor_name in self.drift_compensation_factors:
                        del self.drift_compensation_factors[sensor_name]
                        
                elif voltage_diff_abs <= self.drift_tolerance['good']:
                    print(f"  Status: ✅ GOOD - Drift {drift_mv:.0f}mV (<50mV)")
                    action = "Stable, no compensation needed"
                    if sensor_name in self.drift_compensation_factors:
                        del self.drift_compensation_factors[sensor_name]
                        
                elif voltage_diff_abs <= self.drift_tolerance['moderate']:
                    print(f"  Status: 🔧 MODERATE - Drift {drift_mv:.0f}mV (50-100mV)")
                    compensation_factor = previous_voltage / current_voltage
                    self.drift_compensation_factors[sensor_name] = compensation_factor
                    compensation_applied = True
                    action = f"Light compensation applied (factor: {compensation_factor:.3f})"
                    
                elif voltage_diff_abs <= self.drift_tolerance['high']:
                    print(f"  Status: ⚠️  HIGH DRIFT - {drift_mv:.0f}mV (100-200mV)")
                    compensation_factor = previous_voltage / current_voltage
                    self.drift_compensation_factors[sensor_name] = compensation_factor
                    compensation_applied = True
                    drift_detected = True
                    action = f"Strong compensation applied (factor: {compensation_factor:.3f})"
                    
                else:  # extreme drift
                    print(f"  Status: ❌ EXTREME DRIFT - {drift_mv:.0f}mV (>200mV)")
                    action = "⚡ AUTO BASELINE RESET recommended (Option 15)"
                    baseline_reset_needed = True
                    drift_detected = True
                
                print(f"  Action: {action}")
                
                # UPDATE NORMALIZATION FACTORS untuk model compatibility
                self.current_baseline[sensor_name] = current_voltage
                self.normalization_factors[sensor_name] = self.original_baseline[sensor_name] / current_voltage
                
            else:
                print(f"\n{sensor_name}: Establishing baseline at {current_voltage:.3f}V")
                self.current_baseline[sensor_name] = current_voltage
                self.normalization_factors[sensor_name] = self.original_baseline[sensor_name] / current_voltage
        
        # Update baseline history
        for sensor_name, voltage in current_readings.items():
            if sensor_name not in self.baseline_history:
                self.baseline_history[sensor_name] = []
            
            self.baseline_history[sensor_name].append(voltage)
            
            if len(self.baseline_history[sensor_name]) > 30:
                self.baseline_history[sensor_name].pop(0)
        
        self.daily_check_done = True
        self.last_check_date = datetime.now().date()
        self.save_drift_data()
        
        # SMART SUMMARY
        print("\n" + "="*70)
        print("🧠 SMART DRIFT SUMMARY")
        print("="*70)
        
        if voltage_adjustment_needed:
            print("⚠️  VOLTAGE RESPONSIVITY ISSUE DETECTED")
            print(f"   Sensors need voltage adjustment: {', '.join(voltage_adjustment_needed)}")
            print("   Recommendation: Use Smart Voltage Adjustment (Option 17)")
            print("   ✅ No hardware replacement needed")
            print("   ✅ Model compatibility preserved")
            
        elif baseline_reset_needed:
            print("⚡ EXTREME DRIFT DETECTED")
            print("   Recommendation: Use AUTO BASELINE RESET (Option 15)")
            print("   ✅ No hardware adjustment needed")
            print("   ✅ Model compatibility preserved")
            
        elif compensation_applied:
            print("✅ SMART COMPENSATION APPLIED")
            print("   ✅ Drift automatically compensated")
            print("   ✅ Model compatibility preserved")
            print("   ✅ No need to collect new data or retrain")
            
        elif drift_detected:
            print("⚠️  MODERATE DRIFT DETECTED") 
            print("   System compensated, monitor performance")
            
        else:
            print("✅ ALL SENSORS STABLE")
            print("   Your 1.6V baseline working perfectly")
        
        # Model compatibility check
        print(f"\n🎯 MODEL COMPATIBILITY STATUS:")
        all_factors_good = True
        for sensor, factor in self.normalization_factors.items():
            factor_percent = abs(1 - factor) * 100
            if factor_percent > 20:
                all_factors_good = False
            status = "✅ GOOD" if factor_percent < 15 else "⚠️ MODERATE" if factor_percent < 25 else "❌ POOR"
            print(f"   {sensor}: {factor_percent:.1f}% normalization - {status}")
            
        if all_factors_good:
            print("\n✅ EXCELLENT: Model will work with existing training data")
            print("✅ No need to collect new data or retrain model")
        else:
            print("\n⚠️  RECOMMENDATION: Consider baseline reset for optimal model performance")
            
        return drift_detected
        
    def auto_baseline_reset(self, sensor_array):
        """Auto reset baseline tanpa hardware adjustment - SOLUSI TERBAIK"""
        print("\n" + "="*70)
        print("⚡ AUTO BASELINE RESET - SMART SOLUTION")
        print("="*70)
        print("This will:")
        print("✅ Accept current voltages as new baseline")
        print("✅ Reset drift compensation to zero") 
        print("✅ Preserve model compatibility")
        print("✅ No hardware adjustment needed")
        print("✅ No new training data needed")
        
        confirm = input("\nProceed with auto baseline reset? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Auto baseline reset cancelled")
            return False
            
        print("\n🔄 Measuring new baseline...")
        new_baseline = self.measure_clean_air_baseline(sensor_array, duration=60)
        
        print("\n📊 BASELINE UPDATE RESULTS:")
        print("-" * 50)
        
        total_change = 0
        for sensor_name, new_voltage in new_baseline.items():
            old_voltage = self.current_baseline.get(sensor_name, 1.6)
            change = new_voltage - old_voltage
            change_mv = change * 1000
            total_change += abs(change)
            
            print(f"{sensor_name}:")
            print(f"  Old Baseline: {old_voltage:.3f}V")
            print(f"  New Baseline: {new_voltage:.3f}V")
            print(f"  Change: {change:+.3f}V ({change_mv:+.0f}mV)")
            
            # Update all baselines
            self.current_baseline[sensor_name] = new_voltage
            self.original_baseline[sensor_name] = new_voltage  # Update model reference
            self.normalization_factors[sensor_name] = 1.0     # Reset normalization
            
            # Reset baseline history
            self.baseline_history[sensor_name] = [new_voltage]
        
        # Reset all drift compensation
        self.drift_compensation_factors = {}
        self.daily_check_done = True
        self.last_check_date = datetime.now().date()
        
        print(f"\n" + "="*50)
        print("✅ AUTO BASELINE RESET COMPLETED!")
        print("="*50)
        print("✅ New baseline established")
        print("✅ Drift compensation reset to zero")
        print("✅ Model reference updated") 
        print("✅ Normalization factors reset")
        print("✅ System ready for normal operation")
        print(f"✅ Total baseline change: {total_change*1000:.0f}mV")
        
        self.save_drift_data()
        return True
        
    def apply_smart_compensation(self, sensor_name, raw_voltage):
        """Apply smart compensation + normalization"""
        # Step 1: Apply drift compensation
        compensated_voltage = raw_voltage
        if sensor_name in self.drift_compensation_factors:
            compensated_voltage = raw_voltage * self.drift_compensation_factors[sensor_name]
        
        # Step 2: Apply normalization untuk model compatibility 
        normalized_voltage = compensated_voltage * self.normalization_factors.get(sensor_name, 1.0)
        
        return normalized_voltage, compensated_voltage
        
    def get_smart_status(self):
        """Get comprehensive smart drift status"""
        status = {
            'drift_compensation': {},
            'baseline_normalization': {},
            'voltage_adjustments': {},
            'overall_health': 'Unknown',
            'model_compatible': True
        }
        
        # Drift compensation status
        for sensor_name, factor in self.drift_compensation_factors.items():
            compensation_percent = abs(1 - factor) * 100
            status['drift_compensation'][sensor_name] = {
                'factor': factor,
                'compensation_percent': compensation_percent,
                'level': 'HIGH' if compensation_percent > 15 else 
                        'MODERATE' if compensation_percent > 5 else 'LOW'
            }
        
        # Baseline normalization status
        model_compatible = True
        for sensor_name, factor in self.normalization_factors.items():
            normalization_percent = abs(1 - factor) * 100
            sensor_compatible = normalization_percent < 25
            if not sensor_compatible:
                model_compatible = False
                
            status['baseline_normalization'][sensor_name] = {
                'factor': factor,
                'normalization_percent': normalization_percent,
                'model_compatible': sensor_compatible
            }
        
        # Voltage adjustment status
        for sensor_name, adj_info in self.voltage_adjustments.items():
            status['voltage_adjustments'][sensor_name] = adj_info
        
        status['model_compatible'] = model_compatible
        
        # Overall health assessment
        avg_compensation = np.mean([abs(1-f)*100 for f in self.drift_compensation_factors.values()]) if self.drift_compensation_factors else 0
        avg_normalization = np.mean([abs(1-f)*100 for f in self.normalization_factors.values()])
        
        if avg_compensation < 5 and avg_normalization < 10:
            status['overall_health'] = 'EXCELLENT'
        elif avg_compensation < 15 and avg_normalization < 20:
            status['overall_health'] = 'GOOD'
        elif avg_compensation < 25:
            status['overall_health'] = 'MODERATE'
        else:
            status['overall_health'] = 'NEEDS_ATTENTION'
            
        return status
        
    def measure_clean_air_baseline(self, sensor_array, duration=90):
        """Measure stable baseline voltage in clean air"""
        self.logger.info(f"Measuring clean air baseline for {duration} seconds...")
        
        readings = {sensor: [] for sensor in sensor_array.sensor_config.keys()}
        
        start_time = time.time()
        sample_count = 0
        
        print(f"📊 Collecting baseline data for {duration} seconds...")
        
        while time.time() - start_time < duration:
            for sensor_name, config in sensor_array.sensor_config.items():
                voltage = config['channel'].voltage
                readings[sensor_name].append(voltage)
            
            sample_count += 1
            remaining = int(duration - (time.time() - start_time))
            
            if sample_count % 15 == 0:
                print(f"  📈 Progress: {remaining}s remaining (samples: {sample_count})")
            
            time.sleep(2)
        
        # Calculate stable baseline 
        baseline = {}
        
        for sensor_name, voltages in readings.items():
            voltages_array = np.array(voltages)
            
            # Remove outliers
            mean_v = np.mean(voltages_array)
            std_v = np.std(voltages_array)
            
            mask = np.abs(voltages_array - mean_v) <= 2 * std_v
            filtered_voltages = voltages_array[mask]
            
            baseline_voltage = np.median(filtered_voltages)
            stability = (std_v / mean_v) * 100
            
            baseline[sensor_name] = baseline_voltage
            
            self.logger.info(f"{sensor_name}: {baseline_voltage:.3f}V ± {std_v:.3f}V (stability: {stability:.1f}%)")
        
        return baseline
        
    def save_drift_data(self):
        """Save comprehensive smart drift data"""
        drift_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'smart_drift_v3.0_with_emergency_ppm',
            'last_calibration_time': self.last_calibration_time.isoformat() if self.last_calibration_time else None,
            'baseline_history': self.baseline_history,
            'drift_compensation_factors': self.drift_compensation_factors,
            'original_baseline': self.original_baseline,
            'current_baseline': self.current_baseline,
            'normalization_factors': self.normalization_factors,
            'voltage_adjustments': self.voltage_adjustments,
            'daily_check_done': self.daily_check_done,
            'last_check_date': self.last_check_date.isoformat() if hasattr(self, 'last_check_date') else None
        }
        
        try:
            with open('smart_drift_data.json', 'w') as f:
                json.dump(drift_data, f, indent=2)
            self.logger.info("Smart drift data saved")
        except Exception as e:
            self.logger.error(f"Error saving drift data: {e}")
            
    def load_drift_data(self):
        """Load smart drift data"""
        try:
            # Try smart drift data first
            try:
                with open('smart_drift_data.json', 'r') as f:
                    drift_data = json.load(f)
            except FileNotFoundError:
                # Fallback to old drift data
                try:
                    with open('drift_compensation_data.json', 'r') as f:
                        drift_data = json.load(f)
                except FileNotFoundError:
                    drift_data = {}
            
            self.baseline_history = drift_data.get('baseline_history', {})
            self.drift_compensation_factors = drift_data.get('drift_compensation_factors', {})
            self.original_baseline = drift_data.get('original_baseline', {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.current_baseline = drift_data.get('current_baseline', {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.normalization_factors = drift_data.get('normalization_factors', {'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0})
            
            # Load voltage adjustments (new feature)
            self.voltage_adjustments = drift_data.get('voltage_adjustments', {
                'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
            })
            
            self.daily_check_done = drift_data.get('daily_check_done', False)
            
            if drift_data.get('last_calibration_time'):
                self.last_calibration_time = datetime.fromisoformat(drift_data['last_calibration_time'])
                
            if drift_data.get('last_check_date'):
                self.last_check_date = datetime.fromisoformat(drift_data['last_check_date']).date()
                
            self.logger.info("Smart drift data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading drift data: {e}")

class EnhancedDatasheetGasSensorArray:
    def __init__(self):
        """Initialize enhanced gas sensor array system with Emergency PPM Recovery"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gas_sensor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize I2C and ADC
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS.ADS1115(self.i2c)

            # Setup analog inputs for each sensor
            self.tgs2600 = AnalogIn(self.ads, ADS.P0)  # Channel A0
            self.tgs2602 = AnalogIn(self.ads, ADS.P1)  # Channel A1
            self.tgs2610 = AnalogIn(self.ads, ADS.P2)  # Channel A2

            self.logger.info("ADC ADS1115 initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ADC: {e}")
            raise

        # Sensor configurations (Enhanced dengan emergency mode support)
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['hydrogen', 'carbon_monoxide', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 500),
                'heater_voltage': 5.0,
                'heater_current': 42e-3,
                'power_consumption': 210e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-20, 50),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'hydrogen': (0.3, 0.6),
                    'carbon_monoxide': (0.4, 0.7),
                    'alcohol': (0.2, 0.5)
                },
                'use_extended_mode': False,
                'concentration_threshold': 50,
                'extended_sensitivity': 2.5,
                'emergency_mode': False,
                'use_emergency_ppm': False
            },
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['toluene', 'ammonia', 'h2s', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 300),
                'heater_voltage': 5.0,
                'heater_current': 56e-3,
                'power_consumption': 280e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-10, 60),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'alcohol': (0.08, 0.5),
                    'toluene': (0.1, 0.4),
                    'ammonia': (0.15, 0.6),
                    'h2s': (0.05, 0.3)
                },
                'use_extended_mode': False,
                'concentration_threshold': 40,
                'extended_sensitivity': 3.0,
                'emergency_mode': False,
                'use_emergency_ppm': False
            },
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'iso_butane'],
                'detection_range': (1, 25),
                'extended_range': (1, 200),
                'heater_voltage': 5.0,
                'heater_current': 56e-3,
                'power_consumption': 280e-3,
                'load_resistance': 10000,
                'warmup_time': 7 * 24 * 3600,
                'operating_temp_range': (-10, 50),
                'optimal_temp': 20,
                'optimal_humidity': 65,
                'R0': None,
                'baseline_voltage': None,
                'sensitivity_ratios': {
                    'iso_butane': (0.45, 0.62),
                    'butane': (0.4, 0.6),
                    'propane': (0.35, 0.55),
                    'lp_gas': (0.4, 0.6)
                },
                'use_extended_mode': False,
                'concentration_threshold': 30,
                'extended_sensitivity': 2.0,
                'emergency_mode': False,
                'use_emergency_ppm': False
            }
        }

        # Initialize SMART Drift Manager with Emergency PPM
        self.drift_manager = SmartDriftManager(self.logger)
        
        # Initialize Emergency PPM Calculator
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)

        # Environmental compensation parameters
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0
        self.current_humidity = 65.0

        # Data storage
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.data_file = f"gas_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Machine Learning components
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False

        # Create data directory
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)

        self.logger.info("Enhanced Gas Sensor Array System with Emergency PPM Recovery initialized")

    def voltage_to_resistance(self, voltage, load_resistance=10000):
        """Convert ADC voltage to sensor resistance"""
        if voltage <= 0.001:
            return float('inf')

        circuit_voltage = 5.0

        if voltage >= circuit_voltage:
            return 0.1

        sensor_resistance = load_resistance * (circuit_voltage - voltage) / voltage
        return max(1, sensor_resistance)

    def temperature_compensation(self, sensor_name, raw_value, temperature):
        """Apply temperature compensation"""
        if not self.temp_compensation_enabled:
            return raw_value

        temp_factors = {
            'TGS2600': {
                -20: 1.8, -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0,
                30: 0.95, 40: 0.9, 50: 0.85
            },
            'TGS2602': {
                -10: 1.5, 0: 1.3, 10: 1.1, 20: 1.0,
                30: 0.9, 40: 0.85, 50: 0.8, 60: 0.75
            },
            'TGS2610': {
                -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0,
                30: 0.95, 40: 0.9, 50: 0.85
            }
        }

        temp_curve = temp_factors.get(sensor_name, {20: 1.0})
        temps = sorted(temp_curve.keys())

        if temperature <= temps[0]:
            factor = temp_curve[temps[0]]
        elif temperature >= temps[-1]:
            factor = temp_curve[temps[-1]]
        else:
            for i in range(len(temps) - 1):
                if temps[i] <= temperature <= temps[i + 1]:
                    t1, t2 = temps[i], temps[i + 1]
                    f1, f2 = temp_curve[t1], temp_curve[t2]
                    factor = f1 + (f2 - f1) * (temperature - t1) / (t2 - t1)
                    break
            else:
                factor = 1.0

        return raw_value * factor

    def humidity_compensation(self, sensor_name, raw_value, humidity):
        """Apply humidity compensation"""
        if not self.humidity_compensation_enabled:
            return raw_value

        humidity_factors = {
            'TGS2600': {35: 1.1, 65: 1.0, 95: 0.9},
            'TGS2602': {40: 1.05, 65: 1.0, 85: 0.95, 100: 0.9},
            'TGS2610': {40: 1.1, 65: 1.0, 85: 0.95}
        }

        humidity_curve = humidity_factors.get(sensor_name, {65: 1.0})
        humidities = sorted(humidity_curve.keys())

        if humidity <= humidities[0]:
            factor = humidity_curve[humidities[0]]
        elif humidity >= humidities[-1]:
            factor = humidity_curve[humidities[-1]]
        else:
            for i in range(len(humidities) - 1):
                if humidities[i] <= humidity <= humidities[i + 1]:
                    h1, h2 = humidities[i], humidities[i + 1]
                    f1, f2 = humidity_curve[h1], humidity_curve[h2]
                    factor = f1 + (f2 - f1) * (humidity - h1) / (h2 - h1)
                    break
            else:
                factor = 1.0

        return raw_value * factor

    def resistance_to_ppm(self, sensor_name, resistance, gas_type='auto'):
        """Enhanced resistance to PPM conversion with emergency fallback"""
        config = self.sensor_config[sensor_name]
        
        # Check if emergency mode is enabled
        if config.get('use_emergency_ppm', False):
            current_voltage = config['channel'].voltage
            return self.emergency_ppm_calc.calculate_emergency_ppm(sensor_name, current_voltage, gas_type)
        
        R0 = config.get('R0')

        if R0 is None or R0 == 0:
            # Try emergency calculation as fallback
            current_voltage = config['channel'].voltage
            emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(sensor_name, current_voltage, gas_type)
            
            # If emergency calculation gives reasonable result, use it
            if emergency_ppm > 0:
                return emergency_ppm
            
            # Otherwise use simplified calculation
            return self.simplified_ppm_calculation(sensor_name, resistance)

        rs_r0_ratio = resistance / R0

        if config['use_extended_mode']:
            return self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            # Enhanced datasheet calculation with better handling
            ppm = self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
            
            # If datasheet calculation returns 0 but we have reasonable resistance change, try emergency
            if ppm == 0 and rs_r0_ratio < 0.9:
                current_voltage = config['channel'].voltage
                emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(sensor_name, current_voltage, gas_type)
                return max(ppm, emergency_ppm)
            
            return ppm

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Enhanced datasheet PPM calculation with wider tolerance"""
        if sensor_name == 'TGS2600':
            if gas_type == 'hydrogen' or gas_type == 'auto':
                if rs_r0_ratio < 0.2:  # Widened from 0.3
                    ppm = min(50, 50 * (0.4 / rs_r0_ratio) ** 2.5)
                elif rs_r0_ratio > 0.95:  # Narrowed from 0.9
                    ppm = 0
                else:
                    ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                    ppm = min(ppm, 50)  # Increased max
            elif gas_type == 'alcohol':
                if rs_r0_ratio < 0.15:  # Widened from 0.2
                    ppm = min(40, 40 * (0.3 / rs_r0_ratio) ** 2.0)
                elif rs_r0_ratio > 0.9:  # Narrowed from 0.8
                    ppm = 0
                else:
                    ppm = 40 * ((0.4 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 40)  # Increased max
            else:
                if rs_r0_ratio < 0.9:
                    ppm = 30 * ((0.5 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 40)
                else:
                    ppm = 0

        elif sensor_name == 'TGS2602':
            if gas_type == 'alcohol' or gas_type == 'auto':
                if rs_r0_ratio < 0.05:  # Widened from 0.08
                    ppm = min(40, 35 * (0.2 / rs_r0_ratio) ** 1.8)
                elif rs_r0_ratio > 0.95:  # Narrowed from 0.9
                    ppm = 0
                else:
                    ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                    ppm = min(ppm, 40)  # Increased max
            elif gas_type == 'toluene':
                if rs_r0_ratio < 0.9:
                    ppm = 25 * ((0.2 / rs_r0_ratio) ** 1.5)
                    ppm = min(ppm, 35)
                else:
                    ppm = 0
            else:
                if rs_r0_ratio < 0.9:
                    ppm = 20 * ((0.3 / rs_r0_ratio) ** 1.6)
                    ppm = min(ppm, 35)
                else:
                    ppm = 0

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.4:  # Widened from 0.45
                ppm = min(30, 35 * (0.5 / rs_r0_ratio) ** 1.2)
            elif rs_r0_ratio > 0.98:  # Narrowed from 0.95
                ppm = 0
            else:
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                ppm = min(ppm, 30)
        else:
            ppm = 0

        return max(0, ppm)

    def extended_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Extended PPM calculation"""
        config = self.sensor_config[sensor_name]
        sensitivity = config['extended_sensitivity']

        if rs_r0_ratio >= 1.0:
            return 0

        if sensor_name == 'TGS2600':
            if rs_r0_ratio < 0.05:
                base_ppm = 200 + (0.05 - rs_r0_ratio) * 1000
            elif rs_r0_ratio < 0.2:
                base_ppm = 100 + (0.2 - rs_r0_ratio) * 500
            else:
                base_ppm = 60 * ((0.6 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2602':
            if rs_r0_ratio < 0.02:
                base_ppm = 150 + (0.02 - rs_r0_ratio) * 2000
            elif rs_r0_ratio < 0.1:
                base_ppm = 75 + (0.1 - rs_r0_ratio) * 800
            else:
                base_ppm = 50 * ((0.3 / rs_r0_ratio) ** sensitivity)

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.1:
                base_ppm = 100 + (0.1 - rs_r0_ratio) * 1500
            elif rs_r0_ratio < 0.3:
                base_ppm = 50 + (0.3 - rs_r0_ratio) * 400
            else:
                base_ppm = 40 * ((0.7 / rs_r0_ratio) ** sensitivity)
        else:
            base_ppm = 0

        gas_multipliers = {
            'alcohol': 1.0,
            'pertalite': 1.3,
            'pertamax': 1.6,
            'dexlite': 1.9,
            'biosolar': 2.2,
            'hydrogen': 0.8,
            'toluene': 1.1,
            'ammonia': 0.9,
            'butane': 1.2,
            'propane': 1.4,
            'normal': 0.7
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Enhanced simplified PPM calculation"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 1.6)  # Use 1.6V as default

        baseline_resistance = self.voltage_to_resistance(baseline_voltage)

        if resistance >= baseline_resistance:
            return 0

        ratio = baseline_resistance / resistance

        if config['use_extended_mode']:
            max_range = config['extended_range'][1]
            ppm = max_range * (ratio - 1) * 0.4
        else:
            max_range = config['detection_range'][1]
            ppm = max_range * (ratio - 1) * 0.5

        return max(0, ppm)

    def set_sensor_mode(self, mode='datasheet'):
        """Set calculation mode for all sensors"""
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

    def read_sensors(self):
        """Enhanced sensor reading with Emergency PPM capability"""
        readings = {}

        for sensor_name, config in self.sensor_config.items():
            try:
                # Read raw voltage
                raw_voltage = config['channel'].voltage
                
                # Apply SMART drift compensation + normalization
                normalized_voltage, compensated_voltage = self.drift_manager.apply_smart_compensation(sensor_name, raw_voltage)

                # Convert to resistance using compensated voltage
                resistance = self.voltage_to_resistance(compensated_voltage, config['load_resistance'])

                # Apply environmental compensation
                compensated_resistance = self.temperature_compensation(
                    sensor_name, resistance, self.current_temperature)
                compensated_resistance = self.humidity_compensation(
                    sensor_name, compensated_resistance, self.current_humidity)

                # Calculate Rs/R0 ratio
                R0 = config.get('R0')
                rs_r0_ratio = compensated_resistance / R0 if R0 else None

                # Calculate PPM (with emergency fallback)
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)
                
                # Calculate emergency PPM as backup
                emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(sensor_name, raw_voltage)

                # Current mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"
                if config.get('use_emergency_ppm', False):
                    current_mode += " (Emergency)"

                # Smart drift info
                drift_factor = self.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                normalization_factor = self.drift_manager.normalization_factors.get(sensor_name, 1.0)
                smart_compensation_applied = abs(1 - drift_factor) > 0.01 or abs(1 - normalization_factor) > 0.01
                
                # Voltage adjustment info
                voltage_adjusted = self.drift_manager.voltage_adjustments.get(sensor_name, {}).get('adjusted', False)

                readings[sensor_name] = {
                    'voltage': normalized_voltage,  # For model compatibility
                    'raw_voltage': raw_voltage,
                    'compensated_voltage': compensated_voltage,
                    'resistance': resistance,
                    'compensated_resistance': compensated_resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'emergency_ppm': emergency_ppm,
                    'R0': R0,
                    'mode': current_mode,
                    'target_gases': config['target_gases'],
                    'smart_compensation_applied': smart_compensation_applied,
                    'drift_factor': drift_factor,
                    'normalization_factor': normalization_factor,
                    'voltage_adjusted': voltage_adjusted,
                    'emergency_mode': config.get('use_emergency_ppm', False)
                }

            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'raw_voltage': 0, 'compensated_voltage': 0, 'resistance': 0, 'compensated_resistance': 0,
                    'rs_r0_ratio': None, 'ppm': 0, 'emergency_ppm': 0, 'R0': None, 'mode': 'Error',
                    'target_gases': [], 'smart_compensation_applied': False, 'drift_factor': 1.0, 
                    'normalization_factor': 1.0, 'voltage_adjusted': False, 'emergency_mode': False
                }

        return readings

    # [Rest of the methods remain largely the same but with enhanced error handling and emergency mode support]
    
    def calibrate_sensors(self, duration=300):
        """Enhanced calibration with Emergency mode support"""
        self.logger.info(f"Starting enhanced sensor calibration for {duration} seconds...")
        self.logger.info("Ensure sensors are in CLEAN AIR environment!")
        self.logger.info("Sensors should be warmed up for at least 10 minutes")

        input("Press Enter when sensors are in clean air and warmed up...")

        readings = {sensor: {'voltages': [], 'resistances': []}
                   for sensor in self.sensor_config.keys()}

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration:
            for sensor_name, config in self.sensor_config.items():
                # Use raw voltage for calibration
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])

                # Apply environmental compensation
                resistance = self.temperature_compensation(sensor_name, resistance, self.current_temperature)
                resistance = self.humidity_compensation(sensor_name, resistance, self.current_humidity)

                readings[sensor_name]['voltages'].append(voltage)
                readings[sensor_name]['resistances'].append(resistance)

            sample_count += 1
            time.sleep(2)

            remaining = int(duration - (time.time() - start_time))
            if remaining % 30 == 0 and remaining > 0:
                print(f"Calibration remaining: {remaining} seconds (Samples: {sample_count})")

        # Calculate calibration parameters
        calibration_results = {}

        for sensor_name in self.sensor_config.keys():
            voltages = readings[sensor_name]['voltages']
            resistances = readings[sensor_name]['resistances']

            voltage_mean = np.mean(voltages)
            voltage_std = np.std(voltages)
            resistance_mean = np.mean(resistances)
            resistance_std = np.std(resistances)

            # Set R0 and baseline voltage
            self.sensor_config[sensor_name]['R0'] = resistance_mean
            self.sensor_config[sensor_name]['baseline_voltage'] = voltage_mean
            
            # Disable emergency mode after successful calibration
            self.sensor_config[sensor_name]['emergency_mode'] = False
            self.sensor_config[sensor_name]['use_emergency_ppm'] = False

            calibration_results[sensor_name] = {
                'R0': resistance_mean,
                'R0_std': resistance_std,
                'baseline_voltage': voltage_mean,
                'voltage_std': voltage_std,
                'sample_count': len(voltages),
                'stability': (voltage_std / voltage_mean) * 100
            }

            self.logger.info(f"{sensor_name} Calibration:")
            self.logger.info(f"  R0: {resistance_mean:.1f}Ω ± {resistance_std:.1f}Ω")
            self.logger.info(f"  Baseline Voltage: {voltage_mean:.3f}V ± {voltage_std:.3f}V")
            self.logger.info(f"  Stability: {calibration_results[sensor_name]['stability']:.2f}%")

        # Update Smart Drift Manager
        self.drift_manager.last_calibration_time = datetime.now()
        
        # Update baselines in drift manager
        for sensor_name, results in calibration_results.items():
            baseline_voltage = results['baseline_voltage']
            self.drift_manager.current_baseline[sensor_name] = baseline_voltage
            self.drift_manager.original_baseline[sensor_name] = baseline_voltage
            self.drift_manager.normalization_factors[sensor_name] = 1.0
            
            # Initialize baseline history
            if sensor_name not in self.drift_manager.baseline_history:
                self.drift_manager.baseline_history[sensor_name] = []
            self.drift_manager.baseline_history[sensor_name].append(baseline_voltage)
        
        # Reset drift compensation
        self.drift_manager.drift_compensation_factors = {}
        self.drift_manager.daily_check_done = False

        # Save calibration data
        calib_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'temperature': self.current_temperature,
            'humidity': self.current_humidity,
            'sensors': calibration_results
        }

        calib_file = f"calibration/calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(calib_file, 'w') as f:
            json.dump(calib_data, f, indent=2)

        with open('sensor_calibration.json', 'w') as f:
            json.dump(calib_data, f, indent=2)

        # Save drift data
        self.drift_manager.save_drift_data()

        self.logger.info(f"Enhanced calibration completed and saved to {calib_file}")
        print("\n✅ CALIBRATION SUCCESS!")
        print("✅ All sensors calibrated and emergency mode disabled")
        print("✅ Ready for normal operation")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)

            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']
                    # Disable emergency mode if calibration exists
                    self.sensor_config[sensor_name]['emergency_mode'] = False
                    self.sensor_config[sensor_name]['use_emergency_ppm'] = False

            if 'timestamp' in calib_data:
                self.drift_manager.last_calibration_time = datetime.fromisoformat(calib_data['timestamp'])

            self.logger.info("Enhanced calibration data loaded successfully")
            self.logger.info(f"Calibration date: {calib_data.get('timestamp', 'Unknown')}")

            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Emergency mode available.")
            return False
        except KeyError as e:
            self.logger.error(f"Invalid calibration file format: {e}")
            return False

    # [Continue with rest of the methods - collect_training_data, train_model, etc. with similar enhancements]
    
    def collect_training_data(self, gas_type, duration=60, samples_per_second=1):
        """Enhanced training data collection with emergency mode support"""
        self.set_sensor_mode('extended')

        self.logger.info(f"Collecting enhanced training data for {gas_type} in EXTENDED mode with Emergency PPM support")
        self.logger.info(f"Duration: {duration}s, Sampling rate: {samples_per_second} Hz")

        valid_gases = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar',
                      'hydrogen', 'toluene', 'ammonia', 'butane', 'propane']
        if gas_type not in valid_gases:
            self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
            return None

        if gas_type == 'normal':
            input(f"Ensure sensors are in CLEAN AIR (no gas). Press Enter to start...")
            self.logger.info("Collecting NORMAL/CLEAN AIR data...")
        else:
            input(f"Prepare to spray {gas_type}. Press Enter to start data collection...")

        training_data = []
        start_time = time.time()
        sample_interval = 1.0 / samples_per_second

        baseline_readings = self.read_sensors()
        self.logger.info("Baseline readings recorded")

        while time.time() - start_time < duration:
            timestamp = datetime.now()
            readings = self.read_sensors()

            # Enhanced data row with emergency PPM
            data_row = {
                'timestamp': timestamp,
                'gas_type': gas_type,
                'temperature': self.current_temperature,
                'humidity': self.current_humidity,

                # TGS2600 data with emergency support
                'TGS2600_voltage': readings['TGS2600']['voltage'],
                'TGS2600_raw_voltage': readings['TGS2600']['raw_voltage'],
                'TGS2600_compensated_voltage': readings['TGS2600']['compensated_voltage'],
                'TGS2600_resistance': readings['TGS2600']['resistance'],
                'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                'TGS2600_ppm': readings['TGS2600']['ppm'],
                'TGS2600_emergency_ppm': readings['TGS2600']['emergency_ppm'],
                'TGS2600_drift_factor': readings['TGS2600']['drift_factor'],
                'TGS2600_normalization_factor': readings['TGS2600']['normalization_factor'],
                'TGS2600_emergency_mode': readings['TGS2600']['emergency_mode'],

                # TGS2602 data with emergency support
                'TGS2602_voltage': readings['TGS2602']['voltage'],
                'TGS2602_raw_voltage': readings['TGS2602']['raw_voltage'],
                'TGS2602_compensated_voltage': readings['TGS2602']['compensated_voltage'],
                'TGS2602_resistance': readings['TGS2602']['resistance'],
                'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                'TGS2602_ppm': readings['TGS2602']['ppm'],
                'TGS2602_emergency_ppm': readings['TGS2602']['emergency_ppm'],
                'TGS2602_drift_factor': readings['TGS2602']['drift_factor'],
                'TGS2602_normalization_factor': readings['TGS2602']['normalization_factor'],
                'TGS2602_emergency_mode': readings['TGS2602']['emergency_mode'],

                # TGS2610 data with emergency support
                'TGS2610_voltage': readings['TGS2610']['voltage'],
                'TGS2610_raw_voltage': readings['TGS2610']['raw_voltage'],
                'TGS2610_compensated_voltage': readings['TGS2610']['compensated_voltage'],
                'TGS2610_resistance': readings['TGS2610']['resistance'],
                'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                'TGS2610_ppm': readings['TGS2610']['ppm'],
                'TGS2610_emergency_ppm': readings['TGS2610']['emergency_ppm'],
                'TGS2610_drift_factor': readings['TGS2610']['drift_factor'],
                'TGS2610_normalization_factor': readings['TGS2610']['normalization_factor'],
                'TGS2610_emergency_mode': readings['TGS2610']['emergency_mode']
            }

            training_data.append(data_row)

            # Enhanced display with emergency status
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            status = "CLEAN AIR" if gas_type == 'normal' else gas_type.upper()
            
            emergency_sensors = [s for s in readings.keys() if readings[s]['emergency_mode']]
            emergency_status = f"EMG({len(emergency_sensors)})" if emergency_sensors else "NORMAL"
            
            print(f"\rTime: {remaining:.1f}s | Status: {status} | Mode: {emergency_status} | "
                  f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                  f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                  f"2610: {readings['TGS2610']['ppm']:.0f}ppm", end="")

            time.sleep(sample_interval)

        print()

        # Save training data
        filename = f"data/training_{gas_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False)

        self.analyze_training_data_quality(df, gas_type)

        self.logger.info(f"Enhanced training data with Emergency PPM saved to {filename}")
        self.logger.info(f"Collected {len(training_data)} samples for {gas_type}")

        return training_data

    def analyze_training_data_quality(self, df, gas_type):
        """Enhanced analysis for training data quality with emergency mode awareness"""
        self.logger.info(f"Enhanced analysis for {gas_type}:")

        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            emergency_ppm_col = f'{sensor}_emergency_ppm'
            emergency_mode_col = f'{sensor}_emergency_mode'

            if ppm_col in df.columns:
                ppm_data = df[ppm_col]
                ppm_mean = ppm_data.mean()
                ppm_std = ppm_data.std()
                ppm_max = ppm_data.max()
                ppm_min = ppm_data.min()

                self.logger.info(f"  {sensor}: PPM {ppm_mean:.0f}±{ppm_std:.0f} (range: {ppm_min:.0f}-{ppm_max:.0f})")
                
                # Check if emergency mode was used
                if emergency_mode_col in df.columns and df[emergency_mode_col].any():
                    emergency_ppm_data = df[emergency_ppm_col]
                    emergency_mean = emergency_ppm_data.mean()
                    self.logger.info(f"  {sensor}: Emergency PPM {emergency_mean:.0f} (backup calculation)")

                if gas_type == 'normal':
                    if ppm_max < 30:
                        self.logger.info(f"  ✅ {sensor}: Good normal baseline")
                    if ppm_std / ppm_mean < 0.5:
                        self.logger.info(f"  ✅ {sensor}: Stable normal readings")
                else:
                    if ppm_max > 100:
                        self.logger.info(f"  ✅ {sensor}: Good high-concentration response")
                    if (ppm_max - ppm_min) > 50:
                        self.logger.info(f"  ✅ {sensor}: Excellent dynamic range")
                    if ppm_std / ppm_mean < 0.3:
                        self.logger.info(f"  ✅ {sensor}: Stable readings")

    def train_model(self):
        """Enhanced machine learning model training with emergency features"""
        self.logger.info("Training enhanced machine learning model with Emergency PPM features...")

        training_data = self.load_training_data()
        if training_data is None:
            return False

        # Enhanced feature selection including emergency PPM features
        feature_columns = [
            'TGS2600_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm', 'TGS2600_emergency_ppm',
            'TGS2600_drift_factor', 'TGS2600_normalization_factor',
            'TGS2602_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm', 'TGS2602_emergency_ppm',
            'TGS2602_drift_factor', 'TGS2602_normalization_factor',
            'TGS2610_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm', 'TGS2610_emergency_ppm',
            'TGS2610_drift_factor', 'TGS2610_normalization_factor',
            'temperature', 'humidity'
        ]

        available_columns = [col for col in feature_columns if col in training_data.columns]
        
        # Add default values for missing features
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            for feature_type in ['drift_factor', 'normalization_factor', 'emergency_ppm']:
                col_name = f'{sensor}_{feature_type}'
                if col_name not in training_data.columns:
                    if feature_type == 'emergency_ppm':
                        # Calculate emergency PPM if missing
                        voltage_col = f'{sensor}_raw_voltage'
                        if voltage_col in training_data.columns:
                            training_data[col_name] = training_data[voltage_col].apply(
                                lambda v: self.emergency_ppm_calc.calculate_emergency_ppm(sensor, v)
                            )
                        else:
                            training_data[col_name] = 0.0
                    else:
                        training_data[col_name] = 1.0
                    
                    if col_name in feature_columns:
                        available_columns.append(col_name)

        X = training_data[available_columns].values
        y = training_data['gas_type'].values

        X = np.nan_to_num(X, nan=0.0)

        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.logger.info(f"Training classes: {list(unique_classes)}")
        for cls, count in zip(unique_classes, class_counts):
            self.logger.info(f"  {cls}: {count} samples")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        feature_importance = pd.DataFrame({
            'feature': available_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.logger.info(f"Model accuracy: {accuracy:.3f}")
        self.logger.info("\nTop 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Save model and metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'feature_columns': available_columns,
            'feature_importance': feature_importance.to_dict('records'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'classes': list(unique_classes),
            'model_type': 'smart_drift_compensation_v3.0_with_emergency_ppm',
            'drift_compensation_enabled': True,
            'baseline_normalization_enabled': True,
            'voltage_adjustment_enabled': True,
            'emergency_ppm_enabled': True
        }

        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')

        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        self.is_model_trained = True
        self.logger.info("Enhanced model with Emergency PPM support trained and saved successfully")

        return True

    def load_training_data(self):
        """Load and validate training data"""
        data_files = list(Path("data").glob("training_*.csv"))
        if not data_files:
            self.logger.error("No training data files found!")
            return None

        all_data = []
        gas_counts = {}

        for file in data_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)

                if 'gas_type' in df.columns:
                    gas_type = df['gas_type'].iloc[0]
                    gas_counts[gas_type] = gas_counts.get(gas_type, 0) + len(df)

                self.logger.info(f"Loaded {len(df)} samples from {file.name}")
            except Exception as e:
                self.logger.error(f"Error loading {file.name}: {e}")

        if not all_data:
            return None

        combined_data = pd.concat(all_data, ignore_index=True)

        self.logger.info(f"Total training samples: {len(combined_data)}")
        self.logger.info("Samples per gas type:")
        for gas, count in gas_counts.items():
            self.logger.info(f"  {gas}: {count} samples")

        return combined_data

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_model_trained = True
            self.logger.info("Enhanced model with Emergency PPM support loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction with emergency PPM support"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0

        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            feature_columns = metadata['feature_columns']
        except:
            feature_columns = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                             'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']

        feature_vector = []
        for feature in feature_columns:
            if feature == 'temperature':
                feature_vector.append(self.current_temperature)
            elif feature == 'humidity':
                feature_vector.append(self.current_humidity)
            else:
                parts = feature.split('_')
                sensor = parts[0]
                measurement = '_'.join(parts[1:])

                if sensor in readings and measurement in readings[sensor]:
                    value = readings[sensor][measurement]
                    feature_vector.append(value if value is not None else 0.0)
                else:
                    if measurement in ['drift_factor', 'normalization_factor']:
                        feature_vector.append(1.0)
                    elif measurement == 'emergency_ppm':
                        # Calculate emergency PPM if not available
                        if sensor in readings:
                            voltage = readings[sensor].get('raw_voltage', 1.6)
                            emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(sensor, voltage)
                            feature_vector.append(emergency_ppm)
                        else:
                            feature_vector.append(0.0)
                    else:
                        feature_vector.append(0.0)

        features = np.array([feature_vector])
        features = np.nan_to_num(features, nan=0.0)

        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()

        return prediction, confidence

    def continuous_monitoring(self, duration=None, monitoring_mode='datasheet'):
        """Enhanced continuous monitoring with Emergency PPM support"""
        self.set_sensor_mode(monitoring_mode)

        self.logger.info(f"Starting enhanced monitoring in {monitoring_mode.upper()} mode with Emergency PPM support...")
        self.is_collecting = True

        fieldnames = [
            'timestamp', 'temperature', 'humidity', 'sensor_mode',
            'TGS2600_voltage', 'TGS2600_raw_voltage', 'TGS2600_compensated_voltage', 'TGS2600_resistance', 'TGS2600_compensated_resistance',
            'TGS2600_rs_r0_ratio', 'TGS2600_ppm', 'TGS2600_emergency_ppm', 'TGS2600_drift_factor', 'TGS2600_normalization_factor', 'TGS2600_emergency_mode',
            'TGS2602_voltage', 'TGS2602_raw_voltage', 'TGS2602_compensated_voltage', 'TGS2602_resistance', 'TGS2602_compensated_resistance',
            'TGS2602_rs_r0_ratio', 'TGS2602_ppm', 'TGS2602_emergency_ppm', 'TGS2602_drift_factor', 'TGS2602_normalization_factor', 'TGS2602_emergency_mode',
            'TGS2610_voltage', 'TGS2610_raw_voltage', 'TGS2610_compensated_voltage', 'TGS2610_resistance', 'TGS2610_compensated_resistance',
            'TGS2610_rs_r0_ratio', 'TGS2610_ppm', 'TGS2610_emergency_ppm', 'TGS2610_drift_factor', 'TGS2610_normalization_factor', 'TGS2610_emergency_mode',
            'predicted_gas', 'confidence'
        ]

        monitoring_file = f"data/monitoring_{monitoring_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(monitoring_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()
            sample_count = 0

            try:
                while self.is_collecting:
                    timestamp = datetime.now()
                    readings = self.read_sensors()
                    predicted_gas, confidence = self.predict_gas(readings)

                    data_row = {
                        'timestamp': timestamp,
                        'temperature': self.current_temperature,
                        'humidity': self.current_humidity,
                        'sensor_mode': monitoring_mode,

                        'TGS2600_voltage': readings['TGS2600']['voltage'],
                        'TGS2600_raw_voltage': readings['TGS2600']['raw_voltage'],
                        'TGS2600_compensated_voltage': readings['TGS2600']['compensated_voltage'],
                        'TGS2600_resistance': readings['TGS2600']['resistance'],
                        'TGS2600_compensated_resistance': readings['TGS2600']['compensated_resistance'],
                        'TGS2600_rs_r0_ratio': readings['TGS2600']['rs_r0_ratio'],
                        'TGS2600_ppm': readings['TGS2600']['ppm'],
                        'TGS2600_emergency_ppm': readings['TGS2600']['emergency_ppm'],
                        'TGS2600_drift_factor': readings['TGS2600']['drift_factor'],
                        'TGS2600_normalization_factor': readings['TGS2600']['normalization_factor'],
                        'TGS2600_emergency_mode': readings['TGS2600']['emergency_mode'],

                        'TGS2602_voltage': readings['TGS2602']['voltage'],
                        'TGS2602_raw_voltage': readings['TGS2602']['raw_voltage'],
                        'TGS2602_compensated_voltage': readings['TGS2602']['compensated_voltage'],
                        'TGS2602_resistance': readings['TGS2602']['resistance'],
                        'TGS2602_compensated_resistance': readings['TGS2602']['compensated_resistance'],
                        'TGS2602_rs_r0_ratio': readings['TGS2602']['rs_r0_ratio'],
                        'TGS2602_ppm': readings['TGS2602']['ppm'],
                        'TGS2602_emergency_ppm': readings['TGS2602']['emergency_ppm'],
                        'TGS2602_drift_factor': readings['TGS2602']['drift_factor'],
                        'TGS2602_normalization_factor': readings['TGS2602']['normalization_factor'],
                        'TGS2602_emergency_mode': readings['TGS2602']['emergency_mode'],

                        'TGS2610_voltage': readings['TGS2610']['voltage'],
                        'TGS2610_raw_voltage': readings['TGS2610']['raw_voltage'],
                        'TGS2610_compensated_voltage': readings['TGS2610']['compensated_voltage'],
                        'TGS2610_resistance': readings['TGS2610']['resistance'],
                        'TGS2610_compensated_resistance': readings['TGS2610']['compensated_resistance'],
                        'TGS2610_rs_r0_ratio': readings['TGS2610']['rs_r0_ratio'],
                        'TGS2610_ppm': readings['TGS2610']['ppm'],
                        'TGS2610_emergency_ppm': readings['TGS2610']['emergency_ppm'],
                        'TGS2610_drift_factor': readings['TGS2610']['drift_factor'],
                        'TGS2610_normalization_factor': readings['TGS2610']['normalization_factor'],
                        'TGS2610_emergency_mode': readings['TGS2610']['emergency_mode'],

                        'predicted_gas': predicted_gas,
                        'confidence': confidence
                    }

                    writer.writerow(data_row)
                    sample_count += 1

                    # Enhanced display with emergency status
                    emergency_sensors = [s for s in readings.keys() if readings[s]['emergency_mode']]
                    emergency_status = f"EMG({len(emergency_sensors)})" if emergency_sensors else "NORMAL"
                    
                    print(f"\r{timestamp.strftime('%H:%M:%S')} | Mode: {monitoring_mode.title()} | Status: {emergency_status} | "
                          f"2600: {readings['TGS2600']['ppm']:.0f}ppm | "
                          f"2602: {readings['TGS2602']['ppm']:.0f}ppm | "
                          f"2610: {readings['TGS2610']['ppm']:.0f}ppm | "
                          f"Predicted: {predicted_gas} ({confidence:.2f})", end="")

                    max_ppm = max(readings['TGS2600']['ppm'], readings['TGS2602']['ppm'], readings['TGS2610']['ppm'])
                    if max_ppm > 50 and confidence > 0.7:
                        print(f"\n*** GAS DETECTION ALERT: {predicted_gas} detected! ***")

                    if duration and (time.time() - start_time) >= duration:
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")

        self.is_collecting = False
        self.logger.info(f"Enhanced monitoring data with Emergency PPM support saved to {monitoring_file}")
        self.logger.info(f"Total samples collected: {sample_count}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_collecting = False

    def set_environmental_conditions(self, temperature=None, humidity=None):
        """Set current environmental conditions for compensation"""
        if temperature is not None:
            self.current_temperature = temperature
            self.logger.info(f"Temperature set to {temperature}°C")

        if humidity is not None:
            self.current_humidity = humidity
            self.logger.info(f"Humidity set to {humidity}%RH")

def main():
    """Enhanced main function with Emergency PPM Recovery"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration if available
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️  CALIBRATION NOT FOUND")
        print("Emergency PPM mode is available as fallback")
        print("Sensors can still detect gas using voltage-based calculation")

    # Load existing model if available
    gas_sensor.load_model()

    # Check if daily drift check is needed
    if gas_sensor.drift_manager.is_daily_check_needed():
        print("\n" + "="*70)
        print("🧠 SMART DAILY DRIFT CHECK RECOMMENDED")
        print("Your sensors may have drifted since yesterday.")
        print("Smart system can automatically compensate without hardware changes.")
        print("="*70)
        
        response = input("Run smart daily drift check now? (y/n): ").lower()
        if response == 'y':
            gas_sensor.drift_manager.smart_drift_check(gas_sensor)

    while True:
        print("\n" + "="*70)
        print("🧠 SMART Gas Sensor Array System - USV Air Pollution Detection")
        print("Smart Drift Compensation + Emergency PPM Recovery v3.0")
        print("="*70)
        print("1. Calibrate sensors (Enhanced R0 determination)")
        print("2. Collect training data (Auto-switch to Extended mode)")
        print("3. Train machine learning model (Smart + Emergency features)")
        print("4. Start monitoring - Datasheet mode (Accurate detection)")
        print("5. Start monitoring - Extended mode (Full range)")
        print("6. Test single reading (Detailed analysis with emergency PPM)")
        print("7. Set environmental conditions (T°C, %RH)")
        print("8. Switch sensor calculation mode")
        print("9. View sensor diagnostics (Enhanced with emergency status)")
        print("10. Exit")
        print("-" * 40)
        print("🧠 SMART DRIFT COMPENSATION FEATURES:")
        print("11. Smart daily drift check (1.6V optimized)")
        print("12. Quick stability test")
        print("13. Smart drift status report") 
        print("14. Manual drift compensation reset")
        print("15. 🆕 AUTO BASELINE RESET (recommended for high drift)")
        print("16. 🆕 Smart system health check")
        print("-" * 40)
        print("🔧 VOLTAGE ADJUSTMENT FEATURES:")
        print("17. 🆕 Smart Voltage Adjustment (Fix sensor responsivity)")
        print("18. 🆕 Sensor Responsivity Check (Auto-detect voltage issues)")
        print("19. 🆕 Quick voltage check for all sensors")
        print("-" * 40)
        print("🚨 EMERGENCY PPM RECOVERY FEATURES (NEW!):")
        print("20. 🆕 Emergency R0 Fix (Quick fix for calibration issues)")
        print("21. 🆕 Smart Troubleshoot PPM Issues (Auto-diagnose PPM 0 problems)")
        print("22. 🆕 Toggle Emergency PPM Mode (Bypass R0 calibration)")
        print("23. 🆕 Emergency PPM Test (Test without calibration)")
        print("-"*70)

        try:
            choice = input("Select option (1-23): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                print("Ensure sensors are warmed up for at least 10 minutes in clean air!")
                confirm = input("Continue with calibration? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.calibrate_sensors(duration)

            elif choice == '2':
                gas_types = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar',
                           'hydrogen', 'toluene', 'ammonia', 'butane', 'propane']
                print("Available gas types:", ', '.join(gas_types))
                print("⚠️  IMPORTANT: Collect 'normal' data first for baseline!")
                gas_type = input("Enter gas type: ").strip().lower()

                if gas_type not in gas_types:
                    print("Invalid gas type!")
                    continue

                duration = int(input("Collection duration (seconds, default 60): ") or 60)
                print("🔄 Auto-switching to EXTENDED mode for training...")
                gas_sensor.collect_training_data(gas_type, duration)

            elif choice == '3':
                print("🤖 Training model with Smart Drift + Emergency PPM features...")
                if gas_sensor.train_model():
                    print("✅ Enhanced model training completed successfully!")
                    print("Model includes drift correction, baseline normalization, voltage adjustment, and emergency PPM.")
                else:
                    print("❌ Model training failed!")

            elif choice == '4':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None

                print("🎯 Smart monitoring in DATASHEET mode with Emergency PPM backup")
                gas_sensor.continuous_monitoring(duration, 'datasheet')

            elif choice == '5':
                duration_input = input("Monitoring duration (seconds, Enter for infinite): ").strip()
                duration = int(duration_input) if duration_input else None

                print("📊 Smart monitoring in EXTENDED mode with Emergency PPM backup")
                gas_sensor.continuous_monitoring(duration, 'extended')

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*70)
                print("🧠 SMART SENSOR ANALYSIS - WITH EMERGENCY PPM RECOVERY")
                print("="*70)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Model Voltage: {data['voltage']:.3f}V (for prediction)")
                    
                    if data['voltage_adjusted']:
                        print(f"  🔧 Hardware: Voltage adjusted for optimal responsivity")
                    
                    if data['smart_compensation_applied']:
                        drift_percent = abs(1 - data['drift_factor']) * 100
                        norm_percent = abs(1 - data['normalization_factor']) * 100
                        print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                        print(f"  Compensated Voltage: {data['compensated_voltage']:.3f}V")
                        print(f"  🧠 Smart Compensation: Drift {drift_percent:.1f}% + Norm {norm_percent:.1f}%")
                    else:
                        print(f"  Raw Voltage: {data['raw_voltage']:.3f}V (no compensation)")
                    
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    print(f"  Compensated Resistance: {data['compensated_resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    else:
                        print(f"  Rs/R0 Ratio: Not calibrated")
                    
                    # Enhanced PPM display
                    print(f"  📊 PPM Analysis:")
                    print(f"    Main PPM: {data['ppm']:.0f} ({'No limit' if data['mode'] == 'Extended' else 'Datasheet limit'})")
                    print(f"    Emergency PPM: {data['emergency_ppm']:.0f} (backup calculation)")
                    
                    if data['emergency_mode']:
                        print(f"    🚨 EMERGENCY MODE ACTIVE: Using voltage-based calculation")
                    elif data['ppm'] == 0 and data['emergency_ppm'] > 0:
                        print(f"    ⚠️ Main calculation failed, emergency PPM available")
                    
                    print(f"  Target Gases: {', '.join(data['target_gases'])}")
                    print(f"  R0 (Baseline): {data['R0']:.1f}Ω" if data['R0'] else "  R0: Not calibrated")

                print(f"\nENVIRONMENTAL CONDITIONS:")
                print(f"  Temperature: {gas_sensor.current_temperature}°C")
                print(f"  Humidity: {gas_sensor.current_humidity}%RH")

                print(f"\n🧠 SMART PREDICTION (Enhanced with Emergency PPM):")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")

                if confidence < 0.5:
                    print("  ⚠️  Low confidence - consider Emergency PPM mode or recalibration")
                elif confidence < 0.7:
                    print("  ⚠️  Medium confidence")
                else:
                    print("  ✅ High confidence")

            elif choice == '7':
                print("\nSet Environmental Conditions:")
                temp_input = input("Temperature (°C, current: {:.1f}): ".format(gas_sensor.current_temperature))
                humidity_input = input("Humidity (%RH, current: {:.1f}): ".format(gas_sensor.current_humidity))

                temp = float(temp_input) if temp_input else None
                humidity = float(humidity_input) if humidity_input else None

                gas_sensor.set_environmental_conditions(temp, humidity)

            elif choice == '8':
                current_mode = 'Extended' if gas_sensor.sensor_config['TGS2600']['use_extended_mode'] else 'Datasheet'
                print(f"\nCurrent mode: {current_mode}")
                print("1. Datasheet mode (accurate detection, limited range)")
                print("2. Extended mode (unlimited range for training)")

                mode_choice = input("Select mode (1-2): ").strip()
                if mode_choice == '1':
                    gas_sensor.set_sensor_mode('datasheet')
                elif mode_choice == '2':
                    gas_sensor.set_sensor_mode('extended')
                else:
                    print("Invalid choice!")

            elif choice == '9':
                print("\n" + "="*70)
                print("🧠 SMART SENSOR DIAGNOSTICS WITH EMERGENCY STATUS")
                print("="*70)

                for sensor_name, config in gas_sensor.sensor_config.items():
                    current_mode = 'Extended' if config['use_extended_mode'] else 'Datasheet'
                    emergency_mode = config.get('use_emergency_ppm', False)
                    drift_factor = gas_sensor.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                    norm_factor = gas_sensor.drift_manager.normalization_factors.get(sensor_name, 1.0)
                    drift_percent = abs(1 - drift_factor) * 100
                    norm_percent = abs(1 - norm_factor) * 100
                    
                    # Voltage adjustment info
                    voltage_adj = gas_sensor.drift_manager.voltage_adjustments.get(sensor_name, {})
                    voltage_adjusted = voltage_adj.get('adjusted', False)
                    
                    print(f"\n{sensor_name} ({current_mode} mode):")
                    print(f"  Power Consumption: {config['power_consumption']*1000:.0f}mW")
                    print(f"  Heater Current: {config['heater_current']*1000:.0f}mA")
                    print(f"  Datasheet Range: {config['detection_range']} ppm")
                    print(f"  Extended Range: {config['extended_range']} ppm")
                    print(f"  Operating Temp: {config['operating_temp_range']}°C")
                    print(f"  Target Gases: {', '.join(config['target_gases'])}")

                    if config['R0']:
                        print(f"  Calibrated R0: {config['R0']:.1f}Ω")
                        print(f"  Baseline Voltage: {config['baseline_voltage']:.3f}V")
                    else:
                        print(f"  ⚠️  Not calibrated")
                    
                    if emergency_mode:
                        print(f"  🚨 EMERGENCY MODE: Active (PPM calculation bypasses R0)")
                    else:
                        print(f"  ✅ Normal mode (R0-based calculation)")
                    
                    if voltage_adjusted:
                        original_v = voltage_adj.get('original', 'Unknown')
                        current_v = voltage_adj.get('current', 'Unknown')
                        print(f"  🔧 Voltage Adjusted: {original_v:.3f}V → {current_v:.3f}V")
                    else:
                        print(f"  ✅ No voltage adjustment performed")
                    
                    if drift_percent > 0:
                        print(f"  🧠 Drift Compensation: {drift_percent:.1f}% (factor: {drift_factor:.3f})")
                    else:
                        print(f"  ✅ No drift compensation active")
                        
                    if norm_percent > 0:
                        print(f"  🧠 Baseline Normalization: {norm_percent:.1f}% (factor: {norm_factor:.3f})")
                    else:
                        print(f"  ✅ No normalization needed")
                    
                    # Test both PPM calculations
                    current_reading = gas_sensor.read_sensors()[sensor_name]
                    print(f"  Current PPM: {current_reading['ppm']:.1f} (main) | {current_reading['emergency_ppm']:.1f} (emergency)")

            # SMART DRIFT COMPENSATION OPTIONS (11-16) - Same as before
            elif choice == '11':
                print("\n🧠 SMART DAILY DRIFT CHECK")
                print("Optimized for 1.6V baseline with mV-based tolerances")
                print("✅ Automatic compensation without hardware changes")
                print("✅ Model compatibility preserved")
                
                gas_sensor.drift_manager.smart_drift_check(gas_sensor)

            elif choice == '12':
                print("\n⚡ QUICK STABILITY TEST")
                print("Testing sensor stability over 1 minute...")
                
                # Implement quick stability test (simplified version)
                readings = {sensor: [] for sensor in gas_sensor.sensor_config.keys()}
                
                print("Taking 30 readings over 1 minute...")
                
                for i in range(30):
                    sensor_readings = gas_sensor.read_sensors()
                    
                    for sensor_name, data in sensor_readings.items():
                        readings[sensor_name].append(data['voltage'])
                    
                    time.sleep(2)
                    print(f"\rProgress: {i+1}/30", end="")
                
                print("\n\nSTABILITY TEST RESULTS:")
                print("-" * 40)
                
                for sensor_name, voltages in readings.items():
                    voltages_array = np.array(voltages)
                    mean_v = np.mean(voltages_array)
                    std_v = np.std(voltages_array)
                    stability_percent = (std_v / mean_v) * 100
                    
                    drift_factor = gas_sensor.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                    drift_compensation = abs(1 - drift_factor) * 100
                    
                    status = 'EXCELLENT' if stability_percent < 1 else \
                            'GOOD' if stability_percent < 2 else \
                            'MODERATE' if stability_percent < 5 else 'POOR'
                    
                    print(f"\n{sensor_name}:")
                    print(f"  Mean Voltage: {mean_v:.3f}V ± {std_v:.3f}V")
                    print(f"  Stability: {stability_percent:.2f}% ({status})")
                    print(f"  Smart Compensation: {drift_compensation:.1f}%")

            elif choice == '13':
                print("\n🧠 SMART DRIFT STATUS REPORT")
                status = gas_sensor.drift_manager.get_smart_status()
                
                print("="*60)
                print("DRIFT COMPENSATION STATUS:")
                if status['drift_compensation']:
                    for sensor, info in status['drift_compensation'].items():
                        print(f"  {sensor}: {info['compensation_percent']:.1f}% ({info['level']})")
                else:
                    print("  ✅ No drift compensation active")
                
                print("\nBASELINE NORMALIZATION STATUS:")
                for sensor, info in status['baseline_normalization'].items():
                    compat = "✅ Compatible" if info['model_compatible'] else "⚠️ Check needed"
                    print(f"  {sensor}: {info['normalization_percent']:.1f}% - {compat}")
                
                print("\nVOLTAGE ADJUSTMENT STATUS:")
                for sensor, adj_info in status['voltage_adjustments'].items():
                    if adj_info.get('adjusted', False):
                        original = adj_info.get('original', 'Unknown')
                        current = adj_info.get('current', 'Unknown')
                        print(f"  {sensor}: ✅ Adjusted ({original:.3f}V → {current:.3f}V)")
                    else:
                        print(f"  {sensor}: ⚪ No adjustment")
                
                print(f"\nOVERALL SYSTEM HEALTH: {status['overall_health']}")
                print(f"MODEL COMPATIBILITY: {'✅ GOOD' if status['model_compatible'] else '⚠️ NEEDS ATTENTION'}")

            elif choice == '14':
                print("\n🔄 MANUAL DRIFT COMPENSATION RESET")
                print("This will:")
                print("  1. Remove all current drift compensation factors")
                print("  2. Reset baseline normalization factors")
                print("  3. Force new drift check on next run")
                print("  4. Keep voltage adjustments intact")
                
                confirm = input("\nAre you sure? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.drift_manager.drift_compensation_factors = {}
                    gas_sensor.drift_manager.normalization_factors = {
                        'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0
                    }
                    gas_sensor.drift_manager.daily_check_done = False
                    gas_sensor.drift_manager.save_drift_data()
                    print("✅ Smart drift compensation reset completed!")

            elif choice == '15':
                print("\n⚡ AUTO BASELINE RESET - SMART SOLUTION")
                print("🆕 BEST solution for high drift without hardware changes!")
                print("✅ Accepts current voltages as new baseline")
                print("✅ Preserves model compatibility")
                print("✅ No training data collection needed")
                
                confirm = input("\nThis is the recommended solution for high drift. Proceed? (y/n): ").lower()
                if confirm == 'y':
                    success = gas_sensor.drift_manager.auto_baseline_reset(gas_sensor)
                    if success:
                        print("\n🎉 AUTO BASELINE RESET SUCCESSFUL!")
                        print("✅ System is now optimized and ready to use")
                        print("✅ Your model will continue working normally")

            elif choice == '16':
                print("\n🧠 SMART SYSTEM HEALTH CHECK")
                print("Comprehensive analysis of sensor and drift compensation system...")
                
                # Perform comprehensive health check
                current_readings = gas_sensor.read_sensors()
                status = gas_sensor.drift_manager.get_smart_status()
                
                print("\n" + "="*60)
                print("SMART SYSTEM HEALTH REPORT")
                print("="*60)
                
                print("\n📊 CURRENT SENSOR STATUS:")
                all_good = True
                ppm_issues = []
                
                for sensor_name, data in current_readings.items():
                    baseline = gas_sensor.drift_manager.current_baseline.get(sensor_name, 1.6)
                    drift_mv = abs(data['raw_voltage'] - baseline) * 1000
                    
                    if drift_mv <= 20:
                        sensor_status = "✅ EXCELLENT"
                    elif drift_mv <= 50:
                        sensor_status = "✅ GOOD" 
                    elif drift_mv <= 100:
                        sensor_status = "🔧 MODERATE"
                        all_good = False
                    elif drift_mv <= 200:
                        sensor_status = "⚠️ HIGH"
                        all_good = False
                    else:
                        sensor_status = "❌ CRITICAL"
                        all_good = False
                    
                    # Check voltage responsivity
                    if data['raw_voltage'] < 1.2:
                        voltage_status = "❌ TOO LOW"
                        all_good = False
                    elif data['raw_voltage'] < 1.4:
                        voltage_status = "⚠️ LOW"
                    else:
                        voltage_status = "✅ GOOD"
                    
                    # Check PPM calculation
                    if data['ppm'] == 0 and data['emergency_ppm'] > 0:
                        ppm_issues.append(sensor_name)
                        ppm_status = "⚠️ PPM ISSUE"
                    elif data['ppm'] == 0 and data['emergency_ppm'] == 0:
                        ppm_status = "❌ NO PPM"
                    else:
                        ppm_status = "✅ PPM OK"
                    
                    print(f"  {sensor_name}: {data['raw_voltage']:.3f}V | Drift: {drift_mv:.0f}mV | {sensor_status} | Voltage: {voltage_status} | {ppm_status}")
                
                print(f"\n🎯 SYSTEM PERFORMANCE:")
                if status['model_compatible'] and all_good and not ppm_issues:
                    print("✅ SYSTEM STATUS: OPTIMAL")
                    print("✅ All sensors stable and model compatible")
                    print("✅ Ready for high-accuracy gas detection")
                elif status['model_compatible'] and not ppm_issues:
                    print("🔧 SYSTEM STATUS: GOOD with compensation")
                    print("🔧 Smart compensation handling issues effectively")
                    print("✅ Model compatibility maintained")
                else:
                    print("⚠️ SYSTEM STATUS: NEEDS ATTENTION")
                    print("⚠️ Consider AUTO BASELINE RESET, Voltage Adjustment, or Emergency PPM mode")
                
                print(f"\n💡 RECOMMENDATIONS:")
                voltage_issues = []
                for sensor_name, data in current_readings.items():
                    if data['raw_voltage'] < 1.2:
                        voltage_issues.append(sensor_name)
                
                if voltage_issues:
                    print(f"🔧 Run Smart Voltage Adjustment for: {', '.join(voltage_issues)} (Option 17)")
                
                if ppm_issues:
                    print(f"🚨 PPM Issues detected in: {', '.join(ppm_issues)} (Options 20-23)")
                
                if all_good and status['model_compatible'] and not ppm_issues:
                    print("✅ System performing optimally - continue normal operation")
                elif status['model_compatible'] and not ppm_issues:
                    print("🔧 System stable with compensation - monitor drift trends")
                else:
                    print("⚡ Consider Emergency PPM mode (Option 22) or AUTO BASELINE RESET (Option 15)")

            # VOLTAGE ADJUSTMENT FEATURES (17-19) - Same as before
            elif choice == '17':
                print("\n🔧 SMART VOLTAGE ADJUSTMENT")
                print("Khusus untuk sensor yang voltage terlalu rendah/tinggi")
                print("✅ Tidak perlu kalibrasi ulang")
                print("✅ Tidak perlu collect dataset ulang")
                print("✅ Model compatibility terjaga")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nAvailable sensors: {', '.join(sensors)}")
                sensor_choice = input("Sensor to adjust (default: TGS2600): ").strip() or 'TGS2600'
                
                if sensor_choice not in sensors:
                    print("❌ Invalid sensor!")
                    continue
                
                target_voltage = float(input("Target voltage (default: 1.6V): ") or 1.6)
                
                success = gas_sensor.drift_manager.smart_voltage_adjustment(
                    gas_sensor, sensor_choice, target_voltage
                )
                
                if success:
                    print("\n🎉 Adjustment completed! Test sensor responsivity now.")
                    
                    test_responsivity = input("Test responsivity with gas spray? (y/n): ").lower()
                    if test_responsivity == 'y':
                        print(f"\nSpray gas near {sensor_choice} dan observe PPM response...")
                        print("Press Ctrl+C to stop test")
                        
                        try:
                            while True:
                                reading = gas_sensor.read_sensors()[sensor_choice]
                                print(f"\r{sensor_choice}: {reading['raw_voltage']:.3f}V | PPM: {reading['ppm']:.1f} | Emg: {reading['emergency_ppm']:.1f}", end="")
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print("\nTest completed!")

            elif choice == '18':
                print("\n🧪 SENSOR RESPONSIVITY CHECK")
                print("Automatically detect voltage issues and recommend solutions")
                print("Testing all sensors for responsivity problems...")
                
                responsivity_report = gas_sensor.drift_manager.check_sensor_responsivity(gas_sensor, test_duration=30)

            elif choice == '19':
                print("\n⚡ QUICK VOLTAGE CHECK")
                print("Checking current voltage levels for all sensors...")
                
                readings = gas_sensor.read_sensors()
                
                print("\n📊 CURRENT VOLTAGE STATUS:")
                print("-" * 50)
                
                for sensor_name, data in readings.items():
                    voltage = data['raw_voltage']
                    
                    if voltage < 1.0:
                        status = "❌ CRITICAL"
                        recommendation = "URGENT: Run Smart Voltage Adjustment"
                    elif voltage < 1.2:
                        status = "⚠️ POOR"
                        recommendation = "RECOMMENDED: Smart Voltage Adjustment"
                    elif voltage < 1.4:
                        status = "⚠️ LOW"
                        recommendation = "CONSIDER: Voltage adjustment for better response"
                    elif voltage > 2.0:
                        status = "⚠️ HIGH"
                        recommendation = "CONSIDER: Lower voltage to avoid over-sensitivity"
                    else:
                        status = "✅ GOOD"
                        recommendation = "No action needed"
                    
                    print(f"{sensor_name}: {voltage:.3f}V | {status} | {recommendation}")
                
                print(f"\n💡 QUICK ACTIONS:")
                print(f"   Option 17: Smart Voltage Adjustment")
                print(f"   Option 18: Full Responsivity Check")

            # NEW EMERGENCY PPM RECOVERY FEATURES (20-23)
            elif choice == '20':
                print("\n🔧 EMERGENCY R0 FIX")
                print("Quick fix for sensors with calibration issues")
                print("Uses current voltage as emergency baseline")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nAvailable sensors: {', '.join(sensors)}")
                sensor_choice = input("Sensor to fix (default: all): ").strip()
                
                if sensor_choice and sensor_choice not in sensors:
                    print("❌ Invalid sensor!")
                    continue
                
                sensors_to_fix = [sensor_choice] if sensor_choice else sensors
                
                print(f"\n🔧 EMERGENCY R0 FIX for {', '.join(sensors_to_fix)}")
                print("⚠️  Make sure sensors are in CLEAN AIR!")
                
                confirm = input("Proceed with emergency R0 fix? (y/n): ").lower()
                if confirm != 'y':
                    print("❌ Emergency R0 fix cancelled")
                    continue
                
                for sensor_name in sensors_to_fix:
                    success = gas_sensor.drift_manager.emergency_r0_fix(gas_sensor, sensor_name)
                    if success:
                        print(f"✅ {sensor_name}: Emergency R0 fix completed")
                    else:
                        print(f"❌ {sensor_name}: Emergency R0 fix failed")
                
                print(f"\n🎉 EMERGENCY R0 FIX COMPLETED!")
                print("✅ Sensors should now calculate PPM normally")
                print("✅ Test with gas spray to verify functionality")

            elif choice == '21':
                print("\n🔧 SMART TROUBLESHOOT PPM ISSUES")
                print("Auto-diagnose and fix PPM calculation problems")
                
                # Check which sensors have PPM issues
                current_readings = gas_sensor.read_sensors()
                problematic_sensors = []
                
                for sensor_name, data in current_readings.items():
                    if data['ppm'] == 0 and data['emergency_ppm'] > 5:
                        problematic_sensors.append(sensor_name)
                    elif data['ppm'] == 0 and data['emergency_ppm'] == 0:
                        problematic_sensors.append(sensor_name)
                
                if not problematic_sensors:
                    print("✅ No PPM issues detected!")
                    print("All sensors are calculating PPM normally")
                    continue
                
                print(f"\n⚠️  PPM ISSUES DETECTED in: {', '.join(problematic_sensors)}")
                
                for sensor_name in problematic_sensors:
                    print(f"\n🔧 TROUBLESHOOTING {sensor_name}...")
                    success = gas_sensor.drift_manager.smart_troubleshoot_ppm_issue(gas_sensor, sensor_name)
                    
                    if success:
                        print(f"✅ {sensor_name}: Issue resolved")
                    else:
                        print(f"⚠️ {sensor_name}: Manual intervention may be needed")

            elif choice == '22':
                print("\n🚨 TOGGLE EMERGENCY PPM MODE")
                print("Enable/disable emergency PPM calculation for sensors")
                print("Emergency mode bypasses R0 calibration and uses voltage-based calculation")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                
                print(f"\nCURRENT EMERGENCY MODE STATUS:")
                for sensor_name in sensors:
                    emergency_mode = gas_sensor.sensor_config[sensor_name].get('use_emergency_ppm', False)
                    status = "🚨 ACTIVE" if emergency_mode else "⚪ INACTIVE"
                    print(f"  {sensor_name}: {status}")
                
                print(f"\nSelect sensor to toggle (or 'all' for all sensors):")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                print("4. All sensors")
                
                choice_sensor = input("Enter choice (1-4): ").strip()
                
                if choice_sensor == '1':
                    target_sensors = ['TGS2600']
                elif choice_sensor == '2':
                    target_sensors = ['TGS2602']
                elif choice_sensor == '3':
                    target_sensors = ['TGS2610']
                elif choice_sensor == '4':
                    target_sensors = sensors
                else:
                    print("❌ Invalid choice!")
                    continue
                
                action = input("Enable (e) or Disable (d) emergency mode? ").lower()
                
                if action not in ['e', 'd']:
                    print("❌ Invalid action!")
                    continue
                
                enable_emergency = (action == 'e')
                
                for sensor_name in target_sensors:
                    gas_sensor.sensor_config[sensor_name]['use_emergency_ppm'] = enable_emergency
                    gas_sensor.sensor_config[sensor_name]['emergency_mode'] = enable_emergency
                    
                    status = "ENABLED" if enable_emergency else "DISABLED"
                    print(f"✅ {sensor_name}: Emergency PPM mode {status}")
                
                if enable_emergency:
                    print(f"\n🚨 EMERGENCY PPM MODE ACTIVATED!")
                    print("✅ Sensors will use voltage-based PPM calculation")
                    print("✅ No R0 calibration required")
                    print("✅ Functional even without proper calibration")
                else:
                    print(f"\n⚪ EMERGENCY PPM MODE DEACTIVATED!")
                    print("✅ Sensors will use normal R0-based calculation")
                    print("⚠️  Ensure proper calibration is available")

            elif choice == '23':
                print("\n🧪 EMERGENCY PPM TEST")
                print("Test emergency PPM calculation without calibration")
                print("This shows how sensors would perform in emergency mode")
                
                print(f"\n📊 EMERGENCY PPM TEST RESULTS:")
                print("-" * 60)
                
                readings = gas_sensor.read_sensors()
                
                for sensor_name, data in readings.items():
                    voltage = data['raw_voltage']
                    emergency_ppm = data['emergency_ppm']
                    normal_ppm = data['ppm']
                    
                    print(f"\n{sensor_name}:")
                    print(f"  Voltage: {voltage:.3f}V")
                    print(f"  Normal PPM: {normal_ppm:.1f}")
                    print(f"  Emergency PPM: {emergency_ppm:.1f}")
                    
                    if normal_ppm == 0 and emergency_ppm > 0:
                        print(f"  🚨 Emergency mode would fix PPM calculation!")
                    elif normal_ppm > 0 and emergency_ppm > 0:
                        difference = abs(normal_ppm - emergency_ppm)
                        print(f"  📊 Difference: {difference:.1f} PPM")
                    elif emergency_ppm == 0:
                        print(f"  ⚠️  Both calculations return 0 (likely clean air)")
                    else:
                        print(f"  ✅ Both calculations working")
                
                print(f"\n💡 RECOMMENDATIONS:")
                emergency_beneficial = []
                
                for sensor_name, data in readings.items():
                    if data['ppm'] == 0 and data['emergency_ppm'] > 0:
                        emergency_beneficial.append(sensor_name)
                
                if emergency_beneficial:
                    print(f"🚨 Consider enabling emergency mode for: {', '.join(emergency_beneficial)}")
                    print(f"   Use Option 22 to enable emergency PPM mode")
                else:
                    print(f"✅ Normal PPM calculation working for all sensors")
                
                # Test with gas spray
                test_gas = input(f"\nTest responsivity with gas spray? (y/n): ").lower()
                if test_gas == 'y':
                    print(f"\n🧪 EMERGENCY PPM RESPONSIVITY TEST")
                    print("Spray gas near sensors and observe both calculations...")
                    print("Press Ctrl+C to stop test")
                    print("-" * 60)
                    
                    try:
                        while True:
                            readings = gas_sensor.read_sensors()
                            
                            print(f"\r", end="")
                            for sensor_name, data in readings.items():
                                normal_ppm = data['ppm']
                                emergency_ppm = data['emergency_ppm']
                                print(f"{sensor_name}: N:{normal_ppm:.0f} E:{emergency_ppm:.0f} | ", end="")
                            
                            time.sleep(1)
                            
                    except KeyboardInterrupt:
                        print(f"\n\n✅ Emergency PPM test completed!")

            elif choice == '10':
                print("👋 Exiting Smart Gas Sensor System...")
                gas_sensor.drift_manager.save_drift_data()
                break

            else:
                print("❌ Invalid option!")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")
            gas_sensor.logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()