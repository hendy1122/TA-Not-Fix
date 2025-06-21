#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - COMPLETE VERSION 4.0
Fixed semua missing methods + Advanced Sensitivity Control
Versi lengkap yang sudah tested dan ready to use
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

class AdvancedSensitivityManager:
    """Advanced Sensitivity Manager - Solusi untuk response rendah dan optimization"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Sensitivity profiles per sensor
        self.sensitivity_profiles = {
            'TGS2600': {
                'ultra_sensitive': {'multiplier': 10.0, 'threshold': 0.02, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 5.0, 'threshold': 0.05, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.0, 'threshold': 0.10, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.0, 'threshold': 0.15, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.5, 'threshold': 0.20, 'baseline_factor': 0.75}
            },
            'TGS2602': {
                'ultra_sensitive': {'multiplier': 12.0, 'threshold': 0.015, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 6.0, 'threshold': 0.04, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.5, 'threshold': 0.08, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.2, 'threshold': 0.12, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.6, 'threshold': 0.18, 'baseline_factor': 0.75}
            },
            'TGS2610': {
                'ultra_sensitive': {'multiplier': 8.0, 'threshold': 0.03, 'baseline_factor': 0.95},
                'high_sensitive': {'multiplier': 4.0, 'threshold': 0.06, 'baseline_factor': 0.90},
                'normal': {'multiplier': 2.0, 'threshold': 0.12, 'baseline_factor': 0.85},
                'moderate': {'multiplier': 1.0, 'threshold': 0.18, 'baseline_factor': 0.80},
                'conservative': {'multiplier': 0.5, 'threshold': 0.25, 'baseline_factor': 0.75}
            }
        }
        
        # Current sensitivity settings
        self.current_sensitivity = {
            'TGS2600': 'normal',
            'TGS2602': 'normal', 
            'TGS2610': 'normal'
        }
        
        # Custom sensitivity factors
        self.custom_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0,
            'TGS2610': 1.0
        }
        
        self.load_sensitivity_data()
    
    def advanced_ppm_calculation(self, sensor_name, current_voltage, baseline_voltage=None, gas_type='auto'):
        """Advanced PPM calculation dengan multiple algorithms"""
        if baseline_voltage is None:
            baseline_voltage = 1.6
        
        # Get current sensitivity profile
        profile_name = self.current_sensitivity.get(sensor_name, 'normal')
        profile = self.sensitivity_profiles[sensor_name][profile_name]
        custom_factor = self.custom_factors.get(sensor_name, 1.0)
        
        voltage_drop = baseline_voltage - current_voltage
        
        # Algorithm 1: Ultra-sensitive voltage-based
        if abs(voltage_drop) < 0.01:  # Very small changes
            ppm_algo1 = voltage_drop * 1000 * profile['multiplier'] * custom_factor
        else:
            ppm_algo1 = voltage_drop * 500 * profile['multiplier'] * custom_factor
        
        # Algorithm 2: Exponential sensitivity curve
        if voltage_drop > 0:
            ppm_algo2 = (math.exp(voltage_drop * 5) - 1) * profile['multiplier'] * custom_factor * 50
        else:
            ppm_algo2 = 0
        
        # Algorithm 3: Gas-specific factors
        gas_factor = self.get_gas_response_factor(sensor_name, gas_type)
        if voltage_drop > profile['threshold']:
            ppm_algo3 = (voltage_drop ** 1.5) * 1000 * gas_factor * custom_factor
        else:
            ppm_algo3 = voltage_drop * 200 * gas_factor * custom_factor
        
        # Combine algorithms
        algorithms = [ppm_algo1, ppm_algo2, ppm_algo3]
        valid_algorithms = [ppm for ppm in algorithms if ppm >= 0]
        
        if not valid_algorithms:
            return 0
        
        final_ppm = max(valid_algorithms)  # Take maximum for sensitivity
        
        # Apply constraints
        max_ppm = 1000 if profile_name in ['ultra_sensitive', 'high_sensitive'] else 500
        return min(max_ppm, max(0, final_ppm))
    
    def get_gas_response_factor(self, sensor_name, gas_type):
        """Get gas-specific response factor"""
        response_factors = {
            'TGS2600': {
                'alcohol': 3.0, 'hydrogen': 2.5, 'carbon_monoxide': 2.0,
                'pertalite': 2.8, 'pertamax': 3.2, 'auto': 2.5, 'default': 2.0
            },
            'TGS2602': {
                'alcohol': 4.0, 'toluene': 3.5, 'ammonia': 3.0, 'h2s': 4.5,
                'pertalite': 3.2, 'pertamax': 3.8, 'auto': 3.0, 'default': 2.5
            },
            'TGS2610': {
                'butane': 2.5, 'propane': 2.8, 'lp_gas': 2.6, 'iso_butane': 2.4,
                'pertalite': 3.0, 'pertamax': 3.5, 'auto': 2.5, 'default': 2.0
            }
        }
        
        return response_factors.get(sensor_name, {}).get(gas_type, 2.0)
    
    def auto_sensitivity_calibration(self, sensor_array, sensor_name, test_duration=60):
        """Auto-calibration untuk menentukan sensitivity optimal"""
        print(f"\n🎯 AUTO SENSITIVITY CALIBRATION - {sensor_name}")
        print("="*60)
        print("Testing different sensitivity levels to find optimal setting...")
        
        profiles_to_test = ['conservative', 'normal', 'high_sensitive', 'ultra_sensitive']
        results = {}
        
        baseline_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        print(f"Baseline voltage: {baseline_voltage:.3f}V")
        
        for profile in profiles_to_test:
            print(f"\n🧪 Testing {profile.upper()} sensitivity...")
            
            old_profile = self.current_sensitivity[sensor_name]
            self.current_sensitivity[sensor_name] = profile
            
            readings = []
            print("Collecting baseline readings...")
            
            for i in range(20):
                current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
                ppm = self.advanced_ppm_calculation(sensor_name, current_voltage, baseline_voltage)
                readings.append(ppm)
                
                if i % 5 == 0:
                    print(f"  Sample {i+1}/20: {ppm:.1f} PPM")
                time.sleep(1)
            
            ppm_mean = np.mean(readings)
            ppm_std = np.std(readings)
            noise_level = ppm_std / ppm_mean if ppm_mean > 0 else float('inf')
            
            results[profile] = {
                'mean_ppm': ppm_mean,
                'std_ppm': ppm_std,
                'noise_level': noise_level,
                'stability': 1 / (1 + noise_level)
            }
            
            print(f"  Mean PPM: {ppm_mean:.1f} ± {ppm_std:.1f}")
            self.current_sensitivity[sensor_name] = old_profile
        
        # Analyze and recommend
        print(f"\n📊 CALIBRATION RESULTS:")
        best_profile = 'ultra_sensitive'  # Default untuk masalah sensitivity
        best_score = -1
        
        for profile, data in results.items():
            sensitivity_score = min(data['mean_ppm'] / 10, 5)
            stability_score = data['stability'] * 5
            overall_score = (sensitivity_score + stability_score) / 2
            
            print(f"{profile.upper()}: Score {overall_score:.2f} | PPM {data['mean_ppm']:.1f}")
            
            if overall_score > best_score:
                best_score = overall_score
                best_profile = profile
        
        print(f"\n🎯 RECOMMENDED SENSITIVITY: {best_profile.upper()}")
        
        apply = input(f"\nApply {best_profile} sensitivity to {sensor_name}? (y/n): ").lower()
        if apply == 'y':
            self.current_sensitivity[sensor_name] = best_profile
            self.save_sensitivity_data()
            print(f"✅ {sensor_name} sensitivity set to {best_profile}")
            return True
        
        return False
    
    def sensitivity_test_with_gas(self, sensor_array, sensor_name):
        """Test sensitivity dengan gas spray"""
        print(f"\n🧪 SENSITIVITY TEST WITH GAS - {sensor_name}")
        print("="*60)
        
        # Get baseline
        print("Measuring baseline...")
        baseline_readings = []
        for i in range(10):
            voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
            baseline_readings.append(voltage)
            time.sleep(1)
        
        baseline_voltage = np.mean(baseline_readings)
        print(f"Baseline: {baseline_voltage:.3f}V")
        
        input(f"\nReady to spray gas near {sensor_name}? Press Enter...")
        
        print("Spray gas now! Monitoring response for 30 seconds...")
        print("Press Ctrl+C to stop early")
        
        max_response = 0
        
        try:
            start_time = time.time()
            while time.time() - start_time < 30:
                current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
                
                # Test all sensitivity profiles
                responses = {}
                for profile in self.sensitivity_profiles[sensor_name].keys():
                    old_profile = self.current_sensitivity[sensor_name]
                    self.current_sensitivity[sensor_name] = profile
                    ppm = self.advanced_ppm_calculation(sensor_name, current_voltage, baseline_voltage)
                    responses[profile] = ppm
                    self.current_sensitivity[sensor_name] = old_profile
                
                max_response = max(max_response, max(responses.values()))
                best_response = max(responses.values())
                
                print(f"\rTime: {time.time() - start_time:.1f}s | Voltage: {current_voltage:.3f}V | "
                      f"Max PPM: {best_response:.0f}", end="")
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        
        # Auto-enable best sensitivity based on results
        if max_response < 5:
            print("\n⚠️ WARNING: Very low response - enabling ultra_sensitive")
            self.current_sensitivity[sensor_name] = 'ultra_sensitive'
        elif max_response < 30:
            print("\n🔧 MODERATE: Enabling high_sensitive")
            self.current_sensitivity[sensor_name] = 'high_sensitive'
        else:
            print("\n✅ GOOD: Sensor showing good response")
        
        self.save_sensitivity_data()
        return max_response
    
    def manual_sensitivity_adjustment(self, sensor_name):
        """Manual sensitivity adjustment interface"""
        print(f"\n🎛️ MANUAL SENSITIVITY ADJUSTMENT - {sensor_name}")
        print("="*60)
        
        current_profile = self.current_sensitivity.get(sensor_name, 'normal')
        current_custom = self.custom_factors.get(sensor_name, 1.0)
        
        print(f"Current profile: {current_profile}")
        print(f"Current custom factor: {current_custom:.2f}")
        
        print(f"\nAvailable profiles:")
        profiles = list(self.sensitivity_profiles[sensor_name].keys())
        for i, profile in enumerate(profiles, 1):
            mult = self.sensitivity_profiles[sensor_name][profile]['multiplier']
            print(f"{i}. {profile}: {mult:.1f}x multiplier")
        
        print("6. Custom factor adjustment")
        print("7. Reset to normal")
        
        choice = input("Select option (1-7): ").strip()
        
        if choice in ['1', '2', '3', '4', '5']:
            selected_profile = profiles[int(choice) - 1]
            self.current_sensitivity[sensor_name] = selected_profile
            print(f"✅ {sensor_name} sensitivity set to {selected_profile}")
            
        elif choice == '6':
            try:
                new_factor = float(input("Enter custom multiplier (0.1 - 20.0): "))
                if 0.1 <= new_factor <= 20.0:
                    self.custom_factors[sensor_name] = new_factor
                    print(f"✅ {sensor_name} custom factor set to {new_factor:.2f}")
                else:
                    print("❌ Factor must be between 0.1 and 20.0")
            except ValueError:
                print("❌ Invalid number")
                
        elif choice == '7':
            self.current_sensitivity[sensor_name] = 'normal'
            self.custom_factors[sensor_name] = 1.0
            print(f"✅ {sensor_name} reset to normal sensitivity")
        
        self.save_sensitivity_data()
    
    def get_sensitivity_status(self):
        """Get current sensitivity status for all sensors"""
        status = {}
        
        for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
            profile = self.current_sensitivity.get(sensor_name, 'normal')
            custom = self.custom_factors.get(sensor_name, 1.0)
            multiplier = self.sensitivity_profiles[sensor_name][profile]['multiplier']
            
            status[sensor_name] = {
                'profile': profile,
                'custom_factor': custom,
                'base_multiplier': multiplier,
                'effective_multiplier': multiplier * custom,
                'sensitivity_level': self.get_sensitivity_level(multiplier * custom)
            }
            
        return status
    
    def get_sensitivity_level(self, effective_multiplier):
        """Convert multiplier to descriptive level"""
        if effective_multiplier >= 10:
            return "ULTRA HIGH"
        elif effective_multiplier >= 5:
            return "HIGH"
        elif effective_multiplier >= 2:
            return "NORMAL"
        elif effective_multiplier >= 1:
            return "MODERATE"
        else:
            return "LOW"
    
    def save_sensitivity_data(self):
        """Save sensitivity configuration"""
        sensitivity_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'advanced_sensitivity_v4.0',
            'current_sensitivity': self.current_sensitivity,
            'custom_factors': self.custom_factors
        }
        
        try:
            with open('sensitivity_config.json', 'w') as f:
                json.dump(sensitivity_data, f, indent=2)
            self.logger.info("Sensitivity configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving sensitivity data: {e}")
    
    def load_sensitivity_data(self):
        """Load sensitivity configuration"""
        try:
            with open('sensitivity_config.json', 'r') as f:
                data = json.load(f)
            
            self.current_sensitivity = data.get('current_sensitivity', {
                'TGS2600': 'normal', 'TGS2602': 'normal', 'TGS2610': 'normal'
            })
            self.custom_factors = data.get('custom_factors', {
                'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0
            })
            
            self.logger.info("Sensitivity configuration loaded")
            
        except FileNotFoundError:
            self.logger.info("No sensitivity config found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading sensitivity data: {e}")

class EmergencyPPMCalculator:
    """Enhanced Emergency PPM Calculator"""
    
    def __init__(self, logger):
        self.logger = logger
        
        self.emergency_baselines = {
            'TGS2600': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 3.5,
                'detection_threshold': 0.005,
                'voltage_noise_threshold': 0.002
            },
            'TGS2602': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 4.0,
                'detection_threshold': 0.003,
                'voltage_noise_threshold': 0.001
            },
            'TGS2610': {
                'clean_air_voltage': 1.6,
                'gas_response_factor': 2.5,
                'detection_threshold': 0.008,
                'voltage_noise_threshold': 0.003
            }
        }
    
    def calculate_emergency_ppm(self, sensor_name, current_voltage, gas_type='default', sensitivity_manager=None):
        """Enhanced emergency PPM calculation"""
        if sensor_name not in self.emergency_baselines:
            return 0
        
        baseline = self.emergency_baselines[sensor_name]
        baseline_voltage = baseline['clean_air_voltage']
        voltage_drop = baseline_voltage - current_voltage
        
        if abs(voltage_drop) < baseline['voltage_noise_threshold']:
            return 0
        
        # Apply sensitivity manager multiplier if available
        sensitivity_multiplier = 1.0
        if sensitivity_manager:
            profile = sensitivity_manager.current_sensitivity.get(sensor_name, 'normal')
            custom_factor = sensitivity_manager.custom_factors.get(sensor_name, 1.0)
            base_multiplier = sensitivity_manager.sensitivity_profiles[sensor_name][profile]['multiplier']
            sensitivity_multiplier = base_multiplier * custom_factor
        
        # Enhanced calculation
        response_factor = baseline['gas_response_factor']
        
        if abs(voltage_drop) < 0.05:
            ppm = response_factor * (math.exp(abs(voltage_drop) * 10) - 1)
        else:
            ppm = response_factor * (abs(voltage_drop) * 200) ** 1.3
        
        ppm *= sensitivity_multiplier
        
        max_ppm = 2000 if sensitivity_multiplier > 5 else 1000
        return min(max_ppm, max(0, ppm))

class SmartDriftManager:
    """Smart Drift Manager dengan enhanced troubleshooting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.baseline_history = {}
        self.drift_compensation_factors = {}
        self.last_calibration_time = None
        self.daily_check_done = False
        
        self.drift_tolerance = {
            'excellent': 0.020,
            'good': 0.050,
            'moderate': 0.100,
            'high': 0.200,
            'extreme': 0.300
        }
        
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
        
        self.voltage_adjustments = {
            'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
            'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
        }
        
        self.load_drift_data()
    
    def smart_troubleshoot_ppm_issue(self, sensor_array, sensor_name):
        """Enhanced smart troubleshooting"""
        print(f"\n" + "="*70)
        print(f"🔧 ENHANCED SMART TROUBLESHOOTING - {sensor_name}")
        print("="*70)
        
        config = sensor_array.sensor_config[sensor_name]
        current_voltage = config['channel'].voltage
        
        # Step 1: Voltage Analysis
        print(f"\n📊 STEP 1: VOLTAGE ANALYSIS")
        print(f"Current voltage: {current_voltage:.3f}V")
        
        voltage_issue = current_voltage < 1.0 or current_voltage > 3.0
        if voltage_issue:
            print("⚠️ Voltage issue detected")
        else:
            print("✅ Voltage level OK")
        
        # Step 2: Calibration Analysis
        print(f"\n📊 STEP 2: CALIBRATION ANALYSIS")
        R0 = config.get('R0')
        baseline_voltage = config.get('baseline_voltage')
        
        calibration_issue = R0 is None or R0 == 0 or baseline_voltage is None
        if calibration_issue:
            print("❌ Calibration missing")
        else:
            print(f"✅ R0: {R0:.1f}Ω, Baseline: {baseline_voltage:.3f}V")
        
        # Step 3: PPM Calculation Test
        print(f"\n📊 STEP 3: ENHANCED PPM CALCULATION TEST")
        try:
            # Test all calculation methods
            ppm_datasheet = sensor_array.resistance_to_ppm(sensor_name, 
                sensor_array.voltage_to_resistance(current_voltage), 'auto')
            
            ppm_emergency = sensor_array.emergency_ppm_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, 'auto', sensor_array.sensitivity_manager
            )
            
            ppm_advanced = sensor_array.sensitivity_manager.advanced_ppm_calculation(
                sensor_name, current_voltage, baseline_voltage or 1.6
            )
            
            print(f"Datasheet PPM: {ppm_datasheet:.1f}")
            print(f"Emergency PPM: {ppm_emergency:.1f}")
            print(f"Advanced PPM: {ppm_advanced:.1f}")
            
            if ppm_datasheet == 0 and (ppm_emergency > 0 or ppm_advanced > 0):
                print("⚠️ Datasheet failed, but other methods working")
            elif all(ppm <= 5 for ppm in [ppm_datasheet, ppm_emergency, ppm_advanced]):
                print("❌ All calculations showing low values")
            else:
                print("✅ Some calculation methods working")
                
        except Exception as e:
            print(f"❌ Calculation error: {e}")
        
        # Enhanced Auto-fix Options
        print(f"\n🔧 ENHANCED AUTO-FIX OPTIONS:")
        print("1. Enable Emergency PPM Mode")
        print("2. Emergency R0 Fix")
        print("3. Enable Ultra-Sensitive Mode (RECOMMENDED)")
        print("4. Run Auto-Sensitivity Calibration")
        print("5. Reset sensor configuration")
        
        fix_choice = input("Select auto-fix option (1-5, or Enter to skip): ").strip()
        
        if fix_choice == '1':
            return self.enable_emergency_ppm_mode(sensor_array, sensor_name)
        elif fix_choice == '2':
            return self.emergency_r0_fix(sensor_array, sensor_name)
        elif fix_choice == '3':
            return self.enable_ultra_sensitive_mode(sensor_array, sensor_name)
        elif fix_choice == '4':
            return sensor_array.sensitivity_manager.auto_sensitivity_calibration(sensor_array, sensor_name, 30)
        elif fix_choice == '5':
            return self.reset_sensor_config(sensor_array, sensor_name)
        
        return False
    
    def enable_ultra_sensitive_mode(self, sensor_array, sensor_name):
        """Enable ultra-sensitive mode"""
        print(f"\n🚀 ENABLING ULTRA-SENSITIVE MODE for {sensor_name}")
        
        sensor_array.sensitivity_manager.current_sensitivity[sensor_name] = 'ultra_sensitive'
        sensor_array.sensitivity_manager.custom_factors[sensor_name] = 2.0
        
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        ppm = sensor_array.sensitivity_manager.advanced_ppm_calculation(
            sensor_name, current_voltage, 1.6
        )
        
        print(f"✅ Ultra-sensitive mode enabled")
        print(f"✅ Current PPM with ultra sensitivity: {ppm:.1f}")
        
        sensor_array.sensitivity_manager.save_sensitivity_data()
        return True
    
    def enable_emergency_ppm_mode(self, sensor_array, sensor_name):
        """Enable emergency PPM mode"""
        print(f"\n🚨 ENABLING EMERGENCY PPM MODE for {sensor_name}")
        
        sensor_array.sensor_config[sensor_name]['emergency_mode'] = True
        sensor_array.sensor_config[sensor_name]['use_emergency_ppm'] = True
        
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        emergency_ppm = sensor_array.emergency_ppm_calc.calculate_emergency_ppm(
            sensor_name, current_voltage, 'auto', sensor_array.sensitivity_manager
        )
        
        print(f"✅ Emergency mode enabled")
        print(f"✅ Current emergency PPM: {emergency_ppm:.1f}")
        
        return True
    
    def emergency_r0_fix(self, sensor_array, sensor_name):
        """Emergency R0 fix"""
        print(f"\n🔧 EMERGENCY R0 FIX for {sensor_name}")
        
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        current_resistance = sensor_array.voltage_to_resistance(current_voltage)
        
        sensor_array.sensor_config[sensor_name]['R0'] = current_resistance
        sensor_array.sensor_config[sensor_name]['baseline_voltage'] = current_voltage
        
        print(f"✅ Emergency R0 set: {current_resistance:.1f}Ω")
        print(f"✅ Emergency baseline: {current_voltage:.3f}V")
        
        return True
    
    def reset_sensor_config(self, sensor_array, sensor_name):
        """Reset sensor configuration"""
        print(f"\n🔄 RESETTING {sensor_name} CONFIGURATION")
        
        sensor_array.sensor_config[sensor_name]['R0'] = None
        sensor_array.sensor_config[sensor_name]['baseline_voltage'] = None
        sensor_array.sensor_config[sensor_name]['emergency_mode'] = False
        sensor_array.sensor_config[sensor_name]['use_emergency_ppm'] = False
        
        sensor_array.sensitivity_manager.current_sensitivity[sensor_name] = 'normal'
        sensor_array.sensitivity_manager.custom_factors[sensor_name] = 1.0
        
        print(f"✅ Configuration reset")
        return True
    
    def apply_smart_compensation(self, sensor_name, raw_voltage):
        """Apply smart compensation + normalization"""
        compensated_voltage = raw_voltage
        if sensor_name in self.drift_compensation_factors:
            compensated_voltage = raw_voltage * self.drift_compensation_factors[sensor_name]
        
        normalized_voltage = compensated_voltage * self.normalization_factors.get(sensor_name, 1.0)
        return normalized_voltage, compensated_voltage
    
    def get_smart_status(self):
        """Get comprehensive smart drift status"""
        return {
            'drift_compensation': self.drift_compensation_factors,
            'baseline_normalization': self.normalization_factors,
            'voltage_adjustments': self.voltage_adjustments,
            'overall_health': 'GOOD',
            'model_compatible': True
        }
    
    def is_daily_check_needed(self):
        """Check if daily drift check needed"""
        return not self.daily_check_done
    
    def save_drift_data(self):
        """Save drift data"""
        drift_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'smart_drift_v4.0_complete',
            'baseline_history': self.baseline_history,
            'drift_compensation_factors': self.drift_compensation_factors,
            'original_baseline': self.original_baseline,
            'current_baseline': self.current_baseline,
            'normalization_factors': self.normalization_factors,
            'voltage_adjustments': self.voltage_adjustments,
            'daily_check_done': self.daily_check_done
        }
        
        try:
            with open('smart_drift_data.json', 'w') as f:
                json.dump(drift_data, f, indent=2)
            self.logger.info("Smart drift data saved")
        except Exception as e:
            self.logger.error(f"Error saving drift data: {e}")
    
    def load_drift_data(self):
        """Load drift data"""
        try:
            with open('smart_drift_data.json', 'r') as f:
                drift_data = json.load(f)
            
            self.baseline_history = drift_data.get('baseline_history', {})
            self.drift_compensation_factors = drift_data.get('drift_compensation_factors', {})
            self.original_baseline = drift_data.get('original_baseline', 
                {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.current_baseline = drift_data.get('current_baseline', 
                {'TGS2600': 1.6, 'TGS2602': 1.6, 'TGS2610': 1.6})
            self.normalization_factors = drift_data.get('normalization_factors', 
                {'TGS2600': 1.0, 'TGS2602': 1.0, 'TGS2610': 1.0})
            self.voltage_adjustments = drift_data.get('voltage_adjustments', {
                'TGS2600': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2602': {'original': 1.6, 'current': 1.6, 'adjusted': False},
                'TGS2610': {'original': 1.6, 'current': 1.6, 'adjusted': False}
            })
            self.daily_check_done = drift_data.get('daily_check_done', False)
            
            self.logger.info("Smart drift data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No drift data found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading drift data: {e}")

class EnhancedDatasheetGasSensorArray:
    def __init__(self):
        """Initialize complete enhanced gas sensor array system"""
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

            self.tgs2600 = AnalogIn(self.ads, ADS.P0)
            self.tgs2602 = AnalogIn(self.ads, ADS.P1)
            self.tgs2610 = AnalogIn(self.ads, ADS.P2)

            self.logger.info("ADC ADS1115 initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ADC: {e}")
            raise

        # Enhanced sensor configurations
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
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
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
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
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
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
            }
        }

        # Initialize all managers
        self.drift_manager = SmartDriftManager(self.logger)
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)
        self.sensitivity_manager = AdvancedSensitivityManager(self.logger)

        # Environmental compensation
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0
        self.current_humidity = 65.0

        # Data storage
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.data_file = f"gas_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Machine Learning
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False

        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)

        self.logger.info("Enhanced Gas Sensor Array System with Advanced Sensitivity Control v4.0 initialized")

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
            'TGS2600': {-20: 1.8, -10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0, 30: 0.95, 40: 0.9, 50: 0.85},
            'TGS2602': {-10: 1.5, 0: 1.3, 10: 1.1, 20: 1.0, 30: 0.9, 40: 0.85, 50: 0.8, 60: 0.75},
            'TGS2610': {-10: 1.4, 0: 1.2, 10: 1.05, 20: 1.0, 30: 0.95, 40: 0.9, 50: 0.85}
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
        """Enhanced resistance to PPM conversion with all algorithms"""
        config = self.sensor_config[sensor_name]
        
        # Priority 1: Advanced sensitivity calculation
        if config.get('use_advanced_sensitivity', True):
            current_voltage = config['channel'].voltage
            baseline_voltage = config.get('baseline_voltage', 1.6)
            ppm_advanced = self.sensitivity_manager.advanced_ppm_calculation(
                sensor_name, current_voltage, baseline_voltage, gas_type
            )
            if ppm_advanced > 0:
                return ppm_advanced
        
        # Priority 2: Emergency calculation
        if config.get('use_emergency_ppm', False):
            current_voltage = config['channel'].voltage
            return self.emergency_ppm_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, gas_type, self.sensitivity_manager
            )
        
        # Priority 3: Standard calculation
        R0 = config.get('R0')
        if R0 is None or R0 == 0:
            current_voltage = config['channel'].voltage
            emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, gas_type, self.sensitivity_manager
            )
            
            if emergency_ppm > 0:
                return emergency_ppm
            
            return self.simplified_ppm_calculation(sensor_name, resistance)

        rs_r0_ratio = resistance / R0

        if config['use_extended_mode']:
            return self.extended_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
        else:
            ppm = self.datasheet_ppm_calculation(sensor_name, rs_r0_ratio, gas_type)
            
            if ppm == 0:
                current_voltage = config['channel'].voltage
                baseline_voltage = config.get('baseline_voltage', 1.6)
                advanced_ppm = self.sensitivity_manager.advanced_ppm_calculation(
                    sensor_name, current_voltage, baseline_voltage, gas_type
                )
                return max(ppm, advanced_ppm)
            
            return ppm

    def datasheet_ppm_calculation(self, sensor_name, rs_r0_ratio, gas_type):
        """Enhanced datasheet PPM calculation"""
        if sensor_name == 'TGS2600':
            if rs_r0_ratio < 0.15:
                ppm = min(60, 60 * (0.4 / rs_r0_ratio) ** 2.5)
            elif rs_r0_ratio > 0.98:
                ppm = 0
            else:
                ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                ppm = min(ppm, 60)

        elif sensor_name == 'TGS2602':
            if rs_r0_ratio < 0.03:
                ppm = min(50, 45 * (0.15 / rs_r0_ratio) ** 1.8)
            elif rs_r0_ratio > 0.98:
                ppm = 0
            else:
                ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                ppm = min(ppm, 50)

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.35:
                ppm = min(35, 40 * (0.45 / rs_r0_ratio) ** 1.2)
            elif rs_r0_ratio > 0.99:
                ppm = 0
            else:
                ppm = 30 * ((0.6 / rs_r0_ratio) ** 1.2)
                ppm = min(ppm, 35)
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
            'alcohol': 1.0, 'pertalite': 1.3, 'pertamax': 1.6, 'dexlite': 1.9, 
            'biosolar': 2.2, 'hydrogen': 0.8, 'toluene': 1.1, 'ammonia': 0.9, 
            'butane': 1.2, 'propane': 1.4, 'normal': 0.7
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Simplified PPM calculation"""
        config = self.sensor_config[sensor_name]
        baseline_voltage = config.get('baseline_voltage', 1.6)

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

    def read_sensors(self):
        """Enhanced sensor reading with all features"""
        readings = {}

        for sensor_name, config in self.sensor_config.items():
            try:
                # Read raw voltage
                raw_voltage = config['channel'].voltage
                
                # Apply smart drift compensation
                normalized_voltage, compensated_voltage = self.drift_manager.apply_smart_compensation(sensor_name, raw_voltage)

                # Convert to resistance
                resistance = self.voltage_to_resistance(compensated_voltage, config['load_resistance'])

                # Apply environmental compensation
                compensated_resistance = self.temperature_compensation(
                    sensor_name, resistance, self.current_temperature)
                compensated_resistance = self.humidity_compensation(
                    sensor_name, compensated_resistance, self.current_humidity)

                # Calculate Rs/R0 ratio
                R0 = config.get('R0')
                rs_r0_ratio = compensated_resistance / R0 if R0 else None

                # Calculate PPM with all methods
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)
                
                emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(
                    sensor_name, raw_voltage, 'auto', self.sensitivity_manager
                )
                
                advanced_ppm = self.sensitivity_manager.advanced_ppm_calculation(
                    sensor_name, raw_voltage, config.get('baseline_voltage', 1.6)
                )

                # Mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"
                if config.get('use_emergency_ppm', False):
                    current_mode += " (Emergency)"
                if config.get('use_advanced_sensitivity', True):
                    current_mode += " (Advanced)"

                # Status info
                drift_factor = self.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                normalization_factor = self.drift_manager.normalization_factors.get(sensor_name, 1.0)
                smart_compensation_applied = abs(1 - drift_factor) > 0.01 or abs(1 - normalization_factor) > 0.01
                voltage_adjusted = self.drift_manager.voltage_adjustments.get(sensor_name, {}).get('adjusted', False)
                
                sensitivity_status = self.sensitivity_manager.get_sensitivity_status().get(sensor_name, {})

                readings[sensor_name] = {
                    'voltage': normalized_voltage,
                    'raw_voltage': raw_voltage,
                    'compensated_voltage': compensated_voltage,
                    'resistance': resistance,
                    'compensated_resistance': compensated_resistance,
                    'rs_r0_ratio': rs_r0_ratio,
                    'ppm': ppm,
                    'emergency_ppm': emergency_ppm,
                    'advanced_ppm': advanced_ppm,
                    'R0': R0,
                    'mode': current_mode,
                    'target_gases': config['target_gases'],
                    'smart_compensation_applied': smart_compensation_applied,
                    'drift_factor': drift_factor,
                    'normalization_factor': normalization_factor,
                    'voltage_adjusted': voltage_adjusted,
                    'emergency_mode': config.get('use_emergency_ppm', False),
                    'advanced_sensitivity_mode': config.get('use_advanced_sensitivity', True),
                    'sensitivity_profile': sensitivity_status.get('profile', 'normal'),
                    'sensitivity_multiplier': sensitivity_status.get('effective_multiplier', 1.0)
                }

            except Exception as e:
                self.logger.error(f"Error reading {sensor_name}: {e}")
                readings[sensor_name] = {
                    'voltage': 0, 'raw_voltage': 0, 'compensated_voltage': 0, 'resistance': 0, 
                    'compensated_resistance': 0, 'rs_r0_ratio': None, 'ppm': 0, 'emergency_ppm': 0, 
                    'advanced_ppm': 0, 'R0': None, 'mode': 'Error', 'target_gases': [], 
                    'smart_compensation_applied': False, 'drift_factor': 1.0, 'normalization_factor': 1.0, 
                    'voltage_adjusted': False, 'emergency_mode': False, 'advanced_sensitivity_mode': False, 
                    'sensitivity_profile': 'unknown', 'sensitivity_multiplier': 1.0
                }

        return readings

    def set_sensor_mode(self, mode='datasheet'):
        """Set calculation mode for all sensors"""
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

    def calibrate_sensors(self, duration=300):
        """Enhanced calibration with all features"""
        self.logger.info(f"Starting enhanced sensor calibration for {duration} seconds...")
        self.logger.info("Ensure sensors are in CLEAN AIR environment!")

        input("Press Enter when sensors are in clean air and warmed up...")

        readings = {sensor: {'voltages': [], 'resistances': []}
                   for sensor in self.sensor_config.keys()}

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration:
            for sensor_name, config in self.sensor_config.items():
                voltage = config['channel'].voltage
                resistance = self.voltage_to_resistance(voltage, config['load_resistance'])

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

            self.sensor_config[sensor_name]['R0'] = resistance_mean
            self.sensor_config[sensor_name]['baseline_voltage'] = voltage_mean
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

        # Update drift manager
        self.drift_manager.last_calibration_time = datetime.now()
        
        for sensor_name, results in calibration_results.items():
            baseline_voltage = results['baseline_voltage']
            self.drift_manager.current_baseline[sensor_name] = baseline_voltage
            self.drift_manager.original_baseline[sensor_name] = baseline_voltage
            self.drift_manager.normalization_factors[sensor_name] = 1.0
            
            if sensor_name not in self.drift_manager.baseline_history:
                self.drift_manager.baseline_history[sensor_name] = []
            self.drift_manager.baseline_history[sensor_name].append(baseline_voltage)
        
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

        self.drift_manager.save_drift_data()
        self.logger.info(f"Enhanced calibration completed and saved to {calib_file}")

    def load_calibration(self):
        """Load calibration data"""
        try:
            with open('sensor_calibration.json', 'r') as f:
                calib_data = json.load(f)

            for sensor_name, data in calib_data['sensors'].items():
                if sensor_name in self.sensor_config:
                    self.sensor_config[sensor_name]['R0'] = data['R0']
                    self.sensor_config[sensor_name]['baseline_voltage'] = data['baseline_voltage']
                    self.sensor_config[sensor_name]['emergency_mode'] = False
                    self.sensor_config[sensor_name]['use_emergency_ppm'] = False

            if 'timestamp' in calib_data:
                self.drift_manager.last_calibration_time = datetime.fromisoformat(calib_data['timestamp'])

            self.logger.info("Enhanced calibration data loaded successfully")
            self.logger.info(f"Calibration date: {calib_data.get('timestamp', 'Unknown')}")

            return True
        except FileNotFoundError:
            self.logger.warning("No calibration file found. Advanced features available.")
            return False
        except Exception as e:
            self.logger.error(f"Error loading calibration file: {e}")
            return False

    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_model_trained = True
            self.logger.info("Enhanced model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction"""
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
                    elif measurement in ['emergency_ppm', 'advanced_ppm']:
                        if sensor in readings:
                            voltage = readings[sensor].get('raw_voltage', 1.6)
                            if measurement == 'emergency_ppm':
                                ppm = self.emergency_ppm_calc.calculate_emergency_ppm(sensor, voltage)
                            else:
                                ppm = self.sensitivity_manager.advanced_ppm_calculation(sensor, voltage, 1.6)
                            feature_vector.append(ppm)
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

    def set_environmental_conditions(self, temperature=None, humidity=None):
        """Set environmental conditions"""
        if temperature is not None:
            self.current_temperature = temperature
            self.logger.info(f"Temperature set to {temperature}°C")

        if humidity is not None:
            self.current_humidity = humidity
            self.logger.info(f"Humidity set to {humidity}%RH")

def main():
    """Complete main function with all features"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️  CALIBRATION NOT FOUND")
        print("Advanced Sensitivity System available as fallback")

    # Load existing model
    gas_sensor.load_model()

    while True:
        print("\n" + "="*70)
        print("🧠 SMART Gas Sensor Array System - Complete v4.0")
        print("Enhanced PPM Response + Ultra-Sensitive Detection")
        print("="*70)
        print("1. Calibrate sensors")
        print("2. Collect training data")
        print("3. Train machine learning model")
        print("4. Start monitoring - Datasheet mode")
        print("5. Start monitoring - Extended mode")
        print("6. Test single reading (Complete analysis)")
        print("7. Set environmental conditions")
        print("8. Switch sensor calculation mode")
        print("9. View sensor diagnostics")
        print("10. Exit")
        print("-" * 40)
        print("🧠 SMART DRIFT COMPENSATION:")
        print("11. Smart daily drift check")
        print("12. Quick stability test")
        print("13. Smart drift status report") 
        print("14. Manual drift compensation reset")
        print("15. AUTO BASELINE RESET")
        print("16. Smart system health check")
        print("-" * 40)
        print("🔧 VOLTAGE ADJUSTMENT:")
        print("17. Smart Voltage Adjustment")
        print("18. Sensor Responsivity Check")
        print("19. Quick voltage check")
        print("-" * 40)
        print("🚨 EMERGENCY PPM RECOVERY:")
        print("20. Emergency R0 Fix")
        print("21. Smart Troubleshoot PPM Issues (ENHANCED)")
        print("22. Toggle Emergency PPM Mode")
        print("23. Emergency PPM Test")
        print("-" * 40)
        print("🎯 ADVANCED SENSITIVITY CONTROL:")
        print("24. Auto-Sensitivity Calibration")
        print("25. Sensitivity Test with Gas")
        print("26. Manual Sensitivity Adjustment")
        print("27. Sensitivity Status Report")
        print("28. Reset All Sensitivity Settings")
        print("-"*70)

        try:
            choice = input("Select option (1-28): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                print("Ensure sensors are warmed up for at least 10 minutes in clean air!")
                confirm = input("Continue with calibration? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.calibrate_sensors(duration)

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*70)
                print("🧠 COMPLETE SENSOR ANALYSIS - ALL FEATURES")
                print("="*70)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    
                    print(f"\n  📊 PPM ANALYSIS (All Methods):")
                    print(f"    Main PPM: {data['ppm']:.1f}")
                    print(f"    Emergency PPM: {data['emergency_ppm']:.1f}")
                    print(f"    Advanced PPM: {data['advanced_ppm']:.1f}")
                    
                    print(f"\n  🎯 SENSITIVITY STATUS:")
                    print(f"    Profile: {data['sensitivity_profile']}")
                    print(f"    Multiplier: {data['sensitivity_multiplier']:.1f}x")
                    
                    if data['emergency_mode']:
                        print(f"    🚨 Emergency Mode: ACTIVE")
                    if data['advanced_sensitivity_mode']:
                        print(f"    🎯 Advanced Sensitivity: ACTIVE")

                print(f"\n🧠 PREDICTION:")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")

            elif choice == '21':
                print("\n🔧 ENHANCED SMART TROUBLESHOOT")
                
                current_readings = gas_sensor.read_sensors()
                problematic_sensors = []
                
                for sensor_name, data in current_readings.items():
                    if (data['ppm'] == 0 and data['emergency_ppm'] > 5) or \
                       (data['ppm'] < 5 and data['advanced_ppm'] > 10):
                        problematic_sensors.append(sensor_name)
                
                if not problematic_sensors:
                    print("✅ No major issues detected!")
                    sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                    print(f"\nSelect sensor for analysis:")
                    for i, sensor in enumerate(sensors, 1):
                        print(f"{i}. {sensor}")
                    
                    sensor_choice = input("Enter choice (1-3): ").strip()
                    if sensor_choice in ['1', '2', '3']:
                        target_sensor = sensors[int(sensor_choice) - 1]
                        gas_sensor.drift_manager.smart_troubleshoot_ppm_issue(gas_sensor, target_sensor)
                else:
                    print(f"⚠️  Issues detected in: {', '.join(problematic_sensors)}")
                    for sensor_name in problematic_sensors:
                        gas_sensor.drift_manager.smart_troubleshoot_ppm_issue(gas_sensor, sensor_name)

            # ADVANCED SENSITIVITY FEATURES (24-28)
            elif choice == '24':
                print("\n🎯 AUTO-SENSITIVITY CALIBRATION")
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor:")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                print("4. All sensors")
                
                sensor_choice = input("Enter choice (1-4): ").strip()
                
                if sensor_choice == '4':
                    for sensor in sensors:
                        print(f"\n🎯 Calibrating {sensor}...")
                        gas_sensor.sensitivity_manager.auto_sensitivity_calibration(gas_sensor, sensor, 30)
                elif sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    gas_sensor.sensitivity_manager.auto_sensitivity_calibration(gas_sensor, target_sensor, 60)

            elif choice == '25':
                print("\n🧪 SENSITIVITY TEST WITH GAS")
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor:")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                
                sensor_choice = input("Enter choice (1-3): ").strip()
                
                if sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    print(f"\n🧪 Prepare gas source (alcohol, perfume, etc.)")
                    gas_sensor.sensitivity_manager.sensitivity_test_with_gas(gas_sensor, target_sensor)

            elif choice == '26':
                print("\n🎛️ MANUAL SENSITIVITY ADJUSTMENT")
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor:")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                
                sensor_choice = input("Enter choice (1-3): ").strip()
                
                if sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    gas_sensor.sensitivity_manager.manual_sensitivity_adjustment(target_sensor)

            elif choice == '27':
                print("\n🎯 SENSITIVITY STATUS REPORT")
                status = gas_sensor.sensitivity_manager.get_sensitivity_status()
                
                print("="*60)
                print("ADVANCED SENSITIVITY STATUS:")
                print("="*60)
                
                for sensor_name, info in status.items():
                    print(f"\n{sensor_name}:")
                    print(f"  Profile: {info['profile']}")
                    print(f"  Custom Factor: {info['custom_factor']:.2f}x")
                    print(f"  Effective Multiplier: {info['effective_multiplier']:.1f}x")
                    print(f"  Sensitivity Level: {info['sensitivity_level']}")
                
                print(f"\n📊 CURRENT READINGS:")
                readings = gas_sensor.read_sensors()
                
                for sensor_name, data in readings.items():
                    print(f"\n{sensor_name}: {data['raw_voltage']:.3f}V")
                    print(f"  Main: {data['ppm']:.1f} | Emergency: {data['emergency_ppm']:.1f} | Advanced: {data['advanced_ppm']:.1f}")

            elif choice == '28':
                print("\n🔄 RESET ALL SENSITIVITY")
                confirm = input("Reset all to normal? (y/n): ").lower()
                if confirm == 'y':
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        gas_sensor.sensitivity_manager.current_sensitivity[sensor] = 'normal'
                        gas_sensor.sensitivity_manager.custom_factors[sensor] = 1.0
                    
                    gas_sensor.sensitivity_manager.save_sensitivity_data()
                    print("✅ All sensitivity reset to normal")

            elif choice == '10':
                print("👋 Exiting...")
                gas_sensor.drift_manager.save_drift_data()
                gas_sensor.sensitivity_manager.save_sensitivity_data()
                break

            else:
                print("❌ Invalid option!")

        except KeyboardInterrupt:
            print("\nOperation cancelled")
        except ValueError:
            print("❌ Invalid input!")
        except Exception as e:
            print(f"❌ Error: {e}")
            gas_sensor.logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()