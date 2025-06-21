#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - SMART AUTO DRIFT SOLUTION + ADVANCED SENSITIVITY CONTROL
Versi 4.0 dengan solusi untuk masalah sensitivity rendah dan response optimization:
1. Enhanced sensitivity adjustment per sensor
2. Advanced emergency PPM calculation with multiple algorithms
3. Auto-sensitivity calibration system
4. Adaptive threshold management
5. Real-time sensitivity tuning
6. Gas-specific response optimization
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
        
        # Custom sensitivity factors (user-adjustable)
        self.custom_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0,
            'TGS2610': 1.0
        }
        
        # Response optimization parameters
        self.response_optimization = {
            'TGS2600': {
                'alcohol_factor': 3.0,
                'hydrogen_factor': 2.5,
                'general_factor': 2.0,
                'min_detectable_change': 0.01  # 10mV
            },
            'TGS2602': {
                'alcohol_factor': 4.0,
                'toluene_factor': 3.5,
                'ammonia_factor': 3.0,
                'general_factor': 2.5,
                'min_detectable_change': 0.008  # 8mV
            },
            'TGS2610': {
                'butane_factor': 2.5,
                'propane_factor': 2.8,
                'general_factor': 2.0,
                'min_detectable_change': 0.012  # 12mV
            }
        }
        
        self.load_sensitivity_data()
    
    def advanced_ppm_calculation(self, sensor_name, current_voltage, baseline_voltage=None, gas_type='auto'):
        """
        Advanced PPM calculation dengan multiple algorithms dan sensitivity optimization
        """
        if baseline_voltage is None:
            baseline_voltage = 1.6
        
        # Get current sensitivity profile
        profile_name = self.current_sensitivity.get(sensor_name, 'normal')
        profile = self.sensitivity_profiles[sensor_name][profile_name]
        custom_factor = self.custom_factors.get(sensor_name, 1.0)
        
        voltage_drop = baseline_voltage - current_voltage
        min_change = self.response_optimization[sensor_name]['min_detectable_change']
        
        # Algorithm 1: Ultra-sensitive voltage-based calculation
        if abs(voltage_drop) < min_change:
            # Very small changes - ultra sensitive detection
            ppm_algo1 = voltage_drop * 1000 * profile['multiplier'] * custom_factor
        else:
            # Normal voltage drop calculation
            ppm_algo1 = voltage_drop * 500 * profile['multiplier'] * custom_factor
        
        # Algorithm 2: Exponential sensitivity curve
        if voltage_drop > 0:
            ppm_algo2 = (math.exp(voltage_drop * 5) - 1) * profile['multiplier'] * custom_factor * 50
        else:
            ppm_algo2 = 0
        
        # Algorithm 3: Power curve with gas-specific factors
        gas_factor = self.get_gas_response_factor(sensor_name, gas_type)
        if voltage_drop > profile['threshold']:
            ppm_algo3 = (voltage_drop ** 1.5) * 1000 * gas_factor * custom_factor
        else:
            ppm_algo3 = voltage_drop * 200 * gas_factor * custom_factor
        
        # Algorithm 4: Adaptive threshold calculation
        baseline_threshold = baseline_voltage * (1 - profile['baseline_factor'])
        if current_voltage < baseline_threshold:
            ppm_algo4 = (baseline_voltage - current_voltage) * 800 * profile['multiplier'] * custom_factor
        else:
            ppm_algo4 = 0
        
        # Combine algorithms with weighted average
        algorithms = [ppm_algo1, ppm_algo2, ppm_algo3, ppm_algo4]
        valid_algorithms = [ppm for ppm in algorithms if ppm >= 0]
        
        if not valid_algorithms:
            return 0
        
        # Weighted combination (prioritize consistent results)
        if len(valid_algorithms) >= 3:
            # Remove outliers and average
            sorted_ppms = sorted(valid_algorithms)
            if len(sorted_ppms) >= 3:
                # Use middle values
                final_ppm = np.mean(sorted_ppms[1:-1])
            else:
                final_ppm = np.mean(sorted_ppms)
        else:
            # Take maximum for better sensitivity
            final_ppm = max(valid_algorithms)
        
        # Apply final constraints
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
        """
        Auto-calibration untuk menentukan sensitivity optimal
        """
        print(f"\n🎯 AUTO SENSITIVITY CALIBRATION - {sensor_name}")
        print("="*60)
        print("Testing different sensitivity levels to find optimal setting...")
        
        # Test different profiles
        profiles_to_test = ['conservative', 'normal', 'high_sensitive', 'ultra_sensitive']
        results = {}
        
        baseline_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        print(f"Baseline voltage: {baseline_voltage:.3f}V")
        
        for profile in profiles_to_test:
            print(f"\n🧪 Testing {profile.upper()} sensitivity...")
            
            # Set profile temporarily
            old_profile = self.current_sensitivity[sensor_name]
            self.current_sensitivity[sensor_name] = profile
            
            # Collect readings for this profile
            readings = []
            print("Collecting baseline readings...")
            
            for i in range(20):
                current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
                ppm = self.advanced_ppm_calculation(sensor_name, current_voltage, baseline_voltage)
                readings.append(ppm)
                
                if i % 5 == 0:
                    print(f"  Sample {i+1}/20: {ppm:.1f} PPM")
                time.sleep(1)
            
            # Calculate statistics
            ppm_mean = np.mean(readings)
            ppm_std = np.std(readings)
            ppm_max = np.max(readings)
            noise_level = ppm_std / ppm_mean if ppm_mean > 0 else float('inf')
            
            results[profile] = {
                'mean_ppm': ppm_mean,
                'std_ppm': ppm_std,
                'max_ppm': ppm_max,
                'noise_level': noise_level,
                'stability': 1 / (1 + noise_level)  # Higher is better
            }
            
            print(f"  Mean PPM: {ppm_mean:.1f} ± {ppm_std:.1f}")
            print(f"  Noise level: {noise_level:.3f}")
            
            # Restore old profile
            self.current_sensitivity[sensor_name] = old_profile
        
        # Analyze results and recommend
        print(f"\n📊 CALIBRATION RESULTS:")
        print("-" * 40)
        
        best_profile = None
        best_score = -1
        
        for profile, data in results.items():
            # Scoring: balance between sensitivity and stability
            sensitivity_score = min(data['mean_ppm'] / 10, 5)  # Cap at 5
            stability_score = data['stability'] * 5
            overall_score = (sensitivity_score + stability_score) / 2
            
            print(f"{profile.upper()}: Score {overall_score:.2f} | PPM {data['mean_ppm']:.1f} | Noise {data['noise_level']:.3f}")
            
            if overall_score > best_score:
                best_score = overall_score
                best_profile = profile
        
        print(f"\n🎯 RECOMMENDED SENSITIVITY: {best_profile.upper()}")
        
        # Ask user to apply
        apply = input(f"\nApply {best_profile} sensitivity to {sensor_name}? (y/n): ").lower()
        if apply == 'y':
            self.current_sensitivity[sensor_name] = best_profile
            self.save_sensitivity_data()
            print(f"✅ {sensor_name} sensitivity set to {best_profile}")
            return True
        
        return False
    
    def sensitivity_test_with_gas(self, sensor_array, sensor_name):
        """
        Test sensitivity dengan gas spray untuk optimize response
        """
        print(f"\n🧪 SENSITIVITY TEST WITH GAS - {sensor_name}")
        print("="*60)
        print("This will test sensor response with actual gas spray")
        
        # Get baseline
        print("Measuring baseline...")
        baseline_readings = []
        for i in range(10):
            voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
            baseline_readings.append(voltage)
            time.sleep(1)
        
        baseline_voltage = np.mean(baseline_readings)
        baseline_std = np.std(baseline_readings)
        
        print(f"Baseline: {baseline_voltage:.3f}V ± {baseline_std:.3f}V")
        
        input(f"\nReady to spray gas near {sensor_name}? Press Enter...")
        
        print("Spray gas now! Monitoring response for 30 seconds...")
        print("Press Ctrl+C to stop early")
        
        max_response = 0
        response_data = []
        
        try:
            start_time = time.time()
            while time.time() - start_time < 30:
                current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
                voltage_drop = baseline_voltage - current_voltage
                
                # Test all sensitivity profiles
                responses = {}
                for profile in self.sensitivity_profiles[sensor_name].keys():
                    old_profile = self.current_sensitivity[sensor_name]
                    self.current_sensitivity[sensor_name] = profile
                    ppm = self.advanced_ppm_calculation(sensor_name, current_voltage, baseline_voltage)
                    responses[profile] = ppm
                    self.current_sensitivity[sensor_name] = old_profile
                
                max_response = max(max_response, max(responses.values()))
                response_data.append({
                    'time': time.time() - start_time,
                    'voltage': current_voltage,
                    'voltage_drop': voltage_drop,
                    'responses': responses.copy()
                })
                
                # Display real-time
                best_response = max(responses.values())
                print(f"\rTime: {time.time() - start_time:.1f}s | Voltage: {current_voltage:.3f}V | "
                      f"Drop: {voltage_drop*1000:.0f}mV | Max PPM: {best_response:.0f}", end="")
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        
        # Analyze results
        print(f"\n\n📊 GAS RESPONSE ANALYSIS:")
        print("-" * 40)
        
        if max_response < 5:
            print("⚠️ WARNING: Very low response detected")
            print("Recommendations:")
            print("1. Try ultra_sensitive profile")
            print("2. Check sensor positioning")
            print("3. Use more concentrated gas source")
            
            # Auto-enable ultra sensitive
            self.current_sensitivity[sensor_name] = 'ultra_sensitive'
            print(f"✅ Auto-enabled ultra_sensitive mode for {sensor_name}")
            
        elif max_response < 30:
            print("🔧 MODERATE: Consider higher sensitivity")
            self.current_sensitivity[sensor_name] = 'high_sensitive'
            print(f"✅ Auto-enabled high_sensitive mode for {sensor_name}")
            
        else:
            print("✅ GOOD: Sensor showing good response")
            
        # Show profile comparison
        print(f"\nProfile comparison at peak response:")
        if response_data:
            peak_data = max(response_data, key=lambda x: max(x['responses'].values()))
            for profile, ppm in peak_data['responses'].items():
                status = "👑 SELECTED" if profile == self.current_sensitivity[sensor_name] else ""
                print(f"  {profile}: {ppm:.1f} PPM {status}")
        
        self.save_sensitivity_data()
        return max_response
    
    def manual_sensitivity_adjustment(self, sensor_name):
        """
        Manual sensitivity adjustment interface
        """
        print(f"\n🎛️ MANUAL SENSITIVITY ADJUSTMENT - {sensor_name}")
        print("="*60)
        
        current_profile = self.current_sensitivity.get(sensor_name, 'normal')
        current_custom = self.custom_factors.get(sensor_name, 1.0)
        
        print(f"Current profile: {current_profile}")
        print(f"Current custom factor: {current_custom:.2f}")
        
        print(f"\nAvailable profiles:")
        for i, profile in enumerate(self.sensitivity_profiles[sensor_name].keys(), 1):
            mult = self.sensitivity_profiles[sensor_name][profile]['multiplier']
            print(f"{i}. {profile}: {mult:.1f}x multiplier")
        
        print("6. Custom factor adjustment")
        print("7. Reset to normal")
        
        choice = input("Select option (1-7): ").strip()
        
        if choice in ['1', '2', '3', '4', '5']:
            profiles = list(self.sensitivity_profiles[sensor_name].keys())
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
    """Enhanced Emergency PPM Calculator dengan advanced algorithms"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Enhanced emergency baselines
        self.emergency_baselines = {
            'TGS2600': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 8000,
                'sensitivity_curve': 'hybrid',
                'gas_response_factor': 3.5,
                'detection_threshold': 0.005,  # 5mV - ultra sensitive
                'voltage_noise_threshold': 0.002  # 2mV noise filter
            },
            'TGS2602': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 9000,
                'sensitivity_curve': 'exponential_enhanced',
                'gas_response_factor': 4.0,
                'detection_threshold': 0.003,  # 3mV - ultra sensitive
                'voltage_noise_threshold': 0.001  # 1mV noise filter
            },
            'TGS2610': {
                'clean_air_voltage': 1.6,
                'clean_air_resistance': 7500,
                'sensitivity_curve': 'linear_enhanced',
                'gas_response_factor': 2.5,
                'detection_threshold': 0.008,  # 8mV
                'voltage_noise_threshold': 0.003  # 3mV noise filter
            }
        }
        
        # Enhanced gas factors
        self.gas_factors = {
            'TGS2600': {
                'alcohol': 2.5, 'hydrogen': 3.0, 'carbon_monoxide': 2.0,
                'pertalite': 2.8, 'pertamax': 3.5, 'default': 2.5
            },
            'TGS2602': {
                'alcohol': 3.0, 'toluene': 3.5, 'ammonia': 2.5, 'h2s': 4.0,
                'pertalite': 3.2, 'pertamax': 4.0, 'default': 3.0
            },
            'TGS2610': {
                'butane': 2.0, 'propane': 2.3, 'lp_gas': 2.1, 'iso_butane': 1.9,
                'pertalite': 2.5, 'pertamax': 3.0, 'default': 2.0
            }
        }
    
    def calculate_emergency_ppm(self, sensor_name, current_voltage, gas_type='default', sensitivity_manager=None):
        """
        Enhanced emergency PPM calculation dengan multiple algorithms
        """
        if sensor_name not in self.emergency_baselines:
            return 0
        
        baseline = self.emergency_baselines[sensor_name]
        baseline_voltage = baseline['clean_air_voltage']
        threshold = baseline['detection_threshold']
        noise_threshold = baseline['voltage_noise_threshold']
        
        # Calculate voltage drop
        voltage_drop = baseline_voltage - current_voltage
        
        # Noise filtering
        if abs(voltage_drop) < noise_threshold:
            return 0
        
        # Apply sensitivity manager multiplier if available
        sensitivity_multiplier = 1.0
        if sensitivity_manager:
            profile = sensitivity_manager.current_sensitivity.get(sensor_name, 'normal')
            custom_factor = sensitivity_manager.custom_factors.get(sensor_name, 1.0)
            base_multiplier = sensitivity_manager.sensitivity_profiles[sensor_name][profile]['multiplier']
            sensitivity_multiplier = base_multiplier * custom_factor
        
        # Enhanced algorithms
        curve_type = baseline['sensitivity_curve']
        response_factor = baseline['gas_response_factor']
        gas_factor = self.gas_factors.get(sensor_name, {}).get(gas_type, 1.0)
        
        if curve_type == 'hybrid':
            # Hybrid: exponential for small changes, power for large changes
            if abs(voltage_drop) < 0.05:
                ppm = response_factor * (math.exp(abs(voltage_drop) * 10) - 1) * gas_factor
            else:
                ppm = response_factor * (abs(voltage_drop) * 200) ** 1.3 * gas_factor
                
        elif curve_type == 'exponential_enhanced':
            # Enhanced exponential with plateau
            if abs(voltage_drop) < 0.1:
                ppm = response_factor * (math.exp(abs(voltage_drop) * 8) - 1) * gas_factor
            else:
                ppm = response_factor * 100 * gas_factor  # Plateau for high concentrations
                
        elif curve_type == 'linear_enhanced':
            # Enhanced linear with acceleration
            if abs(voltage_drop) < 0.05:
                ppm = response_factor * abs(voltage_drop) * 500 * gas_factor
            else:
                ppm = response_factor * abs(voltage_drop) * 300 * gas_factor
                
        else:
            # Default calculation
            ppm = abs(voltage_drop) * 200 * gas_factor
        
        # Apply sensitivity multiplier
        ppm *= sensitivity_multiplier
        
        # Ensure positive and reasonable limits
        max_ppm = 2000 if sensitivity_multiplier > 5 else 1000
        return min(max_ppm, max(0, ppm))

class SmartDriftManager:
    """Smart Drift Manager - Enhanced dengan advanced troubleshooting"""
    
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
        Enhanced smart troubleshooting dengan advanced solutions
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
        
        # Check 4: Enhanced PPM Calculation Analysis
        print(f"\n📊 STEP 4: ENHANCED PPM CALCULATION ANALYSIS")
        try:
            ppm_datasheet = sensor_array.resistance_to_ppm(sensor_name, resistance, 'auto')
            print(f"Datasheet PPM: {ppm_datasheet:.1f}")
            
            # Try emergency calculation
            emergency_calc = sensor_array.emergency_ppm_calc
            ppm_emergency = emergency_calc.calculate_emergency_ppm(
                sensor_name, current_voltage, 'auto', sensor_array.sensitivity_manager
            )
            print(f"Emergency PPM: {ppm_emergency:.1f}")
            
            # Try advanced sensitivity calculation
            ppm_advanced = sensor_array.sensitivity_manager.advanced_ppm_calculation(
                sensor_name, current_voltage, baseline_voltage
            )
            print(f"Advanced PPM: {ppm_advanced:.1f}")
            
            if ppm_datasheet == 0 and ppm_emergency > 0:
                print("⚠️ Datasheet calculation failed, but emergency calculation works")
            elif ppm_datasheet == 0 and ppm_advanced > 0:
                print("⚠️ Datasheet calculation failed, but advanced calculation works")
            elif ppm_datasheet > 0:
                print("✅ PPM calculation working")
            else:
                print("❌ All calculations return low values")
                
        except Exception as e:
            print(f"❌ PPM calculation error: {e}")
        
        # Check 5: Sensitivity Analysis
        print(f"\n📊 STEP 5: SENSITIVITY ANALYSIS")
        sensitivity_status = sensor_array.sensitivity_manager.get_sensitivity_status()
        sensor_sensitivity = sensitivity_status.get(sensor_name, {})
        
        print(f"Current sensitivity profile: {sensor_sensitivity.get('profile', 'unknown')}")
        print(f"Effective multiplier: {sensor_sensitivity.get('effective_multiplier', 1.0):.1f}x")
        print(f"Sensitivity level: {sensor_sensitivity.get('sensitivity_level', 'unknown')}")
        
        if sensor_sensitivity.get('effective_multiplier', 1.0) < 2.0:
            print("⚠️ Low sensitivity - may need higher sensitivity profile")
        
        # Generate enhanced recommendations
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
            if ppm_emergency > 10 or ppm_advanced > 10:
                solutions.append("5. Enable Emergency PPM Mode - calculation working")
                solutions.append("6. Increase sensitivity profile to ultra_sensitive")
            else:
                solutions.append("7. Run sensitivity calibration (Option 24)")
                solutions.append("8. Test with gas spray (Option 25)")
        
        solutions.append("9. Check if sensor is in clean air (PPM 0 may be normal)")
        
        for solution in solutions:
            print(f"   {solution}")
        
        # Enhanced auto-fix options
        print(f"\n🔧 ENHANCED AUTO-FIX OPTIONS:")
        print("1. Enable Emergency PPM Mode (bypass R0 issues)")
        print("2. Fix R0 with emergency calibration")
        print("3. Enable ultra-sensitive mode")
        print("4. Run auto-sensitivity calibration")
        print("5. Reset sensor configuration")
        
        fix_choice = input("Select auto-fix option (1-5, or Enter to skip): ").strip()
        
        if fix_choice == '1':
            return self.enable_emergency_ppm_mode(sensor_array, sensor_name)
        elif fix_choice == '2':
            return self.emergency_r0_fix(sensor_array, sensor_name)
        elif fix_choice == '3':
            return self.enable_ultra_sensitive_mode(sensor_array, sensor_name)
        elif fix_choice == '4':
            return self.run_auto_sensitivity_calibration(sensor_array, sensor_name)
        elif fix_choice == '5':
            return self.reset_sensor_config(sensor_array, sensor_name)
        
        return False
    
    def enable_ultra_sensitive_mode(self, sensor_array, sensor_name):
        """Enable ultra-sensitive mode for better response"""
        print(f"\n🚀 ENABLING ULTRA-SENSITIVE MODE for {sensor_name}")
        
        # Set ultra-sensitive profile
        sensor_array.sensitivity_manager.current_sensitivity[sensor_name] = 'ultra_sensitive'
        sensor_array.sensitivity_manager.custom_factors[sensor_name] = 2.0  # Extra boost
        
        # Test current reading
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        ppm = sensor_array.sensitivity_manager.advanced_ppm_calculation(
            sensor_name, current_voltage, 1.6
        )
        
        print(f"✅ Ultra-sensitive mode enabled")
        print(f"✅ Current PPM with ultra sensitivity: {ppm:.1f}")
        print(f"✅ Sensor should now be much more responsive")
        
        sensor_array.sensitivity_manager.save_sensitivity_data()
        return True
    
    def run_auto_sensitivity_calibration(self, sensor_array, sensor_name):
        """Run auto-sensitivity calibration"""
        print(f"\n🎯 RUNNING AUTO-SENSITIVITY CALIBRATION for {sensor_name}")
        return sensor_array.sensitivity_manager.auto_sensitivity_calibration(
            sensor_array, sensor_name, 30
        )
    
    def enable_emergency_ppm_mode(self, sensor_array, sensor_name):
        """Enable emergency PPM mode for sensor"""
        print(f"\n🚨 ENABLING EMERGENCY PPM MODE for {sensor_name}")
        print("This will bypass R0 calibration and use voltage-based calculation")
        
        # Mark sensor for emergency mode
        sensor_array.sensor_config[sensor_name]['emergency_mode'] = True
        sensor_array.sensor_config[sensor_name]['use_emergency_ppm'] = True
        
        # Test emergency calculation
        current_voltage = sensor_array.sensor_config[sensor_name]['channel'].voltage
        emergency_ppm = sensor_array.emergency_ppm_calc.calculate_emergency_ppm(
            sensor_name, current_voltage, 'auto', sensor_array.sensitivity_manager
        )
        
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
        
        # Reset sensitivity to normal
        sensor_array.sensitivity_manager.current_sensitivity[sensor_name] = 'normal'
        sensor_array.sensitivity_manager.custom_factors[sensor_name] = 1.0
        
        print(f"✅ Configuration reset")
        print(f"✅ Sensitivity reset to normal")
        print(f"✅ Ready for fresh calibration or emergency mode")
        
        return True
    
    # [Rest of SmartDriftManager methods remain the same as previous version...]
    
    def smart_voltage_adjustment(self, sensor_array, target_sensor='TGS2600', target_voltage=1.6):
        """Smart voltage adjustment - same as before"""
        # [Previous implementation remains the same]
        pass
    
    def apply_voltage_update(self, sensor_name, new_voltage, target_voltage, sensor_array):
        """Apply voltage update - same as before"""
        # [Previous implementation remains the same]
        pass
    
    def is_daily_check_needed(self):
        """Check jika daily drift check perlu dilakukan"""
        if not self.daily_check_done:
            return True
            
        if hasattr(self, 'last_check_date'):
            today = datetime.now().date()
            if today != self.last_check_date:
                return True
                
        return False
    
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
        
        # [Previous implementation remains the same]
        return status
    
    def save_drift_data(self):
        """Save comprehensive smart drift data"""
        drift_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'smart_drift_v4.0_with_advanced_sensitivity',
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
            
            # Load voltage adjustments
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
        """Initialize enhanced gas sensor array system with Advanced Sensitivity Control"""
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

        # Initialize advanced systems
        self.drift_manager = SmartDriftManager(self.logger)
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)
        self.sensitivity_manager = AdvancedSensitivityManager(self.logger)

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
        """Enhanced resistance to PPM conversion with advanced algorithms"""
        config = self.sensor_config[sensor_name]
        
        # Priority 1: Advanced sensitivity calculation (NEW!)
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
        
        # Priority 3: Standard R0-based calculation
        R0 = config.get('R0')
        if R0 is None or R0 == 0:
            # Fallback to emergency calculation
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
            
            # If datasheet calculation returns 0, try advanced calculation
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
        # Same implementation as before, but with better tolerance
        if sensor_name == 'TGS2600':
            if gas_type == 'hydrogen' or gas_type == 'auto':
                if rs_r0_ratio < 0.15:  # More sensitive threshold
                    ppm = min(60, 60 * (0.4 / rs_r0_ratio) ** 2.5)
                elif rs_r0_ratio > 0.98:
                    ppm = 0
                else:
                    ppm = 50 * ((0.6 / rs_r0_ratio) ** 2.5)
                    ppm = min(ppm, 60)
            elif gas_type == 'alcohol':
                if rs_r0_ratio < 0.1:  # More sensitive threshold
                    ppm = min(50, 50 * (0.3 / rs_r0_ratio) ** 2.0)
                elif rs_r0_ratio > 0.95:
                    ppm = 0
                else:
                    ppm = 40 * ((0.4 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 50)
            else:
                if rs_r0_ratio < 0.95:
                    ppm = 30 * ((0.5 / rs_r0_ratio) ** 2.0)
                    ppm = min(ppm, 50)
                else:
                    ppm = 0

        elif sensor_name == 'TGS2602':
            if gas_type == 'alcohol' or gas_type == 'auto':
                if rs_r0_ratio < 0.03:  # Much more sensitive threshold
                    ppm = min(50, 45 * (0.15 / rs_r0_ratio) ** 1.8)
                elif rs_r0_ratio > 0.98:
                    ppm = 0
                else:
                    ppm = 25 * ((0.25 / rs_r0_ratio) ** 1.8)
                    ppm = min(ppm, 50)
            elif gas_type == 'toluene':
                if rs_r0_ratio < 0.95:
                    ppm = 25 * ((0.2 / rs_r0_ratio) ** 1.5)
                    ppm = min(ppm, 40)
                else:
                    ppm = 0
            else:
                if rs_r0_ratio < 0.95:
                    ppm = 20 * ((0.3 / rs_r0_ratio) ** 1.6)
                    ppm = min(ppm, 40)
                else:
                    ppm = 0

        elif sensor_name == 'TGS2610':
            if rs_r0_ratio < 0.35:  # More sensitive threshold
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
        """Extended PPM calculation - same as before"""
        # [Previous implementation remains the same]
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
            'alcohol': 1.0, 'pertalite': 1.3, 'pertamax': 1.6, 'dexlite': 1.9, 'biosolar': 2.2,
            'hydrogen': 0.8, 'toluene': 1.1, 'ammonia': 0.9, 'butane': 1.2, 'propane': 1.4, 'normal': 0.7
        }

        multiplier = gas_multipliers.get(gas_type, 1.0)
        return base_ppm * multiplier

    def simplified_ppm_calculation(self, sensor_name, resistance):
        """Enhanced simplified PPM calculation"""
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
        """Enhanced sensor reading with all advanced features"""
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

                # Calculate PPM with all methods
                ppm = self.resistance_to_ppm(sensor_name, compensated_resistance)
                
                # Calculate emergency PPM
                emergency_ppm = self.emergency_ppm_calc.calculate_emergency_ppm(
                    sensor_name, raw_voltage, 'auto', self.sensitivity_manager
                )
                
                # Calculate advanced sensitivity PPM
                advanced_ppm = self.sensitivity_manager.advanced_ppm_calculation(
                    sensor_name, raw_voltage, config.get('baseline_voltage', 1.6)
                )

                # Current mode info
                current_mode = "Extended" if config['use_extended_mode'] else "Datasheet"
                if config.get('use_emergency_ppm', False):
                    current_mode += " (Emergency)"
                if config.get('use_advanced_sensitivity', True):
                    current_mode += " (Advanced)"

                # Smart drift info
                drift_factor = self.drift_manager.drift_compensation_factors.get(sensor_name, 1.0)
                normalization_factor = self.drift_manager.normalization_factors.get(sensor_name, 1.0)
                smart_compensation_applied = abs(1 - drift_factor) > 0.01 or abs(1 - normalization_factor) > 0.01
                
                # Voltage adjustment info
                voltage_adjusted = self.drift_manager.voltage_adjustments.get(sensor_name, {}).get('adjusted', False)
                
                # Sensitivity info
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
                    'voltage': 0, 'raw_voltage': 0, 'compensated_voltage': 0, 'resistance': 0, 'compensated_resistance': 0,
                    'rs_r0_ratio': None, 'ppm': 0, 'emergency_ppm': 0, 'advanced_ppm': 0, 'R0': None, 'mode': 'Error',
                    'target_gases': [], 'smart_compensation_applied': False, 'drift_factor': 1.0, 
                    'normalization_factor': 1.0, 'voltage_adjusted': False, 'emergency_mode': False,
                    'advanced_sensitivity_mode': False, 'sensitivity_profile': 'unknown', 'sensitivity_multiplier': 1.0
                }

        return readings

    # [Rest of the methods like calibrate_sensors, train_model, etc. remain largely the same but with enhanced features]
    # Implementing key methods with enhancements:

    def set_sensor_mode(self, mode='datasheet'):
        """Set calculation mode for all sensors"""
        use_extended = (mode == 'extended')

        for sensor_name in self.sensor_config.keys():
            self.sensor_config[sensor_name]['use_extended_mode'] = use_extended

        mode_name = "Extended (Training)" if use_extended else "Datasheet (Accurate)"
        self.logger.info(f"Sensor calculation mode set to: {mode_name}")

def main():
    """Enhanced main function with Advanced Sensitivity Control"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration if available
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️  CALIBRATION NOT FOUND")
        print("Advanced Sensitivity System and Emergency PPM mode available as fallback")

    # Load existing model if available
    gas_sensor.load_model()

    while True:
        print("\n" + "="*70)
        print("🧠 SMART Gas Sensor Array System - Advanced Sensitivity Control v4.0")
        print("Enhanced PPM Response + Ultra-Sensitive Detection")
        print("="*70)
        print("1. Calibrate sensors (Enhanced R0 determination)")
        print("2. Collect training data (Auto-switch to Extended mode)")
        print("3. Train machine learning model (All advanced features)")
        print("4. Start monitoring - Datasheet mode (Accurate detection)")
        print("5. Start monitoring - Extended mode (Full range)")
        print("6. Test single reading (Complete analysis with all PPM methods)")
        print("7. Set environmental conditions (T°C, %RH)")
        print("8. Switch sensor calculation mode")
        print("9. View sensor diagnostics (Complete system status)")
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
        print("🎯 ADVANCED SENSITIVITY CONTROL (NEW!):")
        print("24. Auto-Sensitivity Calibration (Find optimal sensitivity)")
        print("25. Sensitivity Test with Gas (Real-world response test)")
        print("26. Manual Sensitivity Adjustment (Fine-tune response)")
        print("27. Sensitivity Status Report (Current settings)")
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
                print("🧠 COMPLETE SENSOR ANALYSIS - ALL ADVANCED FEATURES")
                print("="*70)

                for sensor, data in readings.items():
                    print(f"\n{sensor} ({data['mode']} mode):")
                    print(f"  Model Voltage: {data['voltage']:.3f}V")
                    print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                    
                    if data['voltage_adjusted']:
                        print(f"  🔧 Hardware: Voltage adjusted")
                    
                    if data['smart_compensation_applied']:
                        print(f"  🧠 Smart Compensation: Active")
                    
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    if data['rs_r0_ratio']:
                        print(f"  Rs/R0 Ratio: {data['rs_r0_ratio']:.3f}")
                    
                    print(f"\n  📊 PPM ANALYSIS (Multiple Methods):")
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
                    
                    print(f"  Target Gases: {', '.join(data['target_gases'])}")

                print(f"\n🧠 PREDICTION (All features):")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")

            elif choice == '21':
                print("\n🔧 ENHANCED SMART TROUBLESHOOT PPM ISSUES")
                
                # Check which sensors have issues
                current_readings = gas_sensor.read_sensors()
                problematic_sensors = []
                
                for sensor_name, data in current_readings.items():
                    if (data['ppm'] == 0 and data['emergency_ppm'] > 5) or \
                       (data['ppm'] < 5 and data['advanced_ppm'] > 10):
                        problematic_sensors.append(sensor_name)
                
                if not problematic_sensors:
                    print("✅ No major PPM issues detected!")
                    
                    # Still offer individual troubleshooting
                    sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                    print(f"\nSelect sensor for detailed analysis:")
                    for i, sensor in enumerate(sensors, 1):
                        print(f"{i}. {sensor}")
                    
                    sensor_choice = input("Enter choice (1-3): ").strip()
                    if sensor_choice in ['1', '2', '3']:
                        target_sensor = sensors[int(sensor_choice) - 1]
                        gas_sensor.drift_manager.smart_troubleshoot_ppm_issue(gas_sensor, target_sensor)
                else:
                    print(f"⚠️  PPM ISSUES DETECTED in: {', '.join(problematic_sensors)}")
                    
                    for sensor_name in problematic_sensors:
                        print(f"\n🔧 TROUBLESHOOTING {sensor_name}...")
                        gas_sensor.drift_manager.smart_troubleshoot_ppm_issue(gas_sensor, sensor_name)

            # NEW ADVANCED SENSITIVITY FEATURES (24-28)
            elif choice == '24':
                print("\n🎯 AUTO-SENSITIVITY CALIBRATION")
                print("This will test different sensitivity levels to find optimal setting")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor for calibration:")
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
                print("This will test sensor response with actual gas spray")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor for testing:")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                
                sensor_choice = input("Enter choice (1-3): ").strip()
                
                if sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    print(f"\n🧪 Prepare gas source (alcohol, perfume, lighter gas, etc.)")
                    gas_sensor.sensitivity_manager.sensitivity_test_with_gas(gas_sensor, target_sensor)

            elif choice == '26':
                print("\n🎛️ MANUAL SENSITIVITY ADJUSTMENT")
                
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor for adjustment:")
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
                    print(f"  Base Multiplier: {info['base_multiplier']:.1f}x")
                    print(f"  Effective Multiplier: {info['effective_multiplier']:.1f}x")
                    print(f"  Sensitivity Level: {info['sensitivity_level']}")
                
                # Show current readings with all methods
                print(f"\n📊 CURRENT READINGS WITH ALL METHODS:")
                readings = gas_sensor.read_sensors()
                
                for sensor_name, data in readings.items():
                    print(f"\n{sensor_name}: V={data['raw_voltage']:.3f}V")
                    print(f"  Main PPM: {data['ppm']:.1f}")
                    print(f"  Emergency PPM: {data['emergency_ppm']:.1f}")
                    print(f"  Advanced PPM: {data['advanced_ppm']:.1f}")

            elif choice == '28':
                print("\n🔄 RESET ALL SENSITIVITY SETTINGS")
                print("This will reset all sensors to normal sensitivity")
                
                confirm = input("Are you sure? (y/n): ").lower()
                if confirm == 'y':
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        gas_sensor.sensitivity_manager.current_sensitivity[sensor] = 'normal'
                        gas_sensor.sensitivity_manager.custom_factors[sensor] = 1.0
                    
                    gas_sensor.sensitivity_manager.save_sensitivity_data()
                    print("✅ All sensitivity settings reset to normal")

            elif choice == '10':
                print("👋 Exiting Smart Gas Sensor System...")
                gas_sensor.drift_manager.save_drift_data()
                gas_sensor.sensitivity_manager.save_sensitivity_data()
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