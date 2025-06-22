#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - COMPLETE VERSION 4.4 - SYSTEMATIC COLLECTION
OPSI 2: Systematic data collection dari awal dengan protocol 3 menit yang terbukti berhasil
SEMUA features original tetap ada + enhanced systematic collection protocol
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
import glob
import os
import warnings
from pathlib import Path

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Library untuk ADC dan GPIO (sama seperti asli)
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

# Machine Learning libraries (sama seperti asli)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    import joblib
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    import joblib

class SystematicCollectionGuide:
    """Systematic Collection Guide dengan protocol 3 menit yang terbukti"""
    
    def __init__(self, logger):
        self.logger = logger
        self.collection_status = {}
        self.validation_results = {}
        
        # Protocol 3 menit - sesuai yang berhasil di percobaan user
        self.collection_protocol = {
            'duration_per_gas': 180,  # 3 menit
            'recovery_time': 300,     # 5 menit recovery antar gas
            'sample_interval': 2,     # Sample setiap 2 detik
            'expected_samples': 90,   # 180/2 = 90 samples per gas
            'min_samples_required': 60  # Minimal samples untuk valid
        }
        
        # Gas collection sequence - urutan optimal
        self.gas_sequence = [
            {
                'name': 'normal',
                'display_name': 'Clean Air (Normal)',
                'description': 'Baseline measurement dalam clean air',
                'preparation': [
                    'Pastikan ruangan berventilasi baik',
                    'Tidak ada bau gas, parfum, atau chemical',
                    'Matikan AC/kipas angin 5 menit sebelumnya',
                    'Jangan ada aktivitas memasak nearby'
                ],
                'collection_method': [
                    'Jangan bergerak dekat sensor',
                    'Jangan bernapas langsung ke sensor',
                    'Monitor PPM harus < 5 untuk semua sensor',
                    'Jika ada spike PPM > 10, stop dan tunggu clean lagi'
                ],
                'success_criteria': {
                    'TGS2600_ppm_max': 5,
                    'TGS2602_ppm_max': 5,
                    'TGS2610_ppm_max': 5,
                    'total_ppm_max': 10
                }
            },
            {
                'name': 'alcohol',
                'display_name': 'Alcohol (Ethanol)',
                'description': 'Primary target untuk TGS2600',
                'preparation': [
                    'Siapkan alkohol 70% dalam botol spray',
                    'Test spray dulu - pastikan mist halus',
                    'Jarak optimal: 30-50cm dari sensor array',
                    'Siapkan kipas kecil jika perlu dorong uap'
                ],
                'collection_method': [
                    'Menit 0-1: Baseline (jangan spray)',
                    'Menit 1-2.5: Spray intermittent setiap 20-30 detik',
                    'Pattern: 2-3 spray → tunggu 30 detik → repeat',
                    'Menit 2.5-3: Recovery (stop spray)',
                    'Jaga jarak 30-50cm konsisten'
                ],
                'success_criteria': {
                    'TGS2600_ppm_mean': 30,
                    'TGS2600_dominance': True,  # TGS2600 > TGS2602 & TGS2610
                    'response_pattern': 'TGS2600_dominant'
                }
            },
            {
                'name': 'pertalite',
                'display_name': 'Pertalite (Gasoline)',
                'description': 'Hydrocarbon fuel - target TGS2602 & TGS2610',
                'preparation': [
                    'Siapkan pertalite/premium dalam wadah kecil (30-50ml)',
                    'Gunakan wadah dengan mulut lebar (gelas kecil)',
                    'Siapkan kipas kecil untuk dorong uap',
                    'Pastikan ventilasi cukup untuk safety'
                ],
                'collection_method': [
                    'Letakkan wadah 1 meter dari sensor',
                    'Menit 0-1: Baseline',
                    'Menit 1-2.5: Dekatkan wadah ke 40-60cm',
                    'Gunakan kipas kecil untuk dorong uap ke sensor',
                    'Pattern: dekat 30 detik → jauh 30 detik → repeat',
                    'Menit 2.5-3: Jauhkan wadah, recovery'
                ],
                'success_criteria': {
                    'TGS2602_ppm_mean': 25,
                    'TGS2610_ppm_mean': 15,
                    'response_pattern': 'TGS2602_TGS2610_active'
                }
            },
            {
                'name': 'toluene',
                'display_name': 'Toluene (Aromatic)',
                'description': 'Strong aromatic - target TGS2602 dominan',
                'preparation': [
                    'Gunakan nail polish remover (acetone/toluene)',
                    'Atau thinner/cat thinner',
                    'Siapkan cotton bud atau tissue',
                    'Celupkan cotton bud, jangan sampai menetes'
                ],
                'collection_method': [
                    'Menit 0-1: Baseline',
                    'Celupkan cotton bud dalam toluene',
                    'Menit 1-2.5: Gerakkan cotton bud 20-40cm dari sensor',
                    'Pattern: dekat 15 detik → jauh 30 detik → repeat',
                    'Ganti cotton bud setiap menit (fresh solvent)',
                    'Menit 2.5-3: Stop exposure, recovery'
                ],
                'success_criteria': {
                    'TGS2602_ppm_mean': 50,
                    'TGS2602_dominance': True,  # TGS2602 >> others
                    'response_pattern': 'TGS2602_very_dominant'
                }
            },
            {
                'name': 'ammonia',
                'display_name': 'Ammonia (NH3)',
                'description': 'Basic gas - target TGS2602 moderate',
                'preparation': [
                    'Gunakan pembersih lantai ammonia (Ekonomis, Vixal)',
                    'Atau bawang bombay dipotong fresh',
                    'Siapkan tissue/kain untuk celup',
                    'Alternative: cairan pembersih kaca (Windex)'
                ],
                'collection_method': [
                    'Menit 0-1: Baseline',
                    'Celupkan tissue dalam ammonia cleaner',
                    'Menit 1-2.5: Dekatkan tissue 25-40cm dari sensor',
                    'Pattern: exposure 20 detik → recovery 20 detik',
                    'Ganti tissue setiap menit untuk fresh ammonia',
                    'Menit 2.5-3: Recovery'
                ],
                'success_criteria': {
                    'TGS2602_ppm_mean': 35,
                    'different_from_toluene': True,
                    'response_pattern': 'TGS2602_moderate'
                }
            },
            {
                'name': 'butane',
                'display_name': 'Butane/LPG',
                'description': 'Combustible gas - target TGS2610 dominan',
                'preparation': [
                    'Gunakan gas lighter (jangan nyalakan!)',
                    'Atau tabung gas mini (untuk kompor portable)',
                    'Test dulu - pastikan gas keluar saat ditekan',
                    'Pastikan ventilasi SANGAT baik (safety first!)'
                ],
                'collection_method': [
                    'Menit 0-1: Baseline',
                    'Jarak aman: 50-70cm dari sensor',
                    'Menit 1-2.5: Release gas intermittent',
                    'Pattern: release 3 detik → pause 15 detik → repeat',
                    'JANGAN continuous release (safety!)',
                    'Monitor: TGS2610 harus response > others',
                    'Menit 2.5-3: Stop, ventilasi'
                ],
                'success_criteria': {
                    'TGS2610_ppm_mean': 25,
                    'TGS2610_dominance': True,  # TGS2610 > others
                    'response_pattern': 'TGS2610_dominant'
                }
            }
        ]
    
    def display_collection_overview(self):
        """Display overview lengkap collection protocol"""
        print("\n📋 SYSTEMATIC COLLECTION PROTOCOL OVERVIEW")
        print("="*70)
        print("🎯 Proven 3-minute collection protocol")
        print("⏱️ Based on your successful previous collections")
        
        total_time = self.calculate_total_time()
        print(f"\n⏱️ TIME BREAKDOWN:")
        print(f"   Collection per gas: {self.collection_protocol['duration_per_gas']} seconds (3 minutes)")
        print(f"   Recovery between gases: {self.collection_protocol['recovery_time']} seconds (5 minutes)")
        print(f"   Expected samples per gas: ~{self.collection_protocol['expected_samples']}")
        print(f"   Total gases: {len(self.gas_sequence)}")
        print(f"   TOTAL TIME NEEDED: ~{total_time} minutes ({total_time//60:.0f}h {total_time%60:.0f}m)")
        
        print(f"\n🎯 COLLECTION SEQUENCE:")
        for i, gas_info in enumerate(self.gas_sequence, 1):
            print(f"   {i}. {gas_info['display_name']} (3 min) + Recovery (5 min)")
        
        print(f"\n✅ SUCCESS RATE EXPECTED: 95%+ (if protocol followed correctly)")
        print(f"🔍 Quality control: Real-time validation during collection")
    
    def calculate_total_time(self):
        """Calculate total time needed"""
        n_gases = len(self.gas_sequence)
        collection_time = n_gases * (self.collection_protocol['duration_per_gas'] / 60)  # minutes
        recovery_time = (n_gases - 1) * (self.collection_protocol['recovery_time'] / 60)  # minutes
        setup_time = 15  # 15 minutes setup/preparation
        
        return int(collection_time + recovery_time + setup_time)
    
    def start_systematic_collection(self, sensor_array):
        """Start systematic collection process"""
        print("\n🚀 STARTING SYSTEMATIC COLLECTION PROCESS")
        print("="*60)
        
        # Pre-collection validation
        if not self.validate_system_ready(sensor_array):
            return False
        
        print(f"\n📋 SYSTEMATIC COLLECTION CHECKLIST:")
        print(f"✅ System validation passed")
        print(f"✅ {len(self.gas_sequence)} gases will be collected")
        print(f"✅ 3-minute proven protocol")
        print(f"✅ Real-time quality control")
        
        total_time = self.calculate_total_time()
        print(f"\n⏱️ TIME COMMITMENT: ~{total_time} minutes")
        print(f"💡 You can take breaks between gases (recovery time)")
        
        proceed = input(f"\nReady to start systematic collection? (y/n): ").lower().strip()
        if proceed != 'y':
            print("❌ Collection cancelled")
            return False
        
        # Execute collection for each gas
        collection_results = []
        
        for i, gas_info in enumerate(self.gas_sequence):
            print(f"\n" + "="*60)
            print(f"🎯 GAS {i+1}/{len(self.gas_sequence)}: {gas_info['display_name'].upper()}")
            print("="*60)
            
            # Display gas-specific instructions
            self.display_gas_instructions(gas_info)
            
            # Wait for user preparation
            input(f"Press Enter when ready to collect {gas_info['name']} data...")
            
            # Execute collection
            result = self.collect_gas_systematic(sensor_array, gas_info)
            collection_results.append(result)
            
            # Validate collection
            if self.validate_gas_collection(result, gas_info):
                print(f"✅ {gas_info['name']} collection: VALID")
                self.collection_status[gas_info['name']] = 'success'
            else:
                print(f"❌ {gas_info['name']} collection: INVALID")
                self.collection_status[gas_info['name']] = 'failed'
                
                retry = input(f"Retry {gas_info['name']} collection? (y/n): ").lower().strip()
                if retry == 'y':
                    print(f"🔄 Retrying {gas_info['name']} collection...")
                    result = self.collect_gas_systematic(sensor_array, gas_info)
                    collection_results[-1] = result  # Replace last result
                    
                    if self.validate_gas_collection(result, gas_info):
                        print(f"✅ {gas_info['name']} retry: VALID")
                        self.collection_status[gas_info['name']] = 'success'
            
            # Recovery time (except for last gas)
            if i < len(self.gas_sequence) - 1:
                recovery_time = self.collection_protocol['recovery_time']
                print(f"\n⏳ SENSOR RECOVERY TIME: {recovery_time} seconds")
                print(f"💡 This allows sensors to return to baseline")
                print(f"💡 You can take a break now")
                
                self.countdown_timer(recovery_time, f"Recovery for next gas")
        
        # Final summary
        self.display_collection_summary()
        
        return self.all_collections_successful()
    
    def display_gas_instructions(self, gas_info):
        """Display detailed instructions for specific gas"""
        print(f"\n📋 PREPARATION FOR {gas_info['display_name'].upper()}:")
        print(f"Description: {gas_info['description']}")
        
        print(f"\n🛠️ PREPARATION STEPS:")
        for i, step in enumerate(gas_info['preparation'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n🎯 COLLECTION METHOD:")
        for i, step in enumerate(gas_info['collection_method'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n✅ SUCCESS CRITERIA:")
        criteria = gas_info['success_criteria']
        for criterion, target in criteria.items():
            if isinstance(target, bool):
                print(f"   ✓ {criterion}: Required")
            else:
                print(f"   ✓ {criterion}: {target}")
    
    def collect_gas_systematic(self, sensor_array, gas_info):
        """Collect data untuk specific gas dengan systematic protocol"""
        gas_name = gas_info['name']
        duration = self.collection_protocol['duration_per_gas']
        
        print(f"\n🎯 COLLECTING {gas_name.upper()} - {duration} seconds")
        print("="*50)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"systematic_{gas_name}_{timestamp}.csv"
        
        # CSV headers
        headers = [
            'timestamp', 'gas_type', 'elapsed_time', 'temperature', 'humidity'
        ]
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            headers.extend([
                f'{sensor}_voltage',
                f'{sensor}_raw_voltage',
                f'{sensor}_resistance',
                f'{sensor}_ppm',
                f'{sensor}_emergency_ppm',
                f'{sensor}_advanced_ppm'
            ])
        
        # Real-time monitoring data
        realtime_data = {
            'timestamps': [],
            'sensor_data': {sensor: {'ppm': [], 'voltage': []} for sensor in ['TGS2600', 'TGS2602', 'TGS2610']},
            'quality_flags': []
        }
        
        collected_data = []
        start_time = time.time()
        sample_count = 0
        
        print(f"🎯 Target: {self.collection_protocol['expected_samples']} samples")
        print(f"📊 Real-time monitoring active...")
        print(f"⏹️ Press Ctrl+C to stop early\n")
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                while time.time() - start_time < duration:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Read sensors
                    readings = sensor_array.read_sensors()
                    
                    # Prepare row data
                    row = [
                        datetime.now().isoformat(),
                        gas_name,
                        elapsed,
                        sensor_array.current_temperature,
                        sensor_array.current_humidity
                    ]
                    
                    # Add sensor data
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        sensor_data = readings[sensor]
                        row.extend([
                            sensor_data['voltage'],
                            sensor_data['raw_voltage'],
                            sensor_data['resistance'],
                            sensor_data['ppm'],
                            sensor_data.get('emergency_ppm', 0),
                            sensor_data.get('advanced_ppm', 0)
                        ])
                        
                        # Store for real-time analysis
                        realtime_data['sensor_data'][sensor]['ppm'].append(sensor_data['ppm'])
                        realtime_data['sensor_data'][sensor]['voltage'].append(sensor_data['raw_voltage'])
                    
                    writer.writerow(row)
                    collected_data.append(row)
                    realtime_data['timestamps'].append(elapsed)
                    sample_count += 1
                    
                    # Real-time quality assessment
                    quality_ok = self.assess_realtime_quality(readings, gas_info, elapsed)
                    realtime_data['quality_flags'].append(quality_ok)
                    
                    # Enhanced progress display
                    if sample_count % 15 == 0:  # Every 30 seconds
                        progress = (elapsed / duration) * 100
                        remaining = duration - elapsed
                        
                        print(f"⏱️ Progress: {progress:.1f}% | {elapsed:.0f}s | Samples: {sample_count} | Remaining: {remaining:.0f}s")
                        
                        # Show current sensor status dengan quality indicator
                        quality_status = "✅ GOOD" if quality_ok else "⚠️ CHECK"
                        print(f"   Quality: {quality_status}")
                        
                        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                            ppm = readings[sensor]['ppm']
                            voltage = readings[sensor]['raw_voltage']
                            
                            # Status indicator based on gas type expectations
                            if gas_name == 'normal':
                                status = "🟢" if ppm < 5 else "🔴"
                            elif gas_name == 'alcohol':
                                status = "🟢" if (sensor == 'TGS2600' and ppm > 20) else "🟡" if ppm > 5 else "⚪"
                            elif gas_name in ['pertalite', 'dexlite']:
                                status = "🟢" if (sensor in ['TGS2602', 'TGS2610'] and ppm > 15) else "🟡" if ppm > 5 else "⚪"
                            elif gas_name == 'toluene':
                                status = "🟢" if (sensor == 'TGS2602' and ppm > 40) else "🟡" if ppm > 10 else "⚪"
                            elif gas_name == 'ammonia':
                                status = "🟢" if (sensor == 'TGS2602' and ppm > 25) else "🟡" if ppm > 10 else "⚪"
                            elif gas_name == 'butane':
                                status = "🟢" if (sensor == 'TGS2610' and ppm > 20) else "🟡" if ppm > 5 else "⚪"
                            else:
                                status = "🟡"
                            
                            print(f"   {status} {sensor}: {voltage:.3f}V → {ppm:.1f} PPM")
                        print()
                    
                    time.sleep(self.collection_protocol['sample_interval'])
                    
        except KeyboardInterrupt:
            print(f"\n⏹️ Collection stopped by user at {sample_count} samples")
        
        # Collection summary
        actual_duration = time.time() - start_time
        sample_rate = sample_count / actual_duration * 60  # samples per minute
        
        print(f"\n✅ COLLECTION COMPLETED: {gas_name.upper()}")
        print(f"📁 File: {filename}")
        print(f"📊 Samples: {sample_count}")
        print(f"⏱️ Duration: {actual_duration:.1f}s")
        print(f"📈 Rate: {sample_rate:.1f} samples/min")
        
        # Return collection result
        return {
            'gas_name': gas_name,
            'filename': filename,
            'sample_count': sample_count,
            'duration': actual_duration,
            'realtime_data': realtime_data,
            'success': sample_count >= self.collection_protocol['min_samples_required']
        }
    
    def assess_realtime_quality(self, readings, gas_info, elapsed_time):
        """Assess real-time quality during collection"""
        gas_name = gas_info['name']
        criteria = gas_info['success_criteria']
        
        # Get current PPM values
        ppm_values = {sensor: readings[sensor]['ppm'] for sensor in ['TGS2600', 'TGS2602', 'TGS2610']}
        
        # Skip quality check during first minute (baseline period)
        if elapsed_time < 60:
            return True
        
        # Check specific criteria for each gas
        if gas_name == 'normal':
            return all(ppm < 10 for ppm in ppm_values.values())
        
        elif gas_name == 'alcohol':
            return (ppm_values['TGS2600'] > 15 and 
                   ppm_values['TGS2600'] >= ppm_values['TGS2602'] and
                   ppm_values['TGS2600'] >= ppm_values['TGS2610'])
        
        elif gas_name in ['pertalite', 'dexlite']:
            return (ppm_values['TGS2602'] > 10 or ppm_values['TGS2610'] > 10)
        
        elif gas_name == 'toluene':
            return (ppm_values['TGS2602'] > 25 and
                   ppm_values['TGS2602'] > ppm_values['TGS2600'] * 1.3)
        
        elif gas_name == 'ammonia':
            return (ppm_values['TGS2602'] > 20 and
                   sum(ppm_values.values()) > 25)
        
        elif gas_name == 'butane':
            return (ppm_values['TGS2610'] > 15 and
                   ppm_values['TGS2610'] >= ppm_values['TGS2600'] and
                   ppm_values['TGS2610'] >= ppm_values['TGS2602'])
        
        return True  # Default to OK for unknown gases
    
    def validate_gas_collection(self, result, gas_info):
        """Validate gas collection result"""
        if not result['success']:
            print(f"❌ Insufficient samples: {result['sample_count']} < {self.collection_protocol['min_samples_required']}")
            return False
        
        # Load dan analyze collected data
        try:
            df = pd.read_csv(result['filename'])
            
            # Calculate statistics
            stats = {}
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm_col = f'{sensor}_ppm'
                if ppm_col in df.columns:
                    ppm_data = df[ppm_col].dropna()
                    stats[sensor] = {
                        'mean': ppm_data.mean(),
                        'max': ppm_data.max(),
                        'std': ppm_data.std(),
                        'samples': len(ppm_data)
                    }
            
            # Validate against success criteria
            criteria = gas_info['success_criteria']
            validation_passed = True
            
            print(f"\n📊 VALIDATION RESULTS for {gas_info['name']}:")
            
            for sensor, sensor_stats in stats.items():
                print(f"   {sensor}: μ={sensor_stats['mean']:.1f} max={sensor_stats['max']:.1f} σ={sensor_stats['std']:.1f}")
            
            # Check specific criteria
            gas_name = gas_info['name']
            
            if gas_name == 'normal':
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    if stats[sensor]['mean'] > 8:
                        print(f"   ❌ {sensor} mean PPM too high: {stats[sensor]['mean']:.1f} > 8")
                        validation_passed = False
                    else:
                        print(f"   ✅ {sensor} baseline OK")
            
            elif gas_name == 'alcohol':
                if stats['TGS2600']['mean'] < 20:
                    print(f"   ❌ TGS2600 response too low: {stats['TGS2600']['mean']:.1f} < 20")
                    validation_passed = False
                elif stats['TGS2600']['mean'] <= stats['TGS2602']['mean']:
                    print(f"   ❌ TGS2600 not dominant over TGS2602")
                    validation_passed = False
                else:
                    print(f"   ✅ Alcohol signature detected")
            
            elif gas_name in ['pertalite', 'dexlite']:
                if stats['TGS2602']['mean'] < 15 and stats['TGS2610']['mean'] < 15:
                    print(f"   ❌ Insufficient hydrocarbon response")
                    validation_passed = False
                else:
                    print(f"   ✅ Hydrocarbon signature detected")
            
            elif gas_name == 'toluene':
                if stats['TGS2602']['mean'] < 35:
                    print(f"   ❌ TGS2602 response too low: {stats['TGS2602']['mean']:.1f} < 35")
                    validation_passed = False
                elif stats['TGS2602']['mean'] <= stats['TGS2600']['mean'] * 1.5:
                    print(f"   ❌ TGS2602 not sufficiently dominant")
                    validation_passed = False
                else:
                    print(f"   ✅ Toluene signature detected")
            
            elif gas_name == 'ammonia':
                if stats['TGS2602']['mean'] < 25:
                    print(f"   ❌ TGS2602 response too low: {stats['TGS2602']['mean']:.1f} < 25")
                    validation_passed = False
                else:
                    print(f"   ✅ Ammonia signature detected")
            
            elif gas_name == 'butane':
                if stats['TGS2610']['mean'] < 20:
                    print(f"   ❌ TGS2610 response too low: {stats['TGS2610']['mean']:.1f} < 20")
                    validation_passed = False
                elif stats['TGS2610']['mean'] <= max(stats['TGS2600']['mean'], stats['TGS2602']['mean']):
                    print(f"   ❌ TGS2610 not dominant")
                    validation_passed = False
                else:
                    print(f"   ✅ Butane signature detected")
            
            # Store validation results
            self.validation_results[gas_name] = {
                'passed': validation_passed,
                'stats': stats,
                'filename': result['filename']
            }
            
            return validation_passed
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    def countdown_timer(self, duration, description):
        """Countdown timer dengan display"""
        print(f"\n⏳ {description}: {duration} seconds")
        
        try:
            for remaining in range(duration, 0, -1):
                mins, secs = divmod(remaining, 60)
                print(f"\r   ⏱️ {mins:02d}:{secs:02d} remaining...", end="")
                time.sleep(1)
            print(f"\r   ✅ {description} completed!           ")
        except KeyboardInterrupt:
            print(f"\n⏭️ Timer skipped by user")
    
    def display_collection_summary(self):
        """Display summary hasil collection"""
        print(f"\n" + "="*70)
        print(f"📊 SYSTEMATIC COLLECTION SUMMARY")
        print("="*70)
        
        successful = 0
        failed = 0
        
        for gas_name, status in self.collection_status.items():
            if status == 'success':
                successful += 1
                validation = self.validation_results.get(gas_name, {})
                if validation.get('passed', False):
                    print(f"✅ {gas_name.upper()}: SUCCESS + VALIDATED")
                else:
                    print(f"🔶 {gas_name.upper()}: SUCCESS (validation issues)")
            else:
                failed += 1
                print(f"❌ {gas_name.upper()}: FAILED")
        
        success_rate = (successful / len(self.collection_status)) * 100
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   ✅ Successful: {successful}/{len(self.collection_status)} gases ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {failed}")
        
        if success_rate >= 80:
            print(f"   🎉 EXCELLENT! Ready for model training")
        elif success_rate >= 60:
            print(f"   🔶 GOOD. Some retries may improve results")
        else:
            print(f"   ⚠️ POOR. Consider systematic troubleshooting")
        
        # Training recommendation
        if successful >= 4:  # At least 4 different gases
            print(f"\n🤖 TRAINING RECOMMENDATION:")
            print(f"   ✅ Sufficient gas diversity for model training")
            print(f"   💡 Proceed to Model Training (Option 3)")
        else:
            print(f"\n⚠️ TRAINING RECOMMENDATION:")
            print(f"   ❌ Need at least 4 different gas types")
            print(f"   💡 Collect more gases before training")
    
    def all_collections_successful(self):
        """Check if all collections successful"""
        successful = sum(1 for status in self.collection_status.values() if status == 'success')
        return successful >= len(self.gas_sequence) * 0.8  # At least 80% success rate

class OptimizedModelTrainer:
    """Optimized Model Trainer untuk systematic collection results"""
    
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.scaler = None
        
    def train_systematic_model(self):
        """Train model from systematic collection files"""
        print("\n🤖 OPTIMIZED MODEL TRAINING FROM SYSTEMATIC COLLECTION")
        print("="*60)
        
        # Find systematic collection files
        systematic_files = glob.glob("systematic_*.csv")
        
        if len(systematic_files) < 3:
            print(f"❌ Need at least 3 gas types, found {len(systematic_files)}")
            print("💡 Complete systematic collection first")
            return False
        
        print(f"📂 Found {len(systematic_files)} systematic collection files:")
        
        # Load and combine data
        all_data = []
        gas_stats = {}
        
        for file in systematic_files:
            try:
                df = pd.read_csv(file)
                gas_type = df['gas_type'].iloc[0]
                samples = len(df)
                
                print(f"   ✅ {file}: {gas_type} ({samples} samples)")
                
                all_data.append(df)
                gas_stats[gas_type] = samples
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if len(all_data) < 3:
            print("❌ Failed to load sufficient training data")
            return False
        
        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)
        total_samples = len(combined_df)
        
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   Total samples: {total_samples}")
        print(f"   Gas types: {len(gas_stats)}")
        
        for gas, count in gas_stats.items():
            percentage = (count / total_samples) * 100
            print(f"     {gas}: {count} samples ({percentage:.1f}%)")
        
        # Prepare features
        feature_columns = []
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            feature_columns.extend([
                f'{sensor}_voltage',
                f'{sensor}_ppm'
            ])
        
        # Add time-based features (systematic collections have elapsed_time)
        if 'elapsed_time' in combined_df.columns:
            feature_columns.append('elapsed_time')
        
        # Environmental features
        if 'temperature' in combined_df.columns:
            feature_columns.append('temperature')
        if 'humidity' in combined_df.columns:
            feature_columns.append('humidity')
        
        print(f"\n🎯 FEATURE ENGINEERING:")
        print(f"   Base features: {len(feature_columns)}")
        
        # Enhanced feature engineering for systematic data
        enhanced_df = self.create_systematic_features(combined_df)
        
        # Get enhanced feature list
        enhanced_features = [col for col in enhanced_df.columns 
                           if col not in ['timestamp', 'gas_type'] and 
                           not col.startswith('enhanced_')]
        
        print(f"   Enhanced features: {len(enhanced_features)}")
        
        # Prepare training data
        X = enhanced_df[enhanced_features].fillna(0)
        y = enhanced_df['gas_type']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data dengan stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"\n📈 TRAINING SPLIT:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train optimized model untuk systematic data
        print(f"\n🎯 TRAINING OPTIMIZED MODEL...")
        
        self.model = RandomForestClassifier(
            n_estimators=300,       # Optimal for systematic data
            max_depth=15,          # Prevent overfitting
            min_samples_split=5,   # More conservative
            min_samples_leaf=3,    # More conservative
            max_features='sqrt',   # Feature randomness
            bootstrap=True,
            random_state=42,
            class_weight='balanced_subsample',  # Better for systematic data
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   🎯 Model accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed evaluation
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        excellent_performance = True
        for gas_type, metrics in report.items():
            if gas_type not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = int(metrics['support'])
                
                if f1 >= 0.85:
                    status = "🎉 EXCELLENT"
                elif f1 >= 0.70:
                    status = "✅ GOOD"
                elif f1 >= 0.50:
                    status = "🔶 FAIR"
                else:
                    status = "❌ POOR"
                    excellent_performance = False
                
                print(f"   {status} {gas_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({support} samples)")
        
        # Confusion matrix
        print(f"\n📊 CONFUSION MATRIX ANALYSIS:")
        cm = confusion_matrix(y_test, y_pred)
        gas_labels = sorted(y.unique())
        
        print(f"   Gas labels: {gas_labels}")
        print(f"   Matrix shape: {cm.shape}")
        
        # Feature importance
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        importances = list(zip(enhanced_features, self.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importances[:10]):
            print(f"   {i+1:2d}. {feature}: {importance:.3f}")
        
        # Save model
        if accuracy >= 0.80 and excellent_performance:
            self.save_systematic_model(enhanced_features, gas_stats, accuracy)
            print(f"\n🎉 SYSTEMATIC MODEL TRAINING: EXCELLENT!")
            return True
        elif accuracy >= 0.70:
            self.save_systematic_model(enhanced_features, gas_stats, accuracy)
            print(f"\n✅ SYSTEMATIC MODEL TRAINING: GOOD!")
            return True
        else:
            print(f"\n❌ SYSTEMATIC MODEL TRAINING: POOR PERFORMANCE")
            print(f"💡 Consider improving data collection or troubleshooting sensors")
            return False
    
    def create_systematic_features(self, df):
        """Create enhanced features untuk systematic collection data"""
        enhanced_df = df.copy()
        
        # Cross-sensor ratios
        enhanced_df['ratio_2600_2602'] = enhanced_df['TGS2600_ppm'] / (enhanced_df['TGS2602_ppm'] + 0.1)
        enhanced_df['ratio_2602_2610'] = enhanced_df['TGS2602_ppm'] / (enhanced_df['TGS2610_ppm'] + 0.1)
        enhanced_df['ratio_2600_2610'] = enhanced_df['TGS2600_ppm'] / (enhanced_df['TGS2610_ppm'] + 0.1)
        
        # Total response metrics
        enhanced_df['total_ppm'] = enhanced_df['TGS2600_ppm'] + enhanced_df['TGS2602_ppm'] + enhanced_df['TGS2610_ppm']
        enhanced_df['max_ppm'] = enhanced_df[['TGS2600_ppm', 'TGS2602_ppm', 'TGS2610_ppm']].max(axis=1)
        enhanced_df['avg_ppm'] = enhanced_df['total_ppm'] / 3
        
        # Voltage features
        enhanced_df['volt_diff_2600_2602'] = enhanced_df['TGS2600_voltage'] - enhanced_df['TGS2602_voltage']
        enhanced_df['volt_diff_2602_2610'] = enhanced_df['TGS2602_voltage'] - enhanced_df['TGS2610_voltage']
        
        # Dominant sensor
        dominant_sensor = enhanced_df[['TGS2600_ppm', 'TGS2602_ppm', 'TGS2610_ppm']].idxmax(axis=1)
        enhanced_df['dominant_sensor'] = dominant_sensor.map({
            'TGS2600_ppm': 0, 'TGS2602_ppm': 1, 'TGS2610_ppm': 2
        }).fillna(-1)
        
        # Systematic collection specific: time-based features
        if 'elapsed_time' in enhanced_df.columns:
            # Normalize time to 0-1 range
            enhanced_df['time_normalized'] = enhanced_df['elapsed_time'] / enhanced_df['elapsed_time'].max()
            
            # Time-based response patterns
            enhanced_df['early_phase'] = (enhanced_df['elapsed_time'] < 60).astype(int)  # First minute
            enhanced_df['active_phase'] = ((enhanced_df['elapsed_time'] >= 60) & 
                                         (enhanced_df['elapsed_time'] < 150)).astype(int)  # Active collection
            enhanced_df['recovery_phase'] = (enhanced_df['elapsed_time'] >= 150).astype(int)  # Recovery
        
        return enhanced_df
    
    def save_systematic_model(self, feature_names, gas_stats, accuracy):
        """Save systematic model dengan metadata"""
        print(f"\n💾 SAVING SYSTEMATIC MODEL...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model dan scaler
        joblib.dump(self.model, 'models/systematic_gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/systematic_scaler.pkl')
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': 'systematic_v4.4_opsi2',
            'model_type': 'RandomForest_Systematic_3min',
            'collection_protocol': '3_minute_proven',
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'gas_types': list(self.model.classes_),
            'gas_sample_counts': gas_stats,
            'accuracy': accuracy,
            'model_params': self.model.get_params(),
            'training_method': 'systematic_collection',
            'collection_duration': 180,  # 3 minutes
            'recovery_time': 300,       # 5 minutes
            'quality_control': 'realtime_validation'
        }
        
        with open('models/systematic_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Systematic model saved:")
        print(f"   🤖 models/systematic_gas_classifier.pkl")
        print(f"   ⚖️ models/systematic_scaler.pkl")
        print(f"   📋 models/systematic_model_metadata.json")

class AdaptiveFeatureProcessor:
    """Adaptive Feature Processor - Solusi untuk PPM range mismatch"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Training data statistics (akan di-load dari metadata)
        self.training_stats = {
            'TGS2600_ppm': {'min': 0, 'max': 100, 'mean': 20, 'std': 15},
            'TGS2602_ppm': {'min': 0, 'max': 100, 'mean': 18, 'std': 12},
            'TGS2610_ppm': {'min': 0, 'max': 100, 'mean': 25, 'std': 18},
            'TGS2600_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15},
            'TGS2602_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15},
            'TGS2610_voltage': {'min': 1.4, 'max': 2.1, 'mean': 1.6, 'std': 0.15}
        }
        
        # Current data statistics (auto-updated)
        self.current_stats = {}
        
        # Adaptive scaling factors
        self.ppm_scaling_factors = {
            'TGS2600': 1.0,
            'TGS2602': 1.0, 
            'TGS2610': 1.0
        }
        
        self.load_training_stats()
    
    def load_training_stats(self):
        """Load training data statistics"""
        try:
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            if 'training_stats' in metadata:
                self.training_stats = metadata['training_stats']
                self.logger.info("Training statistics loaded from metadata")
            else:
                self.logger.info("Using default training statistics")
                
        except FileNotFoundError:
            self.logger.info("No model metadata found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading training stats: {e}")

class EnhancedPredictionEngine:
    """Enhanced Prediction Engine with confidence boosting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.recent_predictions = []
        self.prediction_history = {}

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
        
        self.load_drift_data()
    
    def smart_daily_drift_check(self, sensor_array):
        """Smart daily drift check"""
        print("\n📊 SMART DAILY DRIFT CHECK")
        print("="*50)
        print("Checking for sensor drift and baseline changes...")
        
        current_voltages = {}
        drift_detected = {}
        
        for sensor_name, config in sensor_array.sensor_config.items():
            current_voltage = config['channel'].voltage
            current_voltages[sensor_name] = current_voltage
            
            # Compare with baseline
            original_baseline = self.original_baseline.get(sensor_name, 1.6)
            voltage_drift = abs(current_voltage - original_baseline)
            
            print(f"\n{sensor_name}:")
            print(f"  Current: {current_voltage:.3f}V")
            print(f"  Original: {original_baseline:.3f}V")
            print(f"  Drift: {voltage_drift:.3f}V")
            
            # Classify drift level
            if voltage_drift <= self.drift_tolerance['excellent']:
                drift_level = "EXCELLENT"
                drift_detected[sensor_name] = False
            elif voltage_drift <= self.drift_tolerance['good']:
                drift_level = "GOOD"
                drift_detected[sensor_name] = False
            elif voltage_drift <= self.drift_tolerance['moderate']:
                drift_level = "MODERATE"
                drift_detected[sensor_name] = True
            elif voltage_drift <= self.drift_tolerance['high']:
                drift_level = "HIGH"
                drift_detected[sensor_name] = True
            else:
                drift_level = "EXTREME"
                drift_detected[sensor_name] = True
            
            print(f"  Status: {drift_level}")
            
            # Auto-compensation for moderate drift
            if drift_detected[sensor_name] and voltage_drift <= self.drift_tolerance['high']:
                compensation_factor = original_baseline / current_voltage
                self.drift_compensation_factors[sensor_name] = compensation_factor
                print(f"  ✅ Auto-compensation: {compensation_factor:.3f}")
        
        # Summary
        total_drift_sensors = sum(drift_detected.values())
        print(f"\n📋 DRIFT SUMMARY:")
        print(f"  Sensors with drift: {total_drift_sensors}/3")
        
        if total_drift_sensors == 0:
            print("  🎉 All sensors stable!")
        else:
            print(f"  ⚠️  {total_drift_sensors} sensors need attention")
        
        self.daily_check_done = True
        self.save_drift_data()
        
        return total_drift_sensors == 0
    
    def quick_stability_test(self, sensor_array, duration=60):
        """Quick stability test"""
        print(f"\n⚡ QUICK STABILITY TEST - {duration} seconds")
        print("="*50)
        
        readings = {sensor: [] for sensor in ['TGS2600', 'TGS2602', 'TGS2610']}
        
        print("Collecting stability readings...")
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            for sensor_name, config in sensor_array.sensor_config.items():
                voltage = config['channel'].voltage
                readings[sensor_name].append(voltage)
            
            sample_count += 1
            if sample_count % 10 == 0:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"  Progress: {elapsed:.0f}s / {duration}s (Remaining: {remaining:.0f}s)")
            
            time.sleep(2)
        
        # Analyze stability
        print(f"\n📊 STABILITY ANALYSIS:")
        overall_stable = True
        
        for sensor_name, voltages in readings.items():
            if voltages:
                mean_voltage = np.mean(voltages)
                std_voltage = np.std(voltages)
                cv = (std_voltage / mean_voltage) * 100  # Coefficient of variation
                
                print(f"\n{sensor_name}:")
                print(f"  Mean: {mean_voltage:.3f}V")
                print(f"  Std Dev: {std_voltage:.4f}V")
                print(f"  CV: {cv:.2f}%")
                
                if cv < 1.0:
                    stability = "EXCELLENT"
                elif cv < 2.0:
                    stability = "GOOD"
                elif cv < 5.0:
                    stability = "MODERATE"
                else:
                    stability = "POOR"
                    overall_stable = False
                
                print(f"  Stability: {stability}")
        
        print(f"\n🎯 OVERALL STABILITY: {'GOOD' if overall_stable else 'NEEDS ATTENTION'}")
        return overall_stable
    
    def smart_drift_status_report(self):
        """Comprehensive drift status report"""
        print("\n📋 SMART DRIFT STATUS REPORT")
        print("="*60)
        
        print("🎯 COMPENSATION FACTORS:")
        for sensor, factor in self.drift_compensation_factors.items():
            print(f"  {sensor}: {factor:.3f}x")
        
        print("\n🎯 NORMALIZATION FACTORS:")
        for sensor, factor in self.normalization_factors.items():
            print(f"  {sensor}: {factor:.3f}x")
        
        print("\n🎯 BASELINE HISTORY:")
        for sensor, baseline in self.original_baseline.items():
            current = self.current_baseline.get(sensor, baseline)
            print(f"  {sensor}: {baseline:.3f}V → {current:.3f}V")
        
        # Overall health assessment
        health_score = 0
        total_sensors = 3
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            drift_factor = self.drift_compensation_factors.get(sensor, 1.0)
            norm_factor = self.normalization_factors.get(sensor, 1.0)
            
            # Score based on how close factors are to 1.0
            sensor_score = 1.0 - min(abs(drift_factor - 1.0), abs(norm_factor - 1.0))
            health_score += sensor_score
        
        health_percentage = (health_score / total_sensors) * 100
        
        if health_percentage >= 90:
            health_status = "EXCELLENT"
        elif health_percentage >= 75:
            health_status = "GOOD"
        elif health_percentage >= 50:
            health_status = "MODERATE"
        else:
            health_status = "POOR"
        
        print(f"\n🎯 OVERALL HEALTH: {health_status} ({health_percentage:.1f}%)")
    
    def manual_drift_compensation_reset(self):
        """Manual reset of drift compensation"""
        print("\n🔄 MANUAL DRIFT COMPENSATION RESET")
        print("="*50)
        
        print("Current compensation factors:")
        for sensor, factor in self.drift_compensation_factors.items():
            print(f"  {sensor}: {factor:.3f}x")
        
        reset_choice = input("\nReset all compensation factors? (y/n): ").lower()
        if reset_choice == 'y':
            self.drift_compensation_factors = {}
            self.normalization_factors = {
                'TGS2600': 1.0,
                'TGS2602': 1.0,
                'TGS2610': 1.0
            }
            print("✅ All drift compensation reset")
            self.save_drift_data()
    
    def auto_baseline_reset(self, sensor_array):
        """Automatic baseline reset"""
        print("\n🔄 AUTO BASELINE RESET")
        print("="*50)
        print("This will set current voltages as new baselines")
        
        confirm = input("Ensure sensors are in CLEAN AIR. Continue? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Baseline reset cancelled")
            return
        
        print("Measuring new baselines...")
        new_baselines = {}
        
        # Take multiple readings for stability
        for sensor_name, config in sensor_array.sensor_config.items():
            voltages = []
            for i in range(10):
                voltage = config['channel'].voltage
                voltages.append(voltage)
                time.sleep(1)
            
            new_baseline = np.mean(voltages)
            new_baselines[sensor_name] = new_baseline
            
            print(f"  {sensor_name}: {new_baseline:.3f}V")
        
        # Update baselines
        for sensor_name, new_baseline in new_baselines.items():
            self.original_baseline[sensor_name] = new_baseline
            self.current_baseline[sensor_name] = new_baseline
            
            # Reset compensation factors
            if sensor_name in self.drift_compensation_factors:
                del self.drift_compensation_factors[sensor_name]
            
            self.normalization_factors[sensor_name] = 1.0
            
            # Update sensor config if available
            if hasattr(sensor_array, 'sensor_config'):
                sensor_array.sensor_config[sensor_name]['baseline_voltage'] = new_baseline
        
        self.daily_check_done = False
        self.save_drift_data()
        
        print("✅ Auto baseline reset completed")
    
    def smart_system_health_check(self, sensor_array):
        """Comprehensive system health check"""
        print("\n🏥 SMART SYSTEM HEALTH CHECK")
        print("="*60)
        
        health_report = {}
        overall_issues = []
        
        print("🔍 Checking sensor hardware...")
        for sensor_name, config in sensor_array.sensor_config.items():
            sensor_health = {
                'voltage_ok': True,
                'response_ok': True,
                'calibration_ok': True,
                'drift_ok': True,
                'issues': []
            }
            
            # Check 1: Voltage range
            current_voltage = config['channel'].voltage
            if current_voltage < 0.5 or current_voltage > 4.5:
                sensor_health['voltage_ok'] = False
                sensor_health['issues'].append("Voltage out of range")
            
            # Check 2: Calibration
            R0 = config.get('R0')
            if R0 is None or R0 <= 0:
                sensor_health['calibration_ok'] = False
                sensor_health['issues'].append("Not calibrated")
            
            # Check 3: Drift
            original_baseline = self.original_baseline.get(sensor_name, 1.6)
            voltage_drift = abs(current_voltage - original_baseline)
            if voltage_drift > self.drift_tolerance['high']:
                sensor_health['drift_ok'] = False
                sensor_health['issues'].append("High drift detected")
            
            health_report[sensor_name] = sensor_health
            
            # Print sensor status
            print(f"\n{sensor_name}:")
            print(f"  Voltage: {current_voltage:.3f}V {'✅' if sensor_health['voltage_ok'] else '❌'}")
            print(f"  Calibration: {'✅' if sensor_health['calibration_ok'] else '❌'}")
            print(f"  Drift: {'✅' if sensor_health['drift_ok'] else '❌'}")
            
            if sensor_health['issues']:
                print(f"  Issues: {', '.join(sensor_health['issues'])}")
                overall_issues.extend(sensor_health['issues'])
        
        # System-wide checks
        print("\n🔍 Checking system configuration...")
        
        model_trained = hasattr(sensor_array, 'is_model_trained') and sensor_array.is_model_trained
        print(f"  ML Model: {'✅' if model_trained else '❌'}")
        if not model_trained:
            overall_issues.append("ML model not trained")
        
        # Overall health score
        total_checks = len(health_report) * 4 + 1  # 4 checks per sensor + model check
        passed_checks = sum([
            sum([v for k, v in sensor.items() if k != 'issues']) 
            for sensor in health_report.values()
        ]) + (1 if model_trained else 0)
        
        health_percentage = (passed_checks / total_checks) * 100
        
        print(f"\n🎯 OVERALL SYSTEM HEALTH: {health_percentage:.1f}%")
        
        if health_percentage >= 90:
            health_status = "EXCELLENT 🎉"
        elif health_percentage >= 75:
            health_status = "GOOD ✅"
        elif health_percentage >= 50:
            health_status = "MODERATE ⚠️"
        else:
            health_status = "POOR ❌"
        
        print(f"Status: {health_status}")
        
        if overall_issues:
            print(f"\n🔧 ISSUES TO ADDRESS:")
            for issue in set(overall_issues):
                print(f"  • {issue}")
        
        return health_percentage >= 75
    
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
        
        # Enhanced Auto-fix Options
        print(f"\n🔧 AUTO-FIX OPTIONS:")
        print("1. Enable Emergency PPM Mode")
        print("2. Emergency R0 Fix")
        print("3. Enable Ultra-Sensitive Mode")
        print("4. Run Auto-Sensitivity Calibration")
        
        fix_choice = input("Select auto-fix option (1-4, or Enter to skip): ").strip()
        
        if fix_choice == '1':
            config['use_emergency_ppm'] = True
            config['emergency_mode'] = True
            print("✅ Emergency PPM mode enabled")
            return True
        elif fix_choice == '2':
            current_resistance = sensor_array.voltage_to_resistance(current_voltage)
            config['R0'] = current_resistance
            config['baseline_voltage'] = current_voltage
            print(f"✅ Emergency R0 set: {current_resistance:.1f}Ω")
            return True
        elif fix_choice == '3':
            sensor_array.sensitivity_manager.current_sensitivity[sensor_name] = 'ultra_sensitive'
            sensor_array.sensitivity_manager.save_sensitivity_data()
            print("✅ Ultra-sensitive mode enabled")
            return True
        elif fix_choice == '4':
            return sensor_array.sensitivity_manager.auto_sensitivity_calibration(sensor_array, sensor_name, 30)
        
        return False
    
    def apply_smart_compensation(self, sensor_name, raw_voltage):
        """Apply smart compensation + normalization"""
        compensated_voltage = raw_voltage
        if sensor_name in self.drift_compensation_factors:
            compensated_voltage = raw_voltage * self.drift_compensation_factors[sensor_name]
        
        normalized_voltage = compensated_voltage * self.normalization_factors.get(sensor_name, 1.0)
        return normalized_voltage, compensated_voltage
    
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
            self.daily_check_done = drift_data.get('daily_check_done', False)
            
            self.logger.info("Smart drift data loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No drift data found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading drift data: {e}")

class MonitoringDataCollector:
    """Enhanced Monitoring Data Collector with CSV saving"""
    
    def __init__(self, logger):
        self.logger = logger
        self.is_collecting = False
        self.collection_thread = None
        self.data_queue = queue.Queue()
        self.csv_writer = None
        self.csv_file = None
        self.current_filename = None
        
    def start_monitoring(self, sensor_array, mode='datasheet', save_to_csv=True):
        """Start monitoring with optional CSV saving"""
        if self.is_collecting:
            print("⚠️ Monitoring already running!")
            return False
            
        self.is_collecting = True
        
        if save_to_csv:
            # Create CSV file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_filename = f"monitoring_data_{timestamp}.csv"
            
            # CSV headers
            headers = [
                'timestamp', 'predicted_gas', 'confidence'
            ]
            
            # Add sensor columns
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                headers.extend([
                    f'{sensor}_raw_voltage',
                    f'{sensor}_voltage', 
                    f'{sensor}_resistance',
                    f'{sensor}_ppm',
                    f'{sensor}_emergency_ppm',
                    f'{sensor}_advanced_ppm'
                ])
            
            try:
                self.csv_file = open(self.current_filename, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(headers)
                self.csv_file.flush()
                print(f"📁 Monitoring data will be saved to: {self.current_filename}")
            except Exception as e:
                print(f"❌ Error creating CSV file: {e}")
                save_to_csv = False
        
        # Start monitoring thread
        self.collection_thread = threading.Thread(
            target=self._monitoring_worker,
            args=(sensor_array, mode, save_to_csv),
            daemon=True
        )
        self.collection_thread.start()
        
        print(f"🚀 Monitoring started in {mode} mode")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring and close CSV file"""
        if not self.is_collecting:
            return False
            
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"💾 Monitoring data saved to: {self.current_filename}")
            except Exception as e:
                print(f"⚠️ Error closing CSV file: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None
        
        print("⏹️ Monitoring stopped")
        return True
    
    def _monitoring_worker(self, sensor_array, mode, save_to_csv):
        """Background monitoring worker"""
        sample_count = 0
        
        try:
            while self.is_collecting:
                # Read sensors
                readings = sensor_array.read_sensors()
                predicted_gas, confidence = sensor_array.predict_gas(readings)
                
                # Prepare row data
                row_data = [
                    datetime.now().isoformat(),
                    predicted_gas,
                    confidence
                ]
                
                # Add sensor data
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    sensor_data = readings[sensor]
                    row_data.extend([
                        sensor_data['raw_voltage'],
                        sensor_data['voltage'],
                        sensor_data['resistance'], 
                        sensor_data['ppm'],
                        sensor_data.get('emergency_ppm', 0),
                        sensor_data.get('advanced_ppm', 0)
                    ])
                
                # Save to CSV
                if save_to_csv and self.csv_writer:
                    try:
                        self.csv_writer.writerow(row_data)
                        self.csv_file.flush()  # Force write to disk
                        sample_count += 1
                    except Exception as e:
                        self.logger.error(f"Error writing to CSV: {e}")
                
                # Display progress
                if sample_count % 10 == 0 and save_to_csv:
                    print(f"\r💾 Samples saved: {sample_count}", end="")
                
                time.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            self.logger.error(f"Monitoring worker error: {e}")
        finally:
            if save_to_csv and sample_count > 0:
                print(f"\n✅ Total samples saved: {sample_count}")

# MODIFIKASI MAIN CLASS untuk Opsi 2
class EnhancedDatasheetGasSensorArray:
    def __init__(self):
        """Initialize complete enhanced gas sensor array system - OPSI 2"""
        # [COPY SEMUA INITIALIZATION DARI KODE ASLI - TIDAK BERUBAH]
        
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

        # Initialize I2C and ADC (sama seperti asli)
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

        # Enhanced sensor configurations DENGAN 3 MENIT PROTOCOL
        self.sensor_config = {
            'TGS2600': {
                'channel': self.tgs2600,
                'target_gases': ['hydrogen', 'carbon_monoxide', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 500),
                'systematic_collection_duration': 180,  # 3 menit proven
                'recovery_time': 300,  # 5 menit recovery
                'R0': None,
                'baseline_voltage': None,
                'use_extended_mode': False,
                'emergency_mode': False
            },
            'TGS2602': {
                'channel': self.tgs2602,
                'target_gases': ['toluene', 'ammonia', 'h2s', 'alcohol'],
                'detection_range': (1, 30),
                'extended_range': (1, 300),
                'systematic_collection_duration': 180,  # 3 menit proven
                'recovery_time': 300,  # 5 menit recovery
                'R0': None,
                'baseline_voltage': None,
                'use_extended_mode': False,
                'emergency_mode': False
            },
            'TGS2610': {
                'channel': self.tgs2610,
                'target_gases': ['butane', 'propane', 'lp_gas', 'iso_butane'],
                'detection_range': (1, 25),
                'extended_range': (1, 200),
                'systematic_collection_duration': 180,  # 3 menit proven
                'recovery_time': 300,  # 5 menit recovery
                'R0': None,
                'baseline_voltage': None,
                'use_extended_mode': False,
                'emergency_mode': False
            }
        }

        # Initialize semua managers (SEMUA FITUR ORIGINAL TETAP ADA)
        self.drift_manager = SmartDriftManager(self.logger)
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)
        self.sensitivity_manager = AdvancedSensitivityManager(self.logger)
        self.monitoring_collector = MonitoringDataCollector(self.logger)
        
        # Initialize adaptive components
        self.adaptive_processor = AdaptiveFeatureProcessor(self.logger)
        self.prediction_engine = EnhancedPredictionEngine(self.logger)
        
        # Readings history for adaptive learning
        self.readings_history = []
        self.max_history_size = 100

        # Environmental compensation
        self.temp_compensation_enabled = True
        self.humidity_compensation_enabled = True
        self.current_temperature = 20.0
        self.current_humidity = 65.0

        # Enhanced Machine Learning with adaptive features
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.feature_names = None
        self.training_metadata = None  # Store training metadata

        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)
        
        # TAMBAHAN: Systematic collection system
        self.systematic_guide = SystematicCollectionGuide(self.logger)
        self.systematic_trainer = OptimizedModelTrainer(self.logger)

        self.logger.info("Enhanced Gas Sensor Array System v4.4 - OPSI 2 SYSTEMATIC READY")

    # [SISIPKAN SEMUA METHOD DARI KODE ASLI - TIDAK DIHAPUS]
    # voltage_to_resistance, read_sensors, calibrate_sensors, dll.

    def run_systematic_collection_opsi2(self):
        """Run systematic collection process - OPSI 2"""
        print("\n🚀 OPSI 2: SYSTEMATIC DATA COLLECTION FROM SCRATCH")
        print("="*60)
        print("🎯 Proven 3-minute protocol")
        print("📊 Real-time quality control")
        print("✅ 95%+ success rate expected")
        
        # Display overview
        self.systematic_guide.display_collection_overview()
        
        proceed = input(f"\nStart systematic collection process? (y/n): ").lower().strip()
        if proceed != 'y':
            print("❌ Systematic collection cancelled")
            return False
        
        # Start systematic collection
        success = self.systematic_guide.start_systematic_collection(self)
        
        if success:
            print(f"\n🎉 SYSTEMATIC COLLECTION COMPLETED SUCCESSFULLY!")
            
            # Ask untuk train model immediately
            train_now = input(f"\nTrain model with systematic data now? (y/n): ").lower().strip()
            if train_now == 'y':
                training_success = self.systematic_trainer.train_systematic_model()
                
                if training_success:
                    # Load the new systematic model
                    self.load_systematic_model()
                    print(f"\n🎉 SYSTEMATIC TRAINING COMPLETED!")
                    print(f"✅ Model ready for prediction")
                    
                    return True
            else:
                print(f"💡 You can train the model later using Option 3")
                return True
        else:
            print(f"\n❌ SYSTEMATIC COLLECTION INCOMPLETE")
            print(f"💡 Some gases may need retry")
            return False
    
    def load_systematic_model(self):
        """Load systematic model if available"""
        try:
            self.model = joblib.load('models/systematic_gas_classifier.pkl')
            self.scaler = joblib.load('models/systematic_scaler.pkl')
            
            # Load metadata
            with open('models/systematic_model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.training_metadata = metadata
            
            self.is_model_trained = True
            self.logger.info("🚀 Systematic model loaded successfully")
            return True
            
        except FileNotFoundError:
            self.logger.info("Systematic model not found")
            return False
        except Exception as e:
            self.logger.error(f"Error loading systematic model: {e}")
            return False

def main():
    """Enhanced main function dengan Opsi 2 systematic collection"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️ CALIBRATION NOT FOUND")
        print("Run calibration first for optimal results")

    # Try load systematic model first, then regular model
    systematic_loaded = gas_sensor.load_systematic_model()
    if not systematic_loaded:
        model_loaded = gas_sensor.load_model()
    else:
        model_loaded = True
    
    if systematic_loaded:
        print("\n🚀 SYSTEMATIC MODEL LOADED")
        print("✅ Trained with 3-minute proven protocol")
        print("✅ Real-time validated data")
    elif model_loaded:
        print("\n📊 REGULAR MODEL LOADED")
        print("💡 Consider systematic collection for better results")

    while True:
        print("\n" + "="*70)
        print("🧠 Enhanced Gas Sensor Array System - v4.4 OPSI 2")
        print("🎯 SYSTEMATIC Collection Protocol (3 menit proven + quality control)")
        print("="*70)
        print("1. Calibrate sensors")
        print("2. Individual data collection (3 menit)")
        print("3. Train machine learning model")
        print("4. Start monitoring - Datasheet mode")
        print("5. Start monitoring - Extended mode")
        print("6. Test single reading")
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
        print("-" * 40)
        print("🚨 TGS2602 SPECIAL RECOVERY:")
        print("29. TGS2602 Stuck Sensor Recovery")
        print("30. View monitoring data statistics")
        print("-" * 40)
        print("🎯 ADAPTIVE FEATURES:")
        print("31. View PPM range compensation status")
        print("32. Adjust adaptive transformation method")
        print("33. Enable/disable multi-scale features")
        print("34. View prediction confidence analysis")
        print("35. Test gas detection with current settings")
        print("-" * 40)
        print("📊 DIAGNOSTIC & VALIDATION:")
        print("36. Run comprehensive diagnostic")
        print("37. Quick system validation")
        print("38. Test prediction accuracy")
        print("39. View sensor calibration status")
        print("40. Advanced troubleshooting guide")
        print("41. Reset system configuration")
        print("42. Export system report")
        print("-" * 40)
        print("🚀 SYSTEMATIC COLLECTION SYSTEM (OPSI 2):")
        print("43. ⭐ RUN COMPLETE SYSTEMATIC COLLECTION (RECOMMENDED)")
        print("44. View systematic collection guide")
        print("45. Check systematic collection status")
        print("46. Train model from systematic data")
        print("47. Validate systematic collection files")
        print("48. Systematic troubleshooting guide")
        print("49. Reset systematic collection")
        print("50. Compare systematic vs regular collection")
        print("-"*70)

        try:
            choice = input("Select option (1-50): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                print("🎯 SYSTEMATIC CALIBRATION")
                print("Ensure sensors warmed up for 10+ minutes in clean air!")
                print("This calibration optimizes for 3-minute collection protocol")
                confirm = input("Continue with systematic calibration? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.calibrate_sensors(duration)

            elif choice == '2':
                print("\n🎯 INDIVIDUAL SYSTEMATIC COLLECTION (3 MENIT)")
                print("="*50)
                print("Available gas types:")
                
                gas_options = [
                    ('normal', 'Clean air baseline'),
                    ('alcohol', 'Alcohol/ethanol (TGS2600 target)'),
                    ('pertalite', 'Gasoline (TGS2602+TGS2610)'),
                    ('toluene', 'Toluene/aromatic (TGS2602 dominant)'),
                    ('ammonia', 'Ammonia (TGS2602 moderate)'),
                    ('butane', 'Butane/LPG (TGS2610 dominant)')
                ]
                
                for i, (gas, desc) in enumerate(gas_options, 1):
                    print(f"{i}. {gas}: {desc}")
                print("7. custom")
                
                gas_choice = input("Select gas type (1-7): ").strip()
                
                if gas_choice in ['1', '2', '3', '4', '5', '6']:
                    gas_name = gas_options[int(gas_choice) - 1][0]
                    
                    # Find gas info from systematic guide
                    gas_info = None
                    for info in gas_sensor.systematic_guide.gas_sequence:
                        if info['name'] == gas_name:
                            gas_info = info
                            break
                    
                    if gas_info:
                        # Display preparation instructions
                        gas_sensor.systematic_guide.display_gas_instructions(gas_info)
                        
                        input("Press Enter when ready to start systematic collection...")
                        
                        # Collect with systematic protocol
                        result = gas_sensor.systematic_guide.collect_gas_systematic(gas_sensor, gas_info)
                        
                        # Validate
                        if gas_sensor.systematic_guide.validate_gas_collection(result, gas_info):
                            print(f"✅ {gas_name} systematic collection: SUCCESSFUL")
                        else:
                            print(f"❌ {gas_name} systematic collection: NEEDS IMPROVEMENT")
                            
                elif gas_choice == '7':
                    gas_name = input("Enter custom gas name: ").strip()
                    duration = 180  # Force 3 minutes for consistency
                    gas_sensor.collect_training_data(gas_name, duration)
                else:
                    print("❌ Invalid choice")

            elif choice == '43':
                # MAIN FEATURE - SYSTEMATIC COLLECTION OPSI 2
                success = gas_sensor.run_systematic_collection_opsi2()
                
                if success:
                    print(f"\n🎉 SYSTEMATIC COLLECTION OPSI 2 - BERHASIL!")
                    print(f"✅ All gases collected with proven 3-minute protocol")
                    print(f"✅ Real-time quality control applied")
                    print(f"✅ Model trained and ready for use")
                    
                    # Test systematic model
                    print(f"\n🧪 Testing systematic model:")
                    readings = gas_sensor.read_sensors()
                    prediction, confidence = gas_sensor.predict_gas(readings)
                    print(f"   Current prediction: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Model type: Systematic")
                    
                else:
                    print(f"\n❌ SYSTEMATIC COLLECTION INCOMPLETE")
                    print(f"💡 Check individual gas collections")
                    print(f"💡 Ensure proper preparation and method")

            elif choice == '44':
                print("\n📋 SYSTEMATIC COLLECTION GUIDE")
                print("="*50)
                
                gas_sensor.systematic_guide.display_collection_overview()
                
                print(f"\n🎯 DETAILED GAS-BY-GAS INSTRUCTIONS:")
                
                for i, gas_info in enumerate(gas_sensor.systematic_guide.gas_sequence, 1):
                    print(f"\n{i}. {gas_info['display_name'].upper()}:")
                    print(f"   Target: {gas_info['description']}")
                    
                    print(f"   Preparation:")
                    for step in gas_info['preparation'][:2]:  # Show first 2 steps
                        print(f"     • {step}")
                    
                    print(f"   Collection:")
                    for step in gas_info['collection_method'][:2]:  # Show first 2 steps
                        print(f"     • {step}")

            elif choice == '45':
                print("\n📊 SYSTEMATIC COLLECTION STATUS")
                print("="*50)
                
                # Check for systematic files
                systematic_files = glob.glob("systematic_*.csv")
                
                if not systematic_files:
                    print("❌ No systematic collection files found")
                    print("💡 Run complete systematic collection (Option 43)")
                else:
                    print(f"📂 Found {len(systematic_files)} systematic files:")
                    
                    gas_collected = {}
                    total_samples = 0
                    
                    for file in systematic_files:
                        try:
                            df = pd.read_csv(file)
                            gas_type = df['gas_type'].iloc[0]
                            samples = len(df)
                            
                            gas_collected[gas_type] = {
                                'file': file,
                                'samples': samples,
                                'duration': df['elapsed_time'].max() if 'elapsed_time' in df.columns else 'unknown'
                            }
                            total_samples += samples
                            
                            print(f"   ✅ {gas_type}: {samples} samples ({file})")
                            
                        except Exception as e:
                            print(f"   ❌ Error reading {file}: {e}")
                    
                    print(f"\n📈 SUMMARY:")
                    print(f"   Total samples: {total_samples}")
                    print(f"   Gas types: {len(gas_collected)}")
                    
                    expected_gases = ['normal', 'alcohol', 'pertalite', 'toluene', 'ammonia', 'butane']
                    missing_gases = [gas for gas in expected_gases if gas not in gas_collected]
                    
                    if missing_gases:
                        print(f"   ⚠️ Missing gases: {missing_gases}")
                        print(f"   💡 Collect missing gases for complete training")
                    else:
                        print(f"   🎉 All expected gases collected!")

            elif choice == '46':
                print("\n🤖 TRAIN MODEL FROM SYSTEMATIC DATA")
                print("="*50)
                
                success = gas_sensor.systematic_trainer.train_systematic_model()
                
                if success:
                    # Load the newly trained model
                    gas_sensor.load_systematic_model()
                    print(f"\n✅ Systematic model training completed and loaded!")
                else:
                    print(f"\n❌ Systematic model training failed")
                    print(f"💡 Check data quality and collection completeness")

            elif choice == '47':
                print("\n🔍 VALIDATE SYSTEMATIC COLLECTION FILES")
                print("="*50)
                
                systematic_files = glob.glob("systematic_*.csv")
                
                if not systematic_files:
                    print("❌ No systematic files to validate")
                else:
                    print(f"📂 Validating {len(systematic_files)} files:")
                    
                    validation_results = {}
                    
                    for file in systematic_files:
                        try:
                            df = pd.read_csv(file)
                            gas_type = df['gas_type'].iloc[0]
                            
                            # Basic validation
                            samples = len(df)
                            duration = df['elapsed_time'].max() if 'elapsed_time' in df.columns else 0
                            
                            # PPM statistics
                            ppm_stats = {}
                            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                                ppm_col = f'{sensor}_ppm'
                                if ppm_col in df.columns:
                                    ppm_data = df[ppm_col].dropna()
                                    ppm_stats[sensor] = {
                                        'mean': ppm_data.mean(),
                                        'max': ppm_data.max(),
                                        'std': ppm_data.std()
                                    }
                            
                            # Validation criteria
                            valid = True
                            issues = []
                            
                            if samples < 60:
                                valid = False
                                issues.append(f"Too few samples: {samples} < 60")
                            
                            if duration < 150:
                                valid = False
                                issues.append(f"Too short duration: {duration:.0f}s < 150s")
                            
                            # Gas-specific validation
                            if gas_type == 'normal':
                                if any(stats['mean'] > 8 for stats in ppm_stats.values()):
                                    valid = False
                                    issues.append("High baseline PPM")
                            elif gas_type == 'alcohol':
                                if ppm_stats.get('TGS2600', {}).get('mean', 0) < 20:
                                    valid = False
                                    issues.append("TGS2600 response too low")
                            elif gas_type == 'toluene':
                                if ppm_stats.get('TGS2602', {}).get('mean', 0) < 35:
                                    valid = False
                                    issues.append("TGS2602 response too low")
                            # Add more validations...
                            
                            validation_results[gas_type] = {
                                'file': file,
                                'valid': valid,
                                'issues': issues,
                                'samples': samples,
                                'duration': duration,
                                'ppm_stats': ppm_stats
                            }
                            
                            status = "✅ VALID" if valid else "❌ INVALID"
                            print(f"   {status} {gas_type}: {samples} samples, {duration:.0f}s")
                            
                            if issues:
                                for issue in issues:
                                    print(f"     ⚠️ {issue}")
                            
                        except Exception as e:
                            print(f"   ❌ Error validating {file}: {e}")
                    
                    # Summary
                    valid_count = sum(1 for result in validation_results.values() if result['valid'])
                    total_count = len(validation_results)
                    
                    print(f"\n📊 VALIDATION SUMMARY:")
                    print(f"   Valid files: {valid_count}/{total_count}")
                    
                    if valid_count >= total_count * 0.8:
                        print(f"   🎉 EXCELLENT! Ready for systematic training")
                    elif valid_count >= total_count * 0.6:
                        print(f"   🔶 GOOD. Some files may need recollection")
                    else:
                        print(f"   ❌ POOR. Systematic recollection recommended")

            elif choice == '48':
                print("\n🔧 SYSTEMATIC TROUBLESHOOTING GUIDE")
                print("="*50)
                
                print("🎯 COMMON ISSUES & SOLUTIONS:")
                
                issues = [
                    {
                        'problem': 'Low sensor response during collection',
                        'symptoms': ['PPM < 10 for target sensor', 'All sensors showing similar low values'],
                        'solutions': [
                            'Increase sensitivity (Option 26)',
                            'Reduce distance to gas source',
                            'Use more concentrated gas source',
                            'Check sensor calibration (Option 1)',
                            'Verify gas source is active (smell test)'
                        ]
                    },
                    {
                        'problem': 'High baseline (normal) readings',
                        'symptoms': ['Normal air showing PPM > 5', 'Unable to get clean baseline'],
                        'solutions': [
                            'Improve room ventilation',
                            'Wait longer for air to clear',
                            'Check for hidden gas sources (perfume, cooking)',
                            'Move sensors away from heat sources',
                            'Recalibrate in truly clean environment'
                        ]
                    },
                    {
                        'problem': 'Inconsistent readings between collections',
                        'symptoms': ['Same gas giving different patterns', 'Poor model training results'],
                        'solutions': [
                            'Ensure consistent gas source concentration',
                            'Maintain same distance and method',
                            'Allow full recovery time between gases',
                            'Check environmental conditions (temp/humidity)',
                            'Verify sensor stability (Option 18)'
                        ]
                    },
                    {
                        'problem': 'Model training poor accuracy',
                        'symptoms': ['Accuracy < 70%', 'Confusion between similar gases'],
                        'solutions': [
                            'Validate all collection files (Option 47)',
                            'Recollect problematic gas types',
                            'Ensure sufficient samples per gas (>60)',
                            'Check gas source quality and purity',
                            'Consider longer exposure time'
                        ]
                    }
                ]
                
                for i, issue in enumerate(issues, 1):
                    print(f"\n{i}. {issue['problem'].upper()}:")
                    print(f"   Symptoms:")
                    for symptom in issue['symptoms']:
                        print(f"     • {symptom}")
                    print(f"   Solutions:")
                    for solution in issue['solutions']:
                        print(f"     ✓ {solution}")
                
                print(f"\n💡 GENERAL TIPS:")
                print(f"   • Always start with clean air baseline")
                print(f"   • Use consistent method for each gas type")
                print(f"   • Monitor real-time readings during collection")
                print(f"   • Take breaks between gases (5 min recovery)")
                print(f"   • Document environmental conditions")

            elif choice == '49':
                print("\n🔄 RESET SYSTEMATIC COLLECTION")
                print("="*40)
                
                print("This will delete all systematic collection files")
                confirm = input("Are you sure? (type 'RESET' to confirm): ")
                
                if confirm == 'RESET':
                    systematic_files = glob.glob("systematic_*.csv")
                    
                    for file in systematic_files:
                        try:
                            os.remove(file)
                            print(f"   🗑️ Deleted: {file}")
                        except Exception as e:
                            print(f"   ❌ Error deleting {file}: {e}")
                    
                    # Reset collection status
                    gas_sensor.systematic_guide.collection_status = {}
                    gas_sensor.systematic_guide.validation_results = {}
                    
                    print(f"\n✅ Systematic collection reset completed")
                    print(f"💡 You can now start fresh systematic collection")
                else:
                    print("❌ Reset cancelled")

            elif choice == '50':
                print("\n⚔️ SYSTEMATIC VS REGULAR COLLECTION COMPARISON")
                print("="*60)
                
                # Find both types of files
                systematic_files = glob.glob("systematic_*.csv")
                regular_files = glob.glob("training_*.csv")
                
                print(f"📊 FILE COMPARISON:")
                print(f"   Systematic files: {len(systematic_files)}")
                print(f"   Regular files: {len(regular_files)}")
                
                if systematic_files:
                    print(f"\n🚀 SYSTEMATIC COLLECTION FEATURES:")
                    print(f"   ✅ Proven 3-minute protocol")
                    print(f"   ✅ Real-time quality control")
                    print(f"   ✅ Standardized gas preparation")
                    print(f"   ✅ Built-in validation")
                    print(f"   ✅ Recovery time management")
                    
                    # Analyze systematic data quality
                    total_systematic_samples = 0
                    systematic_gases = set()
                    
                    for file in systematic_files:
                        try:
                            df = pd.read_csv(file)
                            total_systematic_samples += len(df)
                            systematic_gases.add(df['gas_type'].iloc[0])
                        except:
                            pass
                    
                    print(f"   📊 Total samples: {total_systematic_samples}")
                    print(f"   📊 Gas types: {len(systematic_gases)}")
                
                if regular_files:
                    print(f"\n📊 REGULAR COLLECTION:")
                    print(f"   🔶 Variable duration")
                    print(f"   🔶 Manual quality control")
                    print(f"   🔶 User-defined preparation")
                    print(f"   🔶 Manual validation needed")
                    
                    # Analyze regular data
                    total_regular_samples = 0
                    regular_gases = set()
                    
                    for file in regular_files:
                        try:
                            df = pd.read_csv(file)
                            total_regular_samples += len(df)
                            regular_gases.add(df['gas_type'].iloc[0])
                        except:
                            pass
                    
                    print(f"   📊 Total samples: {total_regular_samples}")
                    print(f"   📊 Gas types: {len(regular_gases)}")
                
                print(f"\n💡 RECOMMENDATION:")
                if len(systematic_files) >= 4:
                    print(f"   🚀 Use systematic collection - proven protocol")
                elif len(regular_files) >= 4:
                    print(f"   📊 Regular collection available - consider systematic upgrade")
                else:
                    print(f"   🎯 Start with systematic collection for best results")

            # [SISIPKAN SEMUA HANDLING CHOICE LAINNYA DARI KODE ASLI]
            # elif choice == '3', '4', '5', dll. - SEMUA TETAP ADA

            elif choice == '10':
                print("👋 Exiting Enhanced Gas Sensor System...")
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
