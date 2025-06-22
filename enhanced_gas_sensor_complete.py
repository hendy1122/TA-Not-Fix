#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - COMPLETE VERSION 4.4 - FIXED
FIXED: Missing methods, enhanced prediction, complete menu options
Menggabungkan semua fitur dari mainlengkap.py + enhanced features
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

class EnhancedDataAnalyzer:
    """Enhanced Data Analyzer untuk analisis kualitas training data"""
    
    def __init__(self, logger):
        self.logger = logger
        self.training_files = []
        self.combined_data = None
        self.gas_stats = {}
        self.overlap_analysis = {}
        
    def find_and_load_training_data(self):
        """Cari dan load semua training data"""
        print("\n🔍 ANALYZING EXISTING TRAINING DATA")
        print("="*50)
        
        # Cari training files
        patterns = ["training_*.csv", "data/training_*.csv", "*training*.csv"]
        self.training_files = []
        
        for pattern in patterns:
            files = glob.glob(pattern)
            self.training_files.extend(files)
        
        # Remove duplicates
        self.training_files = list(set(self.training_files))
        
        if not self.training_files:
            print("❌ No training data found!")
            return False
            
        print(f"📂 Found {len(self.training_files)} training files:")
        
        # Load dan combine data
        all_data = []
        gas_counts = {}
        
        for file in self.training_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                
                if 'gas_type' in df.columns:
                    gas_type = df['gas_type'].iloc[0]
                    gas_counts[gas_type] = len(df)
                    print(f"   ✅ {os.path.basename(file)}: {gas_type} ({len(df)} samples)")
                    
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if not all_data:
            return False
            
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total samples: {len(self.combined_data)}")
        print(f"   Gas types: {list(gas_counts.keys())}")
        for gas, count in gas_counts.items():
            percentage = (count / len(self.combined_data)) * 100
            print(f"     {gas}: {count} samples ({percentage:.1f}%)")
        
        return True
    
    def analyze_data_quality(self):
        """Analisis kualitas data untuk prediksi"""
        if self.combined_data is None:
            print("❌ No data loaded!")
            return False
            
        print(f"\n🔍 DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Analisis statistik per gas
        self.gas_stats = {}
        
        for gas_type in self.combined_data['gas_type'].unique():
            gas_data = self.combined_data[self.combined_data['gas_type'] == gas_type]
            
            stats = {
                'samples': len(gas_data),
                'sensors': {}
            }
            
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm_col = f'{sensor}_ppm'
                voltage_col = f'{sensor}_voltage'
                
                if ppm_col in gas_data.columns:
                    ppm_values = gas_data[ppm_col].dropna()
                    voltage_values = gas_data[voltage_col].dropna()
                    
                    if len(ppm_values) > 0:
                        stats['sensors'][sensor] = {
                            'ppm_mean': ppm_values.mean(),
                            'ppm_std': ppm_values.std(),
                            'ppm_min': ppm_values.min(),
                            'ppm_max': ppm_values.max(),
                            'voltage_mean': voltage_values.mean(),
                            'voltage_std': voltage_values.std()
                        }
            
            self.gas_stats[gas_type] = stats
        
        # Print analisis
        print("📊 PPM STATISTICS PER GAS TYPE:")
        for gas_type, stats in self.gas_stats.items():
            print(f"\n{gas_type.upper()} ({stats['samples']} samples):")
            
            for sensor, sensor_stats in stats['sensors'].items():
                if sensor_stats:
                    ppm_mean = sensor_stats['ppm_mean']
                    ppm_std = sensor_stats['ppm_std']
                    ppm_range = f"{sensor_stats['ppm_min']:.1f}-{sensor_stats['ppm_max']:.1f}"
                    
                    print(f"  {sensor}: μ={ppm_mean:.1f} σ={ppm_std:.1f} range=[{ppm_range}]")
        
        # Analisis separability
        self.analyze_gas_separability()
        return True
    
    def analyze_gas_separability(self):
        """Analisis separability antar gas types"""
        print(f"\n⚙️ GAS SEPARABILITY ANALYSIS:")
        
        gas_types = list(self.gas_stats.keys())
        problematic_pairs = []
        
        for i, gas1 in enumerate(gas_types):
            for j, gas2 in enumerate(gas_types):
                if i < j:
                    overlap_score = self.calculate_overlap_score(gas1, gas2)
                    
                    if overlap_score > 0.5:  # High overlap
                        problematic_pairs.append((gas1, gas2, overlap_score))
                        print(f"   ⚠️ {gas1} vs {gas2}: HIGH overlap ({overlap_score:.2f})")
                    elif overlap_score > 0.3:  # Moderate overlap
                        print(f"   🔶 {gas1} vs {gas2}: Moderate overlap ({overlap_score:.2f})")
        
        if not problematic_pairs:
            print("   ✅ Good separability between all gas types")
            return True
        else:
            print(f"\n❌ FOUND {len(problematic_pairs)} PROBLEMATIC PAIRS")
            print("💡 Enhanced features needed to improve separability")
            return False
    
    def calculate_overlap_score(self, gas1, gas2):
        """Calculate overlap score between two gases"""
        if gas1 not in self.gas_stats or gas2 not in self.gas_stats:
            return 0
        
        overlap_scores = []
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            if (sensor in self.gas_stats[gas1]['sensors'] and 
                sensor in self.gas_stats[gas2]['sensors']):
                
                stats1 = self.gas_stats[gas1]['sensors'][sensor]
                stats2 = self.gas_stats[gas2]['sensors'][sensor]
                
                if stats1 and stats2:
                    # Calculate range overlap
                    min1, max1 = stats1['ppm_min'], stats1['ppm_max']
                    min2, max2 = stats2['ppm_min'], stats2['ppm_max']
                    
                    # Overlap calculation
                    intersection = max(0, min(max1, max2) - max(min1, min2))
                    union = max(max1, max2) - min(min1, min2)
                    
                    if union > 0:
                        overlap = intersection / union
                        overlap_scores.append(overlap)
        
        return np.mean(overlap_scores) if overlap_scores else 0
    
    def get_recommendation(self):
        """Get recommendation based on analysis"""
        if not self.gas_stats:
            return "No data to analyze"
        
        total_samples = len(self.combined_data)
        num_gases = len(self.gas_stats)
        
        # Check sample balance
        sample_counts = [stats['samples'] for stats in self.gas_stats.values()]
        min_samples = min(sample_counts)
        max_samples = max(sample_counts)
        balance_ratio = min_samples / max_samples if max_samples > 0 else 0
        
        recommendations = []
        
        if total_samples < 500:
            recommendations.append("❌ Too few total samples (need >500)")
        
        if num_gases < 4:
            recommendations.append("❌ Too few gas types (need ≥4 for good model)")
        
        if balance_ratio < 0.5:
            recommendations.append("⚠️ Imbalanced data - some gases have too few samples")
        
        if not recommendations:
            recommendations.append("✅ Data quality acceptable for enhanced training")
        
        return "; ".join(recommendations)

class EnhancedFeatureExtractor:
    """Enhanced Feature Extractor untuk improve prediction"""
    
    def __init__(self, logger):
        self.logger = logger
        self.feature_names = []
        
    def extract_enhanced_features(self, df):
        """Extract enhanced features dari raw sensor data"""
        print("   🎯 Creating enhanced features for better gas discrimination...")
        
        enhanced_df = df.copy()
        
        # Basic features (keep original)
        base_features = ['TGS2600_ppm', 'TGS2602_ppm', 'TGS2610_ppm',
                        'TGS2600_voltage', 'TGS2602_voltage', 'TGS2610_voltage']
        
        # Feature 1: Cross-sensor ratios (untuk discriminate gases)
        enhanced_df['ratio_2600_2602'] = enhanced_df['TGS2600_ppm'] / (enhanced_df['TGS2602_ppm'] + 0.1)
        enhanced_df['ratio_2602_2610'] = enhanced_df['TGS2602_ppm'] / (enhanced_df['TGS2610_ppm'] + 0.1)
        enhanced_df['ratio_2600_2610'] = enhanced_df['TGS2600_ppm'] / (enhanced_df['TGS2610_ppm'] + 0.1)
        
        # Feature 2: Voltage differences (lebih stable dari ratio)
        enhanced_df['volt_diff_2600_2602'] = enhanced_df['TGS2600_voltage'] - enhanced_df['TGS2602_voltage']
        enhanced_df['volt_diff_2602_2610'] = enhanced_df['TGS2602_voltage'] - enhanced_df['TGS2610_voltage']
        enhanced_df['volt_diff_2600_2610'] = enhanced_df['TGS2600_voltage'] - enhanced_df['TGS2610_voltage']
        
        # Feature 3: Total response metrics
        enhanced_df['total_ppm'] = enhanced_df['TGS2600_ppm'] + enhanced_df['TGS2602_ppm'] + enhanced_df['TGS2610_ppm']
        enhanced_df['avg_ppm'] = enhanced_df['total_ppm'] / 3
        enhanced_df['max_ppm'] = enhanced_df[['TGS2600_ppm', 'TGS2602_ppm', 'TGS2610_ppm']].max(axis=1)
        
        # Feature 4: Dominant sensor indicator
        dominant_sensor = enhanced_df[['TGS2600_ppm', 'TGS2602_ppm', 'TGS2610_ppm']].idxmax(axis=1)
        enhanced_df['dominant_sensor'] = dominant_sensor.map({
            'TGS2600_ppm': 0, 'TGS2602_ppm': 1, 'TGS2610_ppm': 2
        }).fillna(-1)
        
        # Feature 5: Normalized responses (per sensor basis)
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            normalized_col = f'{sensor}_norm'
            
            max_val = enhanced_df[ppm_col].max()
            if max_val > 0:
                enhanced_df[normalized_col] = enhanced_df[ppm_col] / max_val
            else:
                enhanced_df[normalized_col] = 0
        
        # Feature 6: Gas-specific signature patterns
        # Alcohol: TGS2600 dominan, moderate overall
        enhanced_df['alcohol_pattern'] = (
            (enhanced_df['TGS2600_ppm'] > enhanced_df['TGS2602_ppm']) & 
            (enhanced_df['TGS2600_ppm'] > enhanced_df['TGS2610_ppm']) &
            (enhanced_df['total_ppm'] > 20) & (enhanced_df['total_ppm'] < 200)
        ).astype(int)
        
        # Toluene: TGS2602 very dominant
        enhanced_df['toluene_pattern'] = (
            (enhanced_df['TGS2602_ppm'] > enhanced_df['TGS2600_ppm'] * 1.5) & 
            (enhanced_df['TGS2602_ppm'] > enhanced_df['TGS2610_ppm'] * 1.5) &
            (enhanced_df['TGS2602_ppm'] > 30)
        ).astype(int)
        
        # Hydrocarbon (pertalite, dexlite): TGS2610 involved, balanced
        enhanced_df['hydrocarbon_pattern'] = (
            (enhanced_df['TGS2610_ppm'] > 10) &
            (enhanced_df['TGS2602_ppm'] > 10) &
            (enhanced_df['ratio_2602_2610'] < 3)
        ).astype(int)
        
        # Ammonia: TGS2602 dominant but different from toluene
        enhanced_df['ammonia_pattern'] = (
            (enhanced_df['TGS2602_ppm'] > enhanced_df['TGS2600_ppm']) & 
            (enhanced_df['TGS2602_ppm'] > enhanced_df['TGS2610_ppm']) &
            (enhanced_df['total_ppm'] > 15) & (enhanced_df['total_ppm'] < 150)
        ).astype(int)
        
        # Clean air: all low
        enhanced_df['clean_pattern'] = (
            (enhanced_df['total_ppm'] < 15) &
            (enhanced_df['max_ppm'] < 10)
        ).astype(int)
        
        # Feature 7: Response intensity levels
        enhanced_df['response_intensity'] = pd.cut(enhanced_df['total_ppm'], 
                                                  bins=[-1, 5, 20, 50, 150, 1000], 
                                                  labels=[0, 1, 2, 3, 4]).astype(int)
        
        # Define comprehensive feature list
        self.feature_names = base_features + [
            'ratio_2600_2602', 'ratio_2602_2610', 'ratio_2600_2610',
            'volt_diff_2600_2602', 'volt_diff_2602_2610', 'volt_diff_2600_2610',
            'total_ppm', 'avg_ppm', 'max_ppm', 'dominant_sensor',
            'TGS2600_norm', 'TGS2602_norm', 'TGS2610_norm',
            'alcohol_pattern', 'toluene_pattern', 'hydrocarbon_pattern', 
            'ammonia_pattern', 'clean_pattern', 'response_intensity'
        ]
        
        print(f"   ✅ Created {len(self.feature_names)} enhanced features")
        
        return enhanced_df

class EnhancedModelTrainer:
    """Enhanced Model Trainer dengan data balancing dan advanced algorithms"""
    
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.scaler = None
        self.feature_names = []
        
    def balance_and_augment_data(self, df):
        """Balance data dengan smart augmentation"""
        print("   ⚖️ Balancing data dengan smart augmentation...")
        
        gas_counts = df['gas_type'].value_counts()
        print(f"   Original distribution: {dict(gas_counts)}")
        
        target_samples = max(int(gas_counts.median() * 1.2), 100)  # Target reasonable
        augmented_data = []
        
        for gas_type in gas_counts.index:
            gas_data = df[df['gas_type'] == gas_type].copy()
            current_count = len(gas_data)
            
            # Add original data
            augmented_data.append(gas_data)
            
            # Augment if needed
            if current_count < target_samples:
                need_augment = target_samples - current_count
                print(f"     Augmenting {gas_type}: +{need_augment} synthetic samples")
                
                for _ in range(need_augment):
                    # Select random base sample
                    base_sample = gas_data.sample(1).copy()
                    
                    # Add controlled noise to sensor readings
                    noise_factor = np.random.uniform(0.9, 1.1)  # ±10% variation
                    
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        ppm_col = f'{sensor}_ppm'
                        voltage_col = f'{sensor}_voltage'
                        
                        if ppm_col in base_sample.columns:
                            # Apply noise
                            base_sample[ppm_col] *= noise_factor
                            
                            # Small voltage noise
                            voltage_noise = np.random.uniform(-0.005, 0.005)
                            base_sample[voltage_col] += voltage_noise
                    
                    augmented_data.append(base_sample)
        
        balanced_df = pd.concat(augmented_data, ignore_index=True)
        
        new_counts = balanced_df['gas_type'].value_counts()
        print(f"   ✅ Balanced distribution: {dict(new_counts)}")
        
        return balanced_df
    
    def train_enhanced_model(self, df, feature_names):
        """Train enhanced model dengan optimal parameters"""
        print("   🤖 Training enhanced RandomForest model...")
        
        # Prepare features dan target
        available_features = [col for col in feature_names if col in df.columns]
        print(f"   📊 Using {len(available_features)} features")
        
        X = df[available_features].fillna(0)
        y = df['gas_type']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"   📈 Training: {len(X_train)} samples")
        print(f"   📊 Testing: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train enhanced model
        self.model = RandomForestClassifier(
            n_estimators=400,        # More trees for stability
            max_depth=18,           # Deeper for complex patterns
            min_samples_split=3,    # Prevent overfitting
            min_samples_leaf=2,     # Prevent overfitting
            max_features='sqrt',    # Feature randomness
            bootstrap=True,
            random_state=42,
            class_weight='balanced', # Handle class imbalance
            n_jobs=-1               # Use all cores
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = available_features
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   🎯 Model accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed evaluation
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        all_good = True
        for gas_type, metrics in report.items():
            if gas_type not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = int(metrics['support'])
                
                status = "✅" if f1 > 0.7 else "⚠️" if f1 > 0.5 else "❌"
                print(f"   {status} {gas_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({support} samples)")
                
                if f1 < 0.6:
                    all_good = False
        
        # Feature importance analysis
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        importances = list(zip(available_features, self.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importances[:10]):
            print(f"   {i+1:2d}. {feature}: {importance:.3f}")
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 CONFUSION MATRIX:")
        gas_labels = sorted(y.unique())
        print(f"   Labels: {gas_labels}")
        
        return accuracy > 0.75 and all_good
    
    def save_enhanced_model(self):
        """Save enhanced model dengan metadata lengkap"""
        print(f"\n💾 SAVING ENHANCED MODEL...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model dan scaler
        joblib.dump(self.model, 'models/enhanced_gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/enhanced_scaler.pkl')
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': 'enhanced_v4.4_opsi1',
            'model_type': 'RandomForest_Enhanced_Balanced',
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'gas_types': list(self.model.classes_),
            'model_params': self.model.get_params(),
            'enhancements': [
                'cross_sensor_ratios',
                'voltage_differences', 
                'pattern_signatures',
                'data_augmentation',
                'class_balancing',
                'optimal_hyperparameters'
            ],
            'collecting_duration': '180_seconds',  # 3 menit
            'success_criteria': 'accuracy>75%_all_f1>60%'
        }
        
        with open('models/enhanced_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Enhanced model saved:")
        print(f"   🤖 models/enhanced_gas_classifier.pkl")
        print(f"   ⚖️ models/enhanced_scaler.pkl")
        print(f"   📋 models/enhanced_model_metadata.json")

class EnhancedPredictor:
    """Enhanced Predictor untuk real-time gas prediction"""
    
    def __init__(self, sensor_array):
        self.sensor_array = sensor_array
        self.enhanced_model = None
        self.enhanced_scaler = None
        self.feature_names = []
        self.is_enhanced_loaded = False
        
    def load_enhanced_model(self):
        """Load enhanced model"""
        try:
            self.enhanced_model = joblib.load('models/enhanced_gas_classifier.pkl')
            self.enhanced_scaler = joblib.load('models/enhanced_scaler.pkl')
            
            # Load metadata
            with open('models/enhanced_model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
            
            self.is_enhanced_loaded = True
            self.sensor_array.logger.info("🚀 Enhanced model loaded successfully")
            return True
            
        except FileNotFoundError:
            self.sensor_array.logger.info("Enhanced model not found")
            return False
        except Exception as e:
            self.sensor_array.logger.error(f"Error loading enhanced model: {e}")
            return False
    
    def extract_features_from_readings(self, readings):
        """Extract enhanced features dari real-time readings"""
        try:
            # Basic sensor readings
            ppm_2600 = readings['TGS2600']['ppm']
            ppm_2602 = readings['TGS2602']['ppm'] 
            ppm_2610 = readings['TGS2610']['ppm']
            
            volt_2600 = readings['TGS2600']['voltage']
            volt_2602 = readings['TGS2602']['voltage']
            volt_2610 = readings['TGS2610']['voltage']
            
            # Calculate enhanced features (same as training)
            features = {}
            
            # Basic features
            features['TGS2600_ppm'] = ppm_2600
            features['TGS2602_ppm'] = ppm_2602
            features['TGS2610_ppm'] = ppm_2610
            features['TGS2600_voltage'] = volt_2600
            features['TGS2602_voltage'] = volt_2602
            features['TGS2610_voltage'] = volt_2610
            
            # Ratios
            features['ratio_2600_2602'] = ppm_2600 / (ppm_2602 + 0.1)
            features['ratio_2602_2610'] = ppm_2602 / (ppm_2610 + 0.1)
            features['ratio_2600_2610'] = ppm_2600 / (ppm_2610 + 0.1)
            
            # Voltage differences
            features['volt_diff_2600_2602'] = volt_2600 - volt_2602
            features['volt_diff_2602_2610'] = volt_2602 - volt_2610
            features['volt_diff_2600_2610'] = volt_2600 - volt_2610
            
            # Total metrics
            total_ppm = ppm_2600 + ppm_2602 + ppm_2610
            features['total_ppm'] = total_ppm
            features['avg_ppm'] = total_ppm / 3
            features['max_ppm'] = max(ppm_2600, ppm_2602, ppm_2610)
            
            # Dominant sensor
            ppms = [ppm_2600, ppm_2602, ppm_2610]
            features['dominant_sensor'] = ppms.index(max(ppms))
            
            # Normalized (estimate max values dari training)
            max_estimates = [200, 200, 200]
            features['TGS2600_norm'] = ppm_2600 / max_estimates[0]
            features['TGS2602_norm'] = ppm_2602 / max_estimates[1]
            features['TGS2610_norm'] = ppm_2610 / max_estimates[2]
            
            # Pattern signatures
            features['alcohol_pattern'] = int(
                (ppm_2600 > ppm_2602) and (ppm_2600 > ppm_2610) and 
                (20 < total_ppm < 200)
            )
            
            features['toluene_pattern'] = int(
                (ppm_2602 > ppm_2600 * 1.5) and (ppm_2602 > ppm_2610 * 1.5) and
                (ppm_2602 > 30)
            )
            
            features['hydrocarbon_pattern'] = int(
                (ppm_2610 > 10) and (ppm_2602 > 10) and 
                (features['ratio_2602_2610'] < 3)
            )
            
            features['ammonia_pattern'] = int(
                (ppm_2602 > ppm_2600) and (ppm_2602 > ppm_2610) and
                (15 < total_ppm < 150)
            )
            
            features['clean_pattern'] = int(
                (total_ppm < 15) and (features['max_ppm'] < 10)
            )
            
            # Response intensity
            if total_ppm < 5:
                features['response_intensity'] = 0
            elif total_ppm < 20:
                features['response_intensity'] = 1
            elif total_ppm < 50:
                features['response_intensity'] = 2
            elif total_ppm < 150:
                features['response_intensity'] = 3
            else:
                features['response_intensity'] = 4
            
            return features
            
        except Exception as e:
            self.sensor_array.logger.error(f"Error extracting features: {e}")
            return None
    
    def predict_gas_enhanced(self, readings):
        """Enhanced gas prediction"""
        if not self.is_enhanced_loaded:
            self.sensor_array.logger.warning("Enhanced model not loaded, using original")
            return self.sensor_array.predict_gas_original(readings)
        
        try:
            # Extract features
            features_dict = self.extract_features_from_readings(readings)
            if features_dict is None:
                return "Error", 0.0
            
            # Convert to array in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features_dict.get(feature_name, 0))
            
            # Handle any remaining NaN or inf
            feature_vector = np.array(feature_vector)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1000, neginf=-1000)
            
            # Scale and predict
            features_scaled = self.enhanced_scaler.transform(feature_vector.reshape(1, -1))
            
            prediction = self.enhanced_model.predict(features_scaled)[0]
            probabilities = self.enhanced_model.predict_proba(features_scaled)[0]
            base_confidence = probabilities.max()
            
            # Enhanced confidence boosting
            enhanced_confidence = self.boost_confidence_with_patterns(
                prediction, base_confidence, readings, features_dict
            )
            
            return prediction, enhanced_confidence
            
        except Exception as e:
            self.sensor_array.logger.error(f"Error in enhanced prediction: {e}")
            return "Error", 0.0
    
    def boost_confidence_with_patterns(self, prediction, base_confidence, readings, features):
        """Boost confidence berdasarkan pattern analysis"""
        try:
            confidence_boost = 0.0
            
            # Pattern-based confidence boosting
            if prediction == 'normal' and features.get('clean_pattern', 0) == 1:
                confidence_boost += 0.15
            elif prediction == 'alcohol' and features.get('alcohol_pattern', 0) == 1:
                confidence_boost += 0.12
            elif prediction == 'toluene' and features.get('toluene_pattern', 0) == 1:
                confidence_boost += 0.15
            elif prediction in ['pertalite', 'dexlite'] and features.get('hydrocarbon_pattern', 0) == 1:
                confidence_boost += 0.10
            elif prediction == 'ammonia' and features.get('ammonia_pattern', 0) == 1:
                confidence_boost += 0.12
            
            # Sensor response consistency boost
            ppm_values = [readings[sensor]['ppm'] for sensor in ['TGS2600', 'TGS2602', 'TGS2610']]
            responding_sensors = sum(1 for ppm in ppm_values if ppm > 5)
            
            if prediction != 'normal' and responding_sensors >= 2:
                confidence_boost += 0.05
            elif prediction == 'normal' and responding_sensors == 0:
                confidence_boost += 0.10
            
            # Dominant sensor consistency
            dominant_sensor = features.get('dominant_sensor', -1)
            if prediction == 'alcohol' and dominant_sensor == 0:  # TGS2600
                confidence_boost += 0.08
            elif prediction in ['toluene', 'ammonia'] and dominant_sensor == 1:  # TGS2602
                confidence_boost += 0.08
            elif prediction in ['pertalite', 'dexlite'] and dominant_sensor == 2:  # TGS2610
                confidence_boost += 0.08
            
            enhanced_confidence = min(1.0, base_confidence + confidence_boost)
            return enhanced_confidence
            
        except Exception:
            return base_confidence

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
        
        # Feature transformation methods
        self.transformation_methods = {
            'linear_scale': self.linear_scale_transform,
            'log_scale': self.log_scale_transform,
            'robust_scale': self.robust_scale_transform,
            'percentile_scale': self.percentile_scale_transform,
            'adaptive_normalize': self.adaptive_normalize_transform
        }
        
        # Current best transformation method
        self.best_transform_method = 'adaptive_normalize'
        
        # Multi-scale features
        self.enable_multi_scale = True
        
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
    
    def update_current_stats(self, readings_history):
        """Update current statistics from recent readings"""
        if len(readings_history) < 10:
            return  # Need minimum samples
        
        # Calculate statistics for recent readings
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_values = [r[sensor]['ppm'] for r in readings_history if r[sensor]['ppm'] > 0]
            voltage_values = [r[sensor]['raw_voltage'] for r in readings_history]
            
            if ppm_values:
                self.current_stats[f'{sensor}_ppm'] = {
                    'min': min(ppm_values),
                    'max': max(ppm_values),
                    'mean': np.mean(ppm_values),
                    'std': np.std(ppm_values)
                }
            
            if voltage_values:
                self.current_stats[f'{sensor}_voltage'] = {
                    'min': min(voltage_values),
                    'max': max(voltage_values),
                    'mean': np.mean(voltage_values),
                    'std': np.std(voltage_values)
                }
        
        # Update scaling factors
        self.calculate_adaptive_scaling_factors()
    
    def calculate_adaptive_scaling_factors(self):
        """Calculate adaptive scaling factors"""
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_key = f'{sensor}_ppm'
            
            if ppm_key in self.current_stats and ppm_key in self.training_stats:
                current_max = self.current_stats[ppm_key]['max']
                training_max = self.training_stats[ppm_key]['max']
                
                if current_max > 0 and training_max > 0:
                    # Calculate scaling factor to bring current range to training range
                    scale_factor = training_max / current_max
                    
                    # Apply smoothing to avoid extreme scaling
                    scale_factor = max(0.1, min(10.0, scale_factor))
                    
                    self.ppm_scaling_factors[sensor] = scale_factor
                    self.logger.info(f"{sensor} PPM scaling factor: {scale_factor:.3f}")
    
    def linear_scale_transform(self, value, feature_name):
        """Linear scaling transformation"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            return value * self.ppm_scaling_factors.get(sensor, 1.0)
        return value
    
    def log_scale_transform(self, value, feature_name):
        """Log scaling transformation"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            return np.log1p(scaled_value)  # log(1 + x) to handle 0 values
        return value
    
    def robust_scale_transform(self, value, feature_name):
        """Robust scaling using percentiles"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Use training stats for normalization
            if feature_name in self.training_stats:
                training_median = self.training_stats[feature_name].get('mean', 20)
                training_iqr = self.training_stats[feature_name].get('std', 10) * 1.349  # Approximate IQR
                
                return (scaled_value - training_median) / (training_iqr + 1e-8)
        
        return value
    
    def percentile_scale_transform(self, value, feature_name):
        """Percentile-based scaling"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Map to percentile of training distribution
            if feature_name in self.training_stats:
                training_max = self.training_stats[feature_name]['max']
                percentile = min(100, (scaled_value / training_max) * 100)
                return percentile / 100.0  # Normalize to 0-1
        
        return value
    
    def adaptive_normalize_transform(self, value, feature_name):
        """Advanced adaptive normalization"""
        if feature_name.endswith('_ppm'):
            sensor = feature_name.split('_')[0]
            
            # Multi-step transformation
            # Step 1: Apply adaptive scaling
            scaled_value = value * self.ppm_scaling_factors.get(sensor, 1.0)
            
            # Step 2: Apply training range normalization
            if feature_name in self.training_stats:
                training_stats = self.training_stats[feature_name]
                training_range = training_stats['max'] - training_stats['min']
                
                if training_range > 0:
                    normalized = (scaled_value - training_stats['min']) / training_range
                    # Clip to reasonable range but allow some extrapolation
                    normalized = np.clip(normalized, -0.5, 1.5)
                    return normalized
            
            # Fallback normalization
            return min(1.0, scaled_value / 100.0)
        
        # Voltage normalization
        elif feature_name.endswith('_voltage'):
            if feature_name in self.training_stats:
                training_stats = self.training_stats[feature_name]
                training_range = training_stats['max'] - training_stats['min']
                
                if training_range > 0:
                    normalized = (value - training_stats['min']) / training_range
                    return np.clip(normalized, 0.0, 1.0)
        
        return value
    
    def extract_multi_scale_features(self, readings):
        """Extract multi-scale features for better detection"""
        multi_scale_features = {}
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            sensor_data = readings[sensor]
            
            # Original features
            multi_scale_features[f'{sensor}_voltage'] = sensor_data['raw_voltage']
            multi_scale_features[f'{sensor}_ppm'] = sensor_data['ppm']
            
            if self.enable_multi_scale:
                # Ratio features
                base_voltage = 1.6
                voltage_ratio = sensor_data['raw_voltage'] / base_voltage
                multi_scale_features[f'{sensor}_voltage_ratio'] = voltage_ratio
                
                # Log features (better for wide range data)
                log_ppm = np.log1p(sensor_data['ppm'])
                multi_scale_features[f'{sensor}_log_ppm'] = log_ppm
                
                # Normalized features
                if sensor_data['ppm'] > 0:
                    # PPM per voltage drop
                    voltage_drop = base_voltage - sensor_data['raw_voltage']
                    if voltage_drop > 0.001:
                        ppm_per_voltage = sensor_data['ppm'] / voltage_drop
                        multi_scale_features[f'{sensor}_ppm_efficiency'] = min(1000, ppm_per_voltage)
                    else:
                        multi_scale_features[f'{sensor}_ppm_efficiency'] = 0
                else:
                    multi_scale_features[f'{sensor}_ppm_efficiency'] = 0
                
                # Sensitivity-aware features
                sensitivity_multiplier = sensor_data.get('sensitivity_multiplier', 1.0)
                if sensitivity_multiplier > 1:
                    raw_ppm = sensor_data['ppm'] / sensitivity_multiplier
                    multi_scale_features[f'{sensor}_raw_ppm'] = raw_ppm
                else:
                    multi_scale_features[f'{sensor}_raw_ppm'] = sensor_data['ppm']
        
        return multi_scale_features
    
    def transform_features(self, feature_vector, feature_names):
        """Transform feature vector using adaptive methods"""
        transformed_features = []
        
        for i, (value, feature_name) in enumerate(zip(feature_vector, feature_names)):
            # Apply selected transformation method
            transform_func = self.transformation_methods[self.best_transform_method]
            transformed_value = transform_func(value, feature_name)
            
            # Handle NaN and infinite values
            if np.isnan(transformed_value) or np.isinf(transformed_value):
                transformed_value = 0.0
            
            transformed_features.append(transformed_value)
        
        return transformed_features
    
    def evaluate_transformation_quality(self, readings_history):
        """Evaluate which transformation method works best"""
        if len(readings_history) < 20:
            return
        
        # Simple heuristic: test different methods and see which gives most stable features
        method_scores = {}
        
        for method_name, transform_func in self.transformation_methods.items():
            feature_stability = []
            
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                ppm_values = [r[sensor]['ppm'] for r in readings_history[-20:]]
                transformed_values = [transform_func(ppm, f'{sensor}_ppm') for ppm in ppm_values]
                
                # Calculate coefficient of variation (stability metric)
                if np.mean(transformed_values) > 0:
                    cv = np.std(transformed_values) / np.mean(transformed_values)
                    feature_stability.append(cv)
            
            if feature_stability:
                method_scores[method_name] = np.mean(feature_stability)
        
        # Select method with lowest CV (most stable)
        if method_scores:
            best_method = min(method_scores.keys(), key=lambda x: method_scores[x])
            if best_method != self.best_transform_method:
                self.best_transform_method = best_method
                self.logger.info(f"Switched to transformation method: {best_method}")
    
    def get_adaptive_confidence_threshold(self, predicted_gas, base_confidence):
        """Get adaptive confidence threshold based on gas type and current conditions"""
        # Base thresholds per gas type (lower for difficult gases)
        base_thresholds = {
            'normal': 0.3,
            'alcohol': 0.4,
            'pertalite': 0.4,
            'toluene': 0.2,    # Lower threshold for harder to detect gases
            'ammonia': 0.2,
            'dexlite': 0.3,
            'butane': 0.25,
            'propane': 0.25
        }
        
        base_threshold = base_thresholds.get(predicted_gas, 0.35)
        return base_threshold
    
    def save_adaptive_config(self):
        """Save adaptive configuration"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'version': 'adaptive_v4.3',
            'ppm_scaling_factors': self.ppm_scaling_factors,
            'current_stats': self.current_stats,
            'best_transform_method': self.best_transform_method,
            'enable_multi_scale': self.enable_multi_scale
        }
        
        try:
            with open('adaptive_feature_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info("Adaptive feature configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving adaptive config: {e}")
    
    def load_adaptive_config(self):
        """Load adaptive configuration"""
        try:
            with open('adaptive_feature_config.json', 'r') as f:
                config = json.load(f)
            
            self.ppm_scaling_factors = config.get('ppm_scaling_factors', self.ppm_scaling_factors)
            self.current_stats = config.get('current_stats', {})
            self.best_transform_method = config.get('best_transform_method', 'adaptive_normalize')
            self.enable_multi_scale = config.get('enable_multi_scale', True)
            
            self.logger.info("Adaptive feature configuration loaded")
            
        except FileNotFoundError:
            self.logger.info("No adaptive config found, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading adaptive config: {e}")

class EnhancedPredictionEngine:
    """Enhanced Prediction Engine with confidence boosting"""
    
    def __init__(self, logger):
        self.logger = logger
        self.recent_predictions = []
        self.prediction_history = {}
        
        # Confidence boosting parameters
        self.consistency_boost = True
        self.temporal_smoothing = True
        self.multi_evidence_fusion = True
        
        # Gas-specific detection patterns
        self.gas_signatures = {
            'alcohol': {
                'primary_sensors': ['TGS2600', 'TGS2602'],
                'voltage_pattern': 'decrease',
                'ppm_ratio_range': (0.5, 2.0),
                'response_speed': 'fast'
            },
            'pertalite': {
                'primary_sensors': ['TGS2602', 'TGS2610'],
                'voltage_pattern': 'decrease',
                'ppm_ratio_range': (0.3, 1.5),
                'response_speed': 'medium'
            },
            'toluene': {
                'primary_sensors': ['TGS2602'],
                'voltage_pattern': 'strong_decrease',
                'ppm_ratio_range': (1.0, 3.0),
                'response_speed': 'fast'
            },
            'ammonia': {
                'primary_sensors': ['TGS2602'],
                'voltage_pattern': 'moderate_decrease',
                'ppm_ratio_range': (0.8, 2.5),
                'response_speed': 'slow'
            }
        }
    
    def analyze_sensor_patterns(self, readings):
        """Analyze sensor response patterns for gas detection"""
        pattern_scores = {}
        
        base_voltage = 1.6
        
        for gas_type, signature in self.gas_signatures.items():
            score = 0.0
            evidence_count = 0
            
            # Check primary sensors
            for sensor in signature['primary_sensors']:
                if sensor in readings:
                    sensor_data = readings[sensor]
                    
                    # Voltage pattern check
                    voltage_drop = base_voltage - sensor_data['raw_voltage']
                    if voltage_drop > 0.02:  # Minimum response
                        if signature['voltage_pattern'] == 'strong_decrease' and voltage_drop > 0.2:
                            score += 2.0
                        elif signature['voltage_pattern'] == 'decrease' and voltage_drop > 0.05:
                            score += 1.5
                        elif signature['voltage_pattern'] == 'moderate_decrease' and voltage_drop > 0.1:
                            score += 1.0
                        evidence_count += 1
                    
                    # PPM response check
                    if sensor_data['ppm'] > 10:  # Minimum PPM for detection
                        score += 1.0
                        evidence_count += 1
            
            # Normalize score by evidence count
            if evidence_count > 0:
                pattern_scores[gas_type] = score / evidence_count
            else:
                pattern_scores[gas_type] = 0.0
        
        return pattern_scores
    
    def boost_confidence_with_patterns(self, prediction, base_confidence, readings):
        """Boost confidence using sensor pattern analysis"""
        pattern_scores = self.analyze_sensor_patterns(readings)
        
        # Get pattern score for predicted gas
        pattern_score = pattern_scores.get(prediction, 0.0)
        
        # Calculate confidence boost
        if pattern_score > 1.0:
            confidence_boost = min(0.3, pattern_score * 0.15)  # Max boost of 0.3
            boosted_confidence = min(1.0, base_confidence + confidence_boost)
            
            self.logger.debug(f"Confidence boosted for {prediction}: {base_confidence:.3f} -> {boosted_confidence:.3f}")
            return boosted_confidence
        
        return base_confidence
    
    def apply_temporal_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to reduce prediction jitter"""
        if not self.temporal_smoothing:
            return prediction, confidence
        
        # Add to recent predictions
        self.recent_predictions.append((prediction, confidence, time.time()))
        
        # Keep only recent predictions (last 30 seconds)
        current_time = time.time()
        self.recent_predictions = [
            (p, c, t) for p, c, t in self.recent_predictions 
            if current_time - t < 30
        ]
        
        if len(self.recent_predictions) < 3:
            return prediction, confidence
        
        # Count recent predictions
        recent_counts = {}
        total_confidence = 0
        
        for p, c, t in self.recent_predictions[-5:]:  # Last 5 predictions
            recent_counts[p] = recent_counts.get(p, 0) + c
            total_confidence += c
        
        # Find most consistent prediction
        if total_confidence > 0:
            most_consistent = max(recent_counts.keys(), key=lambda x: recent_counts[x])
            consistency_score = recent_counts[most_consistent] / total_confidence
            
            # If current prediction matches most consistent and has good consistency
            if prediction == most_consistent and consistency_score > 0.6:
                smoothed_confidence = min(1.0, confidence * (1 + consistency_score * 0.2))
                return prediction, smoothed_confidence
        
        return prediction, confidence
    
    def fuse_multi_evidence(self, ml_prediction, ml_confidence, readings):
        """Fuse ML prediction with rule-based evidence"""
        if not self.multi_evidence_fusion:
            return ml_prediction, ml_confidence
        
        # Get pattern-based evidence
        pattern_scores = self.analyze_sensor_patterns(readings)
        
        # Simple rule-based predictions
        rule_predictions = {}
        
        # Rule 1: High PPM on specific sensors
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            if sensor in readings and readings[sensor]['ppm'] > 50:
                if sensor == 'TGS2600' and readings[sensor]['ppm'] > 100:
                    rule_predictions['alcohol'] = rule_predictions.get('alcohol', 0) + 0.3
                elif sensor == 'TGS2602' and readings[sensor]['ppm'] > 80:
                    rule_predictions['toluene'] = rule_predictions.get('toluene', 0) + 0.4
                    rule_predictions['pertalite'] = rule_predictions.get('pertalite', 0) + 0.3
                elif sensor == 'TGS2610' and readings[sensor]['ppm'] > 60:
                    rule_predictions['pertalite'] = rule_predictions.get('pertalite', 0) + 0.3
        
        # Rule 2: Voltage patterns
        base_voltage = 1.6
        total_voltage_drop = sum([
            max(0, base_voltage - readings[sensor]['raw_voltage']) 
            for sensor in ['TGS2600', 'TGS2602', 'TGS2610'] if sensor in readings
        ])
        
        if total_voltage_drop > 0.3:
            rule_predictions['alcohol'] = rule_predictions.get('alcohol', 0) + 0.2
        
        # Fusion: ML prediction with rule-based evidence
        if rule_predictions:
            best_rule_gas = max(rule_predictions.keys(), key=lambda x: rule_predictions[x])
            best_rule_score = rule_predictions[best_rule_gas]
            
            # If rule-based prediction agrees with ML prediction
            if best_rule_gas == ml_prediction and best_rule_score > 0.3:
                fused_confidence = min(1.0, ml_confidence + best_rule_score * 0.3)
                return ml_prediction, fused_confidence
            
            # If rule-based prediction is strong and ML confidence is low
            elif best_rule_score > 0.5 and ml_confidence < 0.4:
                return best_rule_gas, best_rule_score
        
        return ml_prediction, ml_confidence

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
        
        print("\n🎯 VOLTAGE ADJUSTMENTS:")
        for sensor, adj_data in self.voltage_adjustments.items():
            status = "ADJUSTED" if adj_data.get('adjusted', False) else "ORIGINAL"
            print(f"  {sensor}: {adj_data.get('original', 1.6):.3f}V → {adj_data.get('current', 1.6):.3f}V ({status})")
        
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
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if health_percentage < 75:
            print("  🔧 Consider recalibration")
        if len(self.drift_compensation_factors) > 0:
            print("  📊 Drift compensation active")
        if self.daily_check_done:
            print("  ✅ Daily check completed")
        else:
            print("  ⚠️  Daily check recommended")
    
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
    
    def smart_voltage_adjustment(self, sensor_array):
        """Smart voltage adjustment for problematic sensors"""
        print("\n🔧 SMART VOLTAGE ADJUSTMENT")
        print("="*50)
        
        for sensor_name, config in sensor_array.sensor_config.items():
            current_voltage = config['channel'].voltage
            original_baseline = self.original_baseline.get(sensor_name, 1.6)
            
            print(f"\n{sensor_name}:")
            print(f"  Current: {current_voltage:.3f}V")
            print(f"  Target: {original_baseline:.3f}V")
            
            voltage_diff = abs(current_voltage - original_baseline)
            
            if voltage_diff > 0.1:  # Significant difference
                print(f"  ⚠️  Significant voltage difference: {voltage_diff:.3f}V")
                
                adjust = input(f"  Adjust {sensor_name} baseline? (y/n): ").lower()
                if adjust == 'y':
                    self.voltage_adjustments[sensor_name] = {
                        'original': original_baseline,
                        'current': current_voltage,
                        'adjusted': True
                    }
                    
                    # Update baseline
                    self.current_baseline[sensor_name] = current_voltage
                    config['baseline_voltage'] = current_voltage
                    
                    print(f"  ✅ Baseline adjusted to {current_voltage:.3f}V")
            else:
                print(f"  ✅ Voltage within acceptable range")
        
        self.save_drift_data()
    
    def sensor_responsivity_check(self, sensor_array):
        """Check sensor responsivity"""
        print("\n📡 SENSOR RESPONSIVITY CHECK")
        print("="*50)
        
        print("This test checks if sensors respond to environmental changes")
        print("Gently blow air near sensors or move sensors slightly...")
        
        input("Press Enter to start responsivity test...")
        
        baseline_readings = {}
        
        # Get baseline
        print("Measuring baseline...")
        for sensor_name, config in sensor_array.sensor_config.items():
            readings = []
            for i in range(5):
                voltage = config['channel'].voltage
                readings.append(voltage)
                time.sleep(1)
            baseline_readings[sensor_name] = np.mean(readings)
        
        print("Now gently disturb the sensors (blow air, move slightly)...")
        print("Monitoring for 30 seconds...")
        
        max_deviations = {sensor: 0 for sensor in baseline_readings.keys()}
        
        start_time = time.time()
        while time.time() - start_time < 30:
            for sensor_name, config in sensor_array.sensor_config.items():
                current_voltage = config['channel'].voltage
                baseline = baseline_readings[sensor_name]
                deviation = abs(current_voltage - baseline)
                
                if deviation > max_deviations[sensor_name]:
                    max_deviations[sensor_name] = deviation
            
            time.sleep(1)
        
        # Analyze results
        print(f"\n📊 RESPONSIVITY RESULTS:")
        for sensor_name, max_dev in max_deviations.items():
            print(f"{sensor_name}: Max deviation {max_dev:.4f}V")
            
            if max_dev > 0.01:
                responsivity = "GOOD ✅"
            elif max_dev > 0.005:
                responsivity = "MODERATE ⚠️"
            else:
                responsivity = "LOW ❌"
            
            print(f"  Responsivity: {responsivity}")
    
    def quick_voltage_check(self, sensor_array):
        """Quick voltage check for all sensors"""
        print("\n⚡ QUICK VOLTAGE CHECK")
        print("="*40)
        
        for sensor_name, config in sensor_array.sensor_config.items():
            voltage = config['channel'].voltage
            print(f"{sensor_name}: {voltage:.3f}V")
            
            # Check voltage health
            if 1.0 <= voltage <= 3.0:
                status = "✅ NORMAL"
            elif voltage < 1.0:
                status = "⚠️  LOW"
            elif voltage > 3.0:
                status = "⚠️  HIGH"
            else:
                status = "❌ ERROR"
            
            print(f"  Status: {status}")
    
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
                    f'{sensor}_advanced_ppm',
                    f'{sensor}_mode',
                    f'{sensor}_drift_factor',
                    f'{sensor}_normalization_factor',
                    f'{sensor}_sensitivity_profile',
                    f'{sensor}_sensitivity_multiplier'
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
                        sensor_data['emergency_ppm'],
                        sensor_data['advanced_ppm'],
                        sensor_data['mode'],
                        sensor_data['drift_factor'],
                        sensor_data['normalization_factor'],
                        sensor_data['sensitivity_profile'],
                        sensor_data['sensitivity_multiplier']
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

        # Enhanced sensor configurations dengan 3 menit collecting
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
                'collecting_duration': 180,  # 3 menit - sesuai yang berhasil
                'use_extended_mode': False,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True,
                'sensitivity_ratios': {
                    'hydrogen': (0.3, 0.6),
                    'carbon_monoxide': (0.4, 0.7),
                    'alcohol': (0.2, 0.5)
                },
                'concentration_threshold': 50,
                'extended_sensitivity': 2.5
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
                'collecting_duration': 180,  # 3 menit
                'use_extended_mode': False,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True,
                'sensitivity_ratios': {
                    'alcohol': (0.08, 0.5),
                    'toluene': (0.1, 0.4),
                    'ammonia': (0.15, 0.6),
                    'h2s': (0.05, 0.3)
                },
                'concentration_threshold': 40,
                'extended_sensitivity': 3.0
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
                'collecting_duration': 180,  # 3 menit
                'use_extended_mode': False,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True,
                'sensitivity_ratios': {
                    'iso_butane': (0.45, 0.62),
                    'butane': (0.4, 0.6),
                    'propane': (0.35, 0.55),
                    'lp_gas': (0.4, 0.6)
                },
                'concentration_threshold': 30,
                'extended_sensitivity': 2.0
            }
        }

        # Initialize all managers
        self.drift_manager = SmartDriftManager(self.logger)
        self.emergency_ppm_calc = EmergencyPPMCalculator(self.logger)
        self.sensitivity_manager = AdvancedSensitivityManager(self.logger)
        self.monitoring_collector = MonitoringDataCollector(self.logger)
        
        # Enhanced features
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

        # Enhanced Machine Learning dengan adaptive features
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.feature_names = None
        self.training_metadata = None

        # Enhanced prediction system
        self.enhanced_predictor = None
        self.prediction_mode = 'original'  # 'original' atau 'enhanced'
        
        # Data analyzer untuk opsi 1
        self.data_analyzer = EnhancedDataAnalyzer(self.logger)
        
        # Enhanced model trainer
        self.enhanced_trainer = None

        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("calibration").mkdir(exist_ok=True)

        self.logger.info("Enhanced Gas Sensor Array System v4.4 - COMPLETE FIXED READY")

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
        """Enhanced sensor reading with adaptive features"""
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

        # Add to readings history for adaptive learning
        self.readings_history.append(readings)
        if len(self.readings_history) > self.max_history_size:
            self.readings_history.pop(0)
        
        # Update adaptive processor periodically
        if len(self.readings_history) % 20 == 0:
            self.adaptive_processor.update_current_stats(self.readings_history)
            self.adaptive_processor.evaluate_transformation_quality(self.readings_history)

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

    def find_training_files(self):
        """Enhanced training file detection"""
        # Possible file patterns and locations
        possible_patterns = [
            "training_*.csv",
            "data/training_*.csv", 
            "*training*.csv",
            "data/*training*.csv"
        ]
        
        training_files = []
        
        for pattern in possible_patterns:
            files = glob.glob(pattern)
            training_files.extend(files)
        
        # Remove duplicates
        training_files = list(set(training_files))
        
        # Filter and validate files
        valid_files = []
        for file in training_files:
            if os.path.exists(file) and file.endswith('.csv'):
                try:
                    # Quick validation - check if file has expected columns
                    df = pd.read_csv(file, nrows=1)
                    required_columns = ['gas_type', 'TGS2600_voltage', 'TGS2602_voltage', 'TGS2610_voltage']
                    if all(col in df.columns for col in required_columns):
                        valid_files.append(file)
                        self.logger.info(f"Found valid training file: {file}")
                except Exception as e:
                    self.logger.warning(f"Skipping invalid file {file}: {e}")
        
        return valid_files

    def collect_training_data(self, gas_type, duration=180):
        """Enhanced data collection dengan durasi 3 menit (sesuai yang berhasil)"""
        self.logger.info(f"Starting enhanced data collection for {gas_type} - Duration: {duration} seconds")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_{gas_type}_{timestamp}.csv"
        
        # Enhanced CSV headers
        headers = [
            'timestamp', 'gas_type', 'temperature', 'humidity'
        ]
        
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            headers.extend([
                f'{sensor}_voltage',
                f'{sensor}_raw_voltage',
                f'{sensor}_resistance',
                f'{sensor}_compensated_resistance',
                f'{sensor}_rs_r0_ratio',
                f'{sensor}_ppm',
                f'{sensor}_drift_factor'
            ])
        
        collected_data = []
        
        print(f"\n🎯 ENHANCED DATA COLLECTION: {gas_type.upper()}")
        print(f"⏱️ Duration: {duration} seconds (3 menit - optimal)")
        print(f"📊 Collecting every 2 seconds = ~{duration//2} samples expected")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        sample_count = 0
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                while time.time() - start_time < duration:
                    # Read all sensors
                    readings = self.read_sensors()
                    
                    # Prepare row data
                    row = [
                        datetime.now().isoformat(),
                        gas_type,
                        self.current_temperature,
                        self.current_humidity
                    ]
                    
                    # Add sensor data
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        sensor_data = readings[sensor]
                        row.extend([
                            sensor_data['voltage'],
                            sensor_data['raw_voltage'],
                            sensor_data['resistance'],
                            sensor_data['compensated_resistance'],
                            sensor_data['rs_r0_ratio'],
                            sensor_data['ppm'],
                            sensor_data['drift_factor']
                        ])
                    
                    writer.writerow(row)
                    collected_data.append(row)
                    sample_count += 1
                    
                    # Enhanced progress display
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    progress = (elapsed / duration) * 100
                    
                    if sample_count % 10 == 0:
                        print(f"⏱️ Progress: {progress:.1f}% | Time: {elapsed:.0f}s | Samples: {sample_count} | Remaining: {remaining:.0f}s")
                        
                        # Show current sensor status
                        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                            ppm = readings[sensor]['ppm']
                            voltage = readings[sensor]['raw_voltage']
                            status = "🟢" if ppm > 10 else "🟡" if ppm > 5 else "⚪"
                            print(f"   {status} {sensor}: {voltage:.3f}V → {ppm:.1f} PPM")
                        print()
                    
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            print(f"\n⏹️ Collection stopped by user at {sample_count} samples")
        
        # Enhanced completion report
        print(f"\n✅ ENHANCED DATA COLLECTION COMPLETED!")
        print(f"📁 File saved: {filename}")
        print(f"📊 Total samples: {sample_count}")
        print(f"⏱️ Actual duration: {time.time() - start_time:.1f} seconds")
        print(f"📈 Sample rate: {sample_count / (time.time() - start_time) * 60:.1f} samples/minute")
        
        # Quick validation
        if sample_count >= 60:  # Minimal samples untuk 3 menit
            print(f"✅ Sample count adequate for training")
        else:
            print(f"⚠️ Sample count low - consider extending collection time")
        
        return filename, sample_count

    def train_model(self):
        """Enhanced model training with adaptive feature processing"""
        print("\n🤖 ADAPTIVE MODEL TRAINING WITH PPM RANGE COMPENSATION")
        print("="*60)
        
        # Find training files
        training_files = self.find_training_files()
        
        if not training_files:
            print("❌ No training data found!")
            print("📁 Looking for files matching patterns:")
            print("   - training_*.csv")
            print("   - data/training_*.csv") 
            print("   - *training*.csv")
            print("\n💡 Make sure training files are in current directory or 'data/' folder")
            print("Collect training data first using option 2")
            return False
        
        print(f"📂 Found {len(training_files)} training files:")
        for file in training_files:
            print(f"   ✅ {file}")
        
        # Load and combine data
        print(f"\n📊 Loading training data...")
        all_data = []
        total_samples = 0
        gas_type_counts = {}
        
        # Collect training statistics
        training_stats = {}
        
        for file in training_files:
            try:
                df = pd.read_csv(file)
                samples = len(df)
                total_samples += samples
                
                # Count gas types
                if 'gas_type' in df.columns:
                    gas_types = df['gas_type'].value_counts().to_dict()
                    for gas, count in gas_types.items():
                        gas_type_counts[gas] = gas_type_counts.get(gas, 0) + count
                
                # Collect statistics for adaptive processing
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    for feature_type in ['voltage', 'ppm']:
                        col_name = f'{sensor}_{feature_type}'
                        if col_name in df.columns:
                            values = df[col_name].dropna()
                            if len(values) > 0:
                                if col_name not in training_stats:
                                    training_stats[col_name] = {
                                        'min': float(values.min()),
                                        'max': float(values.max()),
                                        'mean': float(values.mean()),
                                        'std': float(values.std())
                                    }
                                else:
                                    # Update stats with new data
                                    existing = training_stats[col_name]
                                    training_stats[col_name]['min'] = min(existing['min'], float(values.min()))
                                    training_stats[col_name]['max'] = max(existing['max'], float(values.max()))
                
                all_data.append(df)
                print(f"   📄 {os.path.basename(file)}: {samples} samples")
                
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
                continue
        
        if not all_data:
            print("❌ No valid training data could be loaded!")
            return False
        
        # Update adaptive processor with training statistics
        self.adaptive_processor.training_stats = training_stats
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📈 Training Data Summary:")
        print(f"   Total samples: {total_samples}")
        print(f"   Gas types: {len(gas_type_counts)}")
        for gas, count in gas_type_counts.items():
            percentage = (count / total_samples) * 100
            print(f"     {gas}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n📊 PPM RANGE ANALYSIS:")
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            ppm_col = f'{sensor}_ppm'
            if ppm_col in combined_df.columns:
                ppm_values = combined_df[ppm_col].dropna()
                if len(ppm_values) > 0:
                    print(f"   {sensor}: {ppm_values.min():.1f} - {ppm_values.max():.1f} PPM")
        
        # Prepare features and target with adaptive processing
        print(f"\n🔧 Preparing adaptive features...")
        
        # Define base feature columns
        base_feature_columns = []
        for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
            base_feature_columns.extend([
                f'{sensor}_voltage',
                f'{sensor}_ppm'
            ])
        
        # Add environmental features if available
        if 'temperature' in combined_df.columns:
            base_feature_columns.append('temperature')
        if 'humidity' in combined_df.columns:
            base_feature_columns.append('humidity')
        
        # Extract and transform features
        print(f"   🎯 Applying adaptive feature transformations...")
        
        # Prepare feature matrix
        X_raw = combined_df[base_feature_columns].fillna(0)
        y = combined_df['gas_type']
        
        # Apply adaptive transformations to training data
        X_transformed = []
        for idx, row in X_raw.iterrows():
            feature_vector = row.values.tolist()
            transformed_features = self.adaptive_processor.transform_features(
                feature_vector, base_feature_columns
            )
            X_transformed.append(transformed_features)
        
        X_transformed = np.array(X_transformed)
        
        # Store feature names for consistency
        self.feature_names = base_feature_columns
        
        print(f"   Original features: {len(base_feature_columns)}")
        print(f"   Feature columns: {base_feature_columns}")
        print(f"   Adaptive transformation method: {self.adaptive_processor.best_transform_method}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Scale features
        print(f"\n⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\n🎯 Training enhanced RandomForest model...")
        self.model = RandomForestClassifier(
            n_estimators=150,  # Increased for better performance
            random_state=42,
            max_depth=12,      # Slightly deeper for complex patterns
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print(f"\n📊 Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        for gas_type, metrics in report.items():
            if gas_type not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = metrics['precision']
                recall = metrics['recall'] 
                f1 = metrics['f1-score']
                support = int(metrics['support'])
                print(f"   {gas_type}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({support} samples)")
        
        # Save model and enhanced metadata
        print(f"\n💾 Saving enhanced model...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/gas_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save enhanced metadata with training statistics
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': 'adaptive_v4.4_ppm_range_fixed',
            'total_samples': total_samples,
            'feature_columns': base_feature_columns,
            'feature_names': self.feature_names,
            'gas_types': list(gas_type_counts.keys()),
            'gas_type_counts': gas_type_counts,
            'accuracy': accuracy,
            'model_params': self.model.get_params(),
            'training_files': training_files,
            'training_stats': training_stats,  # IMPORTANT: PPM range info
            'adaptive_processor_config': {
                'best_transform_method': self.adaptive_processor.best_transform_method,
                'ppm_scaling_factors': self.adaptive_processor.ppm_scaling_factors
            }
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update adaptive processor training stats
        self.adaptive_processor.training_stats = training_stats
        self.adaptive_processor.save_adaptive_config()
        
        self.is_model_trained = True
        self.training_metadata = metadata
        
        print(f"✅ Enhanced adaptive model training completed!")
        print(f"📁 Files saved:")
        print(f"   🤖 models/gas_classifier.pkl")
        print(f"   ⚖️  models/scaler.pkl") 
        print(f"   📋 models/model_metadata.json (with PPM range info)")
        print(f"   🎯 adaptive_feature_config.json")
        
        print(f"\n🎯 ADAPTIVE FEATURES ACTIVE:")
        print(f"   PPM Range Compensation: ✅")
        print(f"   Multi-scale Features: ✅")
        print(f"   Confidence Boosting: ✅")
        print(f"   Temporal Smoothing: ✅")
        
        return True

    def load_model(self):
        """Load trained model with adaptive features"""
        try:
            self.model = joblib.load('models/gas_classifier.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            
            # Load enhanced metadata with training statistics
            try:
                with open('models/model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', metadata.get('feature_columns', []))
                self.training_metadata = metadata
                
                # Update adaptive processor with training stats
                if 'training_stats' in metadata:
                    self.adaptive_processor.training_stats = metadata['training_stats']
                
                # Load adaptive processor config
                if 'adaptive_processor_config' in metadata:
                    config = metadata['adaptive_processor_config']
                    self.adaptive_processor.best_transform_method = config.get(
                        'best_transform_method', 'adaptive_normalize'
                    )
                    self.adaptive_processor.ppm_scaling_factors = config.get(
                        'ppm_scaling_factors', self.adaptive_processor.ppm_scaling_factors
                    )
                
            except Exception as e:
                self.logger.warning(f"Could not load enhanced metadata: {e}")
                # Fallback feature names
                self.feature_names = ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                                     'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']
            
            # Load adaptive processor config
            self.adaptive_processor.load_adaptive_config()
            
            self.is_model_trained = True
            self.logger.info("Enhanced adaptive model loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("No trained model found")
            return False

    def predict_gas(self, readings):
        """Enhanced gas prediction with adaptive features and confidence boosting"""
        if not self.is_model_trained:
            return "Unknown - Model not trained", 0.0

        try:
            # Extract multi-scale features
            multi_scale_features = self.adaptive_processor.extract_multi_scale_features(readings)
            
            # Use base feature names for consistency
            feature_columns = self.feature_names or ['TGS2600_voltage', 'TGS2600_ppm', 'TGS2602_voltage',
                                                    'TGS2602_ppm', 'TGS2610_voltage', 'TGS2610_ppm']

            feature_vector = []
            for feature in feature_columns:
                if feature == 'temperature':
                    feature_vector.append(self.current_temperature)
                elif feature == 'humidity':
                    feature_vector.append(self.current_humidity)
                else:
                    # Use multi-scale features if available, otherwise fall back to basic features
                    if feature in multi_scale_features:
                        value = multi_scale_features[feature]
                    else:
                        parts = feature.split('_')
                        sensor = parts[0]
                        measurement = '_'.join(parts[1:])
                        
                        if sensor in readings and measurement in readings[sensor]:
                            value = readings[sensor][measurement]
                        else:
                            value = 0.0
                    
                    feature_vector.append(value if value is not None else 0.0)

            # Apply adaptive feature transformations
            transformed_features = self.adaptive_processor.transform_features(feature_vector, feature_columns)

            # Create DataFrame with feature names to avoid warnings
            features_df = pd.DataFrame([transformed_features], columns=feature_columns)
            features_array = np.nan_to_num(features_df.values, nan=0.0)

            # Scale and predict
            features_scaled = self.scaler.transform(features_array)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            base_confidence = probabilities.max()

            # Apply confidence boosting
            boosted_confidence = self.prediction_engine.boost_confidence_with_patterns(
                prediction, base_confidence, readings
            )

            # Apply temporal smoothing
            final_prediction, final_confidence = self.prediction_engine.apply_temporal_smoothing(
                prediction, boosted_confidence
            )

            # Apply multi-evidence fusion
            fused_prediction, fused_confidence = self.prediction_engine.fuse_multi_evidence(
                final_prediction, final_confidence, readings
            )

            # Apply adaptive confidence threshold
            adaptive_threshold = self.adaptive_processor.get_adaptive_confidence_threshold(
                fused_prediction, fused_confidence
            )

            # If confidence is below adaptive threshold, classify as uncertain
            if fused_confidence < adaptive_threshold:
                return f"{fused_prediction} (uncertain)", fused_confidence

            return fused_prediction, fused_confidence

        except Exception as e:
            self.logger.error(f"Error in enhanced prediction: {e}")
            return "Error", 0.0

    def set_environmental_conditions(self, temperature=None, humidity=None):
        """Set environmental conditions"""
        if temperature is not None:
            self.current_temperature = temperature
            self.logger.info(f"Temperature set to {temperature}°C")

        if humidity is not None:
            self.current_humidity = humidity
            self.logger.info(f"Humidity set to {humidity}%RH")

    # TAMBAHAN: Enhanced prediction system methods
    def init_enhanced_prediction_system(self):
        """Initialize enhanced prediction system"""
        self.enhanced_predictor = EnhancedPredictor(self)
        
        # Try to load existing enhanced model
        enhanced_loaded = self.enhanced_predictor.load_enhanced_model()
        
        if enhanced_loaded:
            # Backup original predict_gas method
            self.predict_gas_original = self.predict_gas
            
            # Replace with enhanced version
            self.predict_gas = self.enhanced_predictor.predict_gas_enhanced
            self.prediction_mode = 'enhanced'
            
            self.logger.info("🚀 Enhanced prediction system activated")
            return True
        else:
            self.logger.info("📊 Enhanced model not found, using original system")
            return False
    
    def switch_prediction_mode(self, mode='auto'):
        """Switch between original and enhanced prediction"""
        if mode == 'auto':
            # Auto-switch based on availability
            if hasattr(self, 'enhanced_predictor') and self.enhanced_predictor.is_enhanced_loaded:
                mode = 'enhanced'
            else:
                mode = 'original'
        
        if mode == 'enhanced' and hasattr(self, 'enhanced_predictor') and self.enhanced_predictor.is_enhanced_loaded:
            if not hasattr(self, 'predict_gas_original'):
                self.predict_gas_original = self.predict_gas
            self.predict_gas = self.enhanced_predictor.predict_gas_enhanced
            self.prediction_mode = 'enhanced'
            self.logger.info("🚀 Switched to enhanced prediction mode")
            return True
            
        elif mode == 'original':
            if hasattr(self, 'predict_gas_original'):
                self.predict_gas = self.predict_gas_original
                self.prediction_mode = 'original'
                self.logger.info("📊 Switched to original prediction mode")
                return True
        
        return False
    
    def run_enhanced_fix_opsi1(self):
        """Run enhanced fix untuk Opsi 1"""
        print("\n🚀 RUNNING ENHANCED PREDICTION FIX - OPSI 1")
        print("="*60)
        print("Memperbaiki prediksi tanpa collecting data ulang")
        print("Menggunakan enhanced features dan model balancing")
        
        # Step 1: Analyze existing data
        print(f"\n📊 STEP 1: ANALYZING EXISTING TRAINING DATA")
        if not self.data_analyzer.find_and_load_training_data():
            print("❌ No training data found for enhancement!")
            print("💡 Please collect training data first (Option 2)")
            return False
        
        # Step 2: Quality analysis
        print(f"\n🔍 STEP 2: DATA QUALITY ANALYSIS")
        self.data_analyzer.analyze_data_quality()
        
        recommendation = self.data_analyzer.get_recommendation()
        print(f"\n💡 RECOMMENDATION: {recommendation}")
        
        if "Too few" in recommendation:
            print("⚠️ Warning: Data mungkin kurang untuk enhancement yang optimal")
            proceed = input("Lanjutkan? (y/n): ").lower().strip()
            if proceed != 'y':
                return False
        
        # Step 3: Enhanced feature extraction dan training
        print(f"\n🎯 STEP 3: ENHANCED FEATURE EXTRACTION & TRAINING")
        
        # Initialize trainer
        self.enhanced_trainer = EnhancedModelTrainer(self.logger)
        
        # Extract enhanced features
        feature_extractor = EnhancedFeatureExtractor(self.logger)
        enhanced_data = feature_extractor.extract_enhanced_features(self.data_analyzer.combined_data)
        
        # Balance and augment data
        balanced_data = self.enhanced_trainer.balance_and_augment_data(enhanced_data)
        
        # Train enhanced model
        training_success = self.enhanced_trainer.train_enhanced_model(
            balanced_data, feature_extractor.feature_names
        )
        
        if not training_success:
            print("\n❌ ENHANCED TRAINING FAILED!")
            print("💡 Model accuracy too low or poor F1-scores")
            print("💡 Recommendation: Try Opsi 2 (systematic data collection)")
            return False
        
        # Step 4: Save enhanced model
        print(f"\n💾 STEP 4: SAVING ENHANCED MODEL")
        self.enhanced_trainer.save_enhanced_model()
        
        # Step 5: Load dan activate enhanced system
        print(f"\n🚀 STEP 5: ACTIVATING ENHANCED PREDICTION SYSTEM")
        enhanced_activated = self.init_enhanced_prediction_system()
        
        if enhanced_activated:
            print(f"\n✅ ENHANCED FIX OPSI 1 - BERHASIL!")
            print(f"🎯 Enhanced prediction system is now ACTIVE")
            print(f"🎯 Model dapat memprediksi semua gas types dengan akurasi tinggi")
            print(f"🎯 Enhanced features meningkatkan separability antar gas")
            
            # Test enhanced prediction
            print(f"\n🧪 TESTING ENHANCED PREDICTION:")
            readings = self.read_sensors()
            prediction, confidence = self.predict_gas(readings)
            print(f"   Current prediction: {prediction} (confidence: {confidence:.3f})")
            
            return True
        else:
            print(f"\n❌ FAILED TO ACTIVATE ENHANCED SYSTEM")
            return False

def main():
    """Enhanced main function dengan Opsi 1 terintegrasi"""
    gas_sensor = EnhancedDatasheetGasSensorArray()

    # Load existing calibration
    calibration_loaded = gas_sensor.load_calibration()
    
    if not calibration_loaded:
        print("\n⚠️ CALIBRATION NOT FOUND")
        print("Advanced features available as fallback")

    # Load existing model dan initialize enhanced system
    model_loaded = gas_sensor.load_model()
    enhanced_loaded = gas_sensor.init_enhanced_prediction_system()
    
    if enhanced_loaded:
        print("\n🚀 ENHANCED PREDICTION SYSTEM ACTIVE")
        print("✅ Enhanced features untuk better gas discrimination")
        print("✅ Pattern-based confidence boosting")
    elif model_loaded:
        print("\n📊 ORIGINAL MODEL LOADED")
        print("💡 Run Enhanced Fix (Option 36) untuk upgrade prediction")

    while True:
        print("\n" + "="*70)
        print("🧠 Enhanced Gas Sensor Array System - v4.4 COMPLETE FIXED")
        print("🎯 Enhanced Prediction + Original Features + All Menu Options")
        print("="*70)
        print("1. Calibrate sensors")
        print("2. Collect training data (3 menit optimal)")
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
        print("🔧 VOLTAGE & SENSITIVITY:")
        print("17. Smart Voltage Adjustment")
        print("18. Sensor Responsivity Check")
        print("19. Quick voltage check")
        print("20. Emergency R0 Fix")
        print("21. Smart Troubleshoot PPM Issues")
        print("22. Toggle Emergency PPM Mode")
        print("23. Emergency PPM Test")
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
        print("🚀 ENHANCED PREDICTION SYSTEM:")
        print("36. ⭐ RUN ENHANCED FIX (Improve prediction tanpa collecting ulang)")
        print("37. Analyze current training data quality")
        print("38. Test enhanced vs original prediction")
        print("39. Switch prediction mode (enhanced/original)")
        print("40. Enhanced prediction diagnostic")
        print("41. View enhanced model info")
        print("42. Quick collecting guide (3 menit optimal)")
        print("-"*70)

        try:
            choice = input("Select option (1-42): ").strip()

            if choice == '1':
                duration = int(input("Calibration duration (seconds, default 300): ") or 300)
                print("Ensure sensors are warmed up for at least 10 minutes in clean air!")
                confirm = input("Continue with calibration? (y/n): ").lower()
                if confirm == 'y':
                    gas_sensor.calibrate_sensors(duration)

            elif choice == '2':
                print("\n🎯 ENHANCED DATA COLLECTION (3 MENIT OPTIMAL)")
                print("Available gas types:")
                print("1. normal (clean air)")
                print("2. alcohol")
                print("3. pertalite") 
                print("4. pertamax")
                print("5. dexlite")
                print("6. toluene")
                print("7. ammonia")
                print("8. butane")
                print("9. propane")
                print("10. custom")
                
                gas_choice = input("Select gas type (1-10): ").strip()
                gas_types = {
                    '1': 'normal', '2': 'alcohol', '3': 'pertalite', '4': 'pertamax',
                    '5': 'dexlite', '6': 'toluene', '7': 'ammonia', '8': 'butane', '9': 'propane'
                }
                
                if gas_choice in gas_types:
                    gas_type = gas_types[gas_choice]
                elif gas_choice == '10':
                    gas_type = input("Enter custom gas type: ").strip()
                else:
                    print("❌ Invalid choice")
                    continue
                
                duration = int(input(f"Duration for {gas_type} (seconds, default 180=3min): ") or 180)
                
                print(f"\n🎯 Prepare {gas_type} environment")
                if gas_type == 'normal':
                    print("✅ Ensure clean air environment")
                    print("💡 Tips: No perfume, cooking smell, or chemicals nearby")
                else:
                    print(f"🧪 Prepare {gas_type} source")
                    print("💡 Tips: Use spray bottle, cotton bud, or small container")
                    print("💡 Distance: 30-50cm from sensors for consistent response")
                
                input("Press Enter when ready to start collection...")
                gas_sensor.collect_training_data(gas_type, duration)

            elif choice == '3':
                gas_sensor.train_model()

            elif choice == '4':
                gas_sensor.set_sensor_mode('datasheet')
                print("\n🔬 Starting ADAPTIVE monitoring mode...")
                print("🎯 Enhanced detection with PPM range compensation")
                print("Press Ctrl+C to stop\n")
                
                # Start monitoring with CSV saving
                gas_sensor.monitoring_collector.start_monitoring(gas_sensor, 'datasheet', save_to_csv=True)
                
                try:
                    while gas_sensor.monitoring_collector.is_collecting:
                        readings = gas_sensor.read_sensors()
                        predicted_gas, confidence = gas_sensor.predict_gas(readings)
                        
                        print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | "
                              f"🎯 {predicted_gas} ({confidence:.3f}) | "
                              f"TGS2600: {readings['TGS2600']['ppm']:.1f} | "
                              f"TGS2602: {readings['TGS2602']['ppm']:.1f} | "
                              f"TGS2610: {readings['TGS2610']['ppm']:.1f}", end="")
                        
                        time.sleep(2)
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️  Monitoring stopped by user")
                finally:
                    gas_sensor.monitoring_collector.stop_monitoring()

            elif choice == '5':
                gas_sensor.set_sensor_mode('extended')
                print("\n🚀 Starting extended monitoring mode with CSV saving...")
                print("Press Ctrl+C to stop\n")
                
                # Start monitoring with CSV saving
                gas_sensor.monitoring_collector.start_monitoring(gas_sensor, 'extended', save_to_csv=True)
                
                try:
                    while gas_sensor.monitoring_collector.is_collecting:
                        readings = gas_sensor.read_sensors()
                        predicted_gas, confidence = gas_sensor.predict_gas(readings)
                        
                        print(f"\r⏰ {datetime.now().strftime('%H:%M:%S')} | "
                              f"🎯 {predicted_gas} ({confidence:.3f}) | "
                              f"TGS2600: {readings['TGS2600']['ppm']:.1f} | "
                              f"TGS2602: {readings['TGS2602']['ppm']:.1f} | "
                              f"TGS2610: {readings['TGS2610']['ppm']:.1f}", end="")
                        
                        time.sleep(2)
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️  Monitoring stopped by user")
                finally:
                    gas_sensor.monitoring_collector.stop_monitoring()

            elif choice == '6':
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)

                print("\n" + "="*70)
                print("🎯 ADAPTIVE SENSOR ANALYSIS - ENHANCED DETECTION")
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

                print(f"\n🎯 ADAPTIVE PREDICTION:")
                print(f"  Gas Type: {predicted_gas}")
                print(f"  Confidence: {confidence:.3f}")
                
                # Show adaptive status
                print(f"\n🔧 ADAPTIVE STATUS:")
                print(f"  PPM Scaling: {gas_sensor.adaptive_processor.ppm_scaling_factors}")
                print(f"  Transform Method: {gas_sensor.adaptive_processor.best_transform_method}")
                print(f"  Multi-scale Features: {gas_sensor.adaptive_processor.enable_multi_scale}")

            elif choice == '7':
                print("\n🌡️ SET ENVIRONMENTAL CONDITIONS")
                temp = input("Temperature (°C, current: {:.1f}): ".format(gas_sensor.current_temperature))
                humidity = input("Humidity (%RH, current: {:.1f}): ".format(gas_sensor.current_humidity))
                
                if temp:
                    gas_sensor.set_environmental_conditions(temperature=float(temp))
                if humidity:
                    gas_sensor.set_environmental_conditions(humidity=float(humidity))

            elif choice == '8':
                print("\n⚙️ SENSOR CALCULATION MODE")
                print("1. Datasheet mode (accurate)")
                print("2. Extended mode (sensitive)")
                
                mode_choice = input("Select mode (1-2): ").strip()
                if mode_choice == '1':
                    gas_sensor.set_sensor_mode('datasheet')
                elif mode_choice == '2':
                    gas_sensor.set_sensor_mode('extended')

            elif choice == '9':
                readings = gas_sensor.read_sensors()
                
                print("\n📊 COMPLETE SENSOR DIAGNOSTICS")
                print("="*50)
                
                for sensor, data in readings.items():
                    print(f"\n{sensor}:")
                    print(f"  Status: {'✅ OK' if data['mode'] != 'Error' else '❌ ERROR'}")
                    print(f"  Raw Voltage: {data['raw_voltage']:.3f}V")
                    print(f"  Compensated: {data['voltage']:.3f}V")
                    print(f"  Resistance: {data['resistance']:.1f}Ω")
                    print(f"  PPM: {data['ppm']:.1f}")
                    print(f"  Mode: {data['mode']}")
                    print(f"  Drift Applied: {'Yes' if data['smart_compensation_applied'] else 'No'}")

            # SMART DRIFT COMPENSATION (11-16)
            elif choice == '11':
                gas_sensor.drift_manager.smart_daily_drift_check(gas_sensor)

            elif choice == '12':
                duration = int(input("Test duration (seconds, default 60): ") or 60)
                gas_sensor.drift_manager.quick_stability_test(gas_sensor, duration)

            elif choice == '13':
                gas_sensor.drift_manager.smart_drift_status_report()

            elif choice == '14':
                gas_sensor.drift_manager.manual_drift_compensation_reset()

            elif choice == '15':
                gas_sensor.drift_manager.auto_baseline_reset(gas_sensor)

            elif choice == '16':
                gas_sensor.drift_manager.smart_system_health_check(gas_sensor)

            # VOLTAGE ADJUSTMENT (17-19)
            elif choice == '17':
                gas_sensor.drift_manager.smart_voltage_adjustment(gas_sensor)

            elif choice == '18':
                gas_sensor.drift_manager.sensor_responsivity_check(gas_sensor)

            elif choice == '19':
                gas_sensor.drift_manager.quick_voltage_check(gas_sensor)

            # EMERGENCY PPM RECOVERY (20-28)
            elif choice == '20':
                print("\n🔧 EMERGENCY R0 FIX")
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor:")
                for i, sensor in enumerate(sensors, 1):
                    print(f"{i}. {sensor}")
                print("4. All sensors")
                
                sensor_choice = input("Enter choice (1-4): ").strip()
                
                if sensor_choice == '4':
                    for sensor in sensors:
                        gas_sensor.drift_manager.emergency_r0_fix(gas_sensor, sensor)
                elif sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    gas_sensor.drift_manager.emergency_r0_fix(gas_sensor, target_sensor)

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

            elif choice == '22':
                print("\n🚨 TOGGLE EMERGENCY PPM MODE")
                sensors = ['TGS2600', 'TGS2602', 'TGS2610']
                print(f"\nSelect sensor:")
                for i, sensor in enumerate(sensors, 1):
                    current_mode = gas_sensor.sensor_config[sensor].get('use_emergency_ppm', False)
                    status = "ACTIVE" if current_mode else "INACTIVE"
                    print(f"{i}. {sensor} (Currently: {status})")
                
                sensor_choice = input("Enter choice (1-3): ").strip()
                
                if sensor_choice in ['1', '2', '3']:
                    target_sensor = sensors[int(sensor_choice) - 1]
                    current_mode = gas_sensor.sensor_config[target_sensor].get('use_emergency_ppm', False)
                    new_mode = not current_mode
                    gas_sensor.sensor_config[target_sensor]['use_emergency_ppm'] = new_mode
                    gas_sensor.sensor_config[target_sensor]['emergency_mode'] = new_mode
                    
                    status = "ENABLED" if new_mode else "DISABLED"
                    print(f"✅ Emergency PPM mode {status} for {target_sensor}")

            elif choice == '23':
                print("\n🧪 EMERGENCY PPM TEST")
                readings = gas_sensor.read_sensors()
                
                print("Testing Emergency PPM calculation for all sensors:")
                for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
                    current_voltage = readings[sensor_name]['raw_voltage']
                    emergency_ppm = gas_sensor.emergency_ppm_calc.calculate_emergency_ppm(
                        sensor_name, current_voltage, 'auto', gas_sensor.sensitivity_manager
                    )
                    print(f"  {sensor_name}: {current_voltage:.3f}V → {emergency_ppm:.1f} PPM")

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

            elif choice == '29':
                print("\n🚨 TGS2602 STUCK SENSOR RECOVERY")
                print("This will apply all available fixes for TGS2602")
                confirm = input("Proceed with TGS2602 recovery? (y/n): ").lower()
                if confirm == 'y':
                    # Step 1: Reset sensitivity
                    gas_sensor.sensitivity_manager.current_sensitivity['TGS2602'] = 'normal'
                    gas_sensor.sensitivity_manager.custom_factors['TGS2602'] = 1.0
                    print("✅ Step 1: Sensitivity reset")
                    
                    # Step 2: Disable emergency mode
                    gas_sensor.sensor_config['TGS2602']['use_emergency_ppm'] = False
                    gas_sensor.sensor_config['TGS2602']['emergency_mode'] = False
                    print("✅ Step 2: Emergency mode disabled")
                    
                    # Step 3: Emergency R0 fix
                    current_voltage = gas_sensor.sensor_config['TGS2602']['channel'].voltage
                    current_resistance = gas_sensor.voltage_to_resistance(current_voltage)
                    gas_sensor.sensor_config['TGS2602']['R0'] = current_resistance
                    gas_sensor.sensor_config['TGS2602']['baseline_voltage'] = current_voltage
                    print(f"✅ Step 3: Emergency R0 fix ({current_resistance:.1f}Ω)")
                    
                    # Step 4: Test result
                    readings = gas_sensor.read_sensors()
                    tgs2602_ppm = readings['TGS2602']['ppm']
                    print(f"✅ Step 4: TGS2602 PPM now: {tgs2602_ppm:.1f}")
                    
                    if tgs2602_ppm < 200:
                        print("🎉 TGS2602 RECOVERY SUCCESSFUL!")
                    else:
                        print("⚠️ TGS2602 still showing high values")

            elif choice == '30':
                print("\n📊 MONITORING DATA STATISTICS")
                
                # Find monitoring data files
                monitoring_files = glob.glob("monitoring_data_*.csv")
                if not monitoring_files:
                    print("❌ No monitoring data files found")
                    continue
                
                print(f"Found {len(monitoring_files)} monitoring files:")
                for file in monitoring_files:
                    try:
                        df = pd.read_csv(file)
                        print(f"\n📄 {file}:")
                        print(f"   Samples: {len(df)}")
                        if 'predicted_gas' in df.columns:
                            gas_types = df['predicted_gas'].value_counts()
                            print(f"   Gas predictions: {dict(gas_types)}")
                        if 'confidence' in df.columns:
                            avg_confidence = df['confidence'].mean()
                            print(f"   Average confidence: {avg_confidence:.3f}")
                    except Exception as e:
                        print(f"   Error reading {file}: {e}")

            # ADAPTIVE FEATURES (31-35)
            elif choice == '31':
                print("\n🎯 PPM RANGE COMPENSATION STATUS")
                print("="*50)
                
                print("Training PPM Ranges:")
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    ppm_key = f'{sensor}_ppm'
                    if ppm_key in gas_sensor.adaptive_processor.training_stats:
                        stats = gas_sensor.adaptive_processor.training_stats[ppm_key]
                        print(f"  {sensor}: {stats['min']:.1f} - {stats['max']:.1f} PPM")
                
                print("\nCurrent PPM Ranges:")
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    ppm_key = f'{sensor}_ppm'
                    if ppm_key in gas_sensor.adaptive_processor.current_stats:
                        stats = gas_sensor.adaptive_processor.current_stats[ppm_key]
                        print(f"  {sensor}: {stats['min']:.1f} - {stats['max']:.1f} PPM")
                
                print("\nScaling Factors:")
                for sensor, factor in gas_sensor.adaptive_processor.ppm_scaling_factors.items():
                    print(f"  {sensor}: {factor:.3f}x")

            elif choice == '32':
                print("\n🔧 ADJUST ADAPTIVE TRANSFORMATION METHOD")
                print("="*50)
                
                methods = list(gas_sensor.adaptive_processor.transformation_methods.keys())
                current = gas_sensor.adaptive_processor.best_transform_method
                
                print(f"Current method: {current}")
                print("\nAvailable methods:")
                for i, method in enumerate(methods, 1):
                    print(f"{i}. {method}")
                
                choice_method = input("Select method (1-5): ").strip()
                if choice_method in ['1', '2', '3', '4', '5']:
                    new_method = methods[int(choice_method) - 1]
                    gas_sensor.adaptive_processor.best_transform_method = new_method
                    gas_sensor.adaptive_processor.save_adaptive_config()
                    print(f"✅ Transformation method set to: {new_method}")

            elif choice == '33':
                print("\n🎯 MULTI-SCALE FEATURES")
                print("="*40)
                
                current_status = gas_sensor.adaptive_processor.enable_multi_scale
                print(f"Current status: {'ENABLED' if current_status else 'DISABLED'}")
                
                toggle = input("Toggle multi-scale features? (y/n): ").lower()
                if toggle == 'y':
                    gas_sensor.adaptive_processor.enable_multi_scale = not current_status
                    new_status = gas_sensor.adaptive_processor.enable_multi_scale
                    gas_sensor.adaptive_processor.save_adaptive_config()
                    print(f"✅ Multi-scale features: {'ENABLED' if new_status else 'DISABLED'}")

            elif choice == '34':
                print("\n📊 PREDICTION CONFIDENCE ANALYSIS")
                print("="*50)
                
                readings = gas_sensor.read_sensors()
                predicted_gas, confidence = gas_sensor.predict_gas(readings)
                
                # Get detailed analysis
                pattern_scores = gas_sensor.prediction_engine.analyze_sensor_patterns(readings)
                
                print(f"Current Prediction: {predicted_gas}")
                print(f"Base Confidence: {confidence:.3f}")
                
                print(f"\nPattern Analysis Scores:")
                for gas_type, score in pattern_scores.items():
                    print(f"  {gas_type}: {score:.3f}")
                
                print(f"\nAdaptive Threshold: {gas_sensor.adaptive_processor.get_adaptive_confidence_threshold(predicted_gas, confidence):.3f}")
                
                print(f"\nRecent Predictions:")
                recent = gas_sensor.prediction_engine.recent_predictions[-5:] if gas_sensor.prediction_engine.recent_predictions else []
                for pred, conf, timestamp in recent:
                    time_ago = time.time() - timestamp
                    print(f"  {pred} ({conf:.3f}) - {time_ago:.0f}s ago")

            elif choice == '35':
                print("\n🧪 ENHANCED GAS DETECTION TEST")
                print("="*50)
                
                print("Testing with current sensor readings...")
                readings = gas_sensor.read_sensors()
                
                # Test multiple prediction methods
                print("\n📊 DETECTION METHODS COMPARISON:")
                
                # Basic prediction
                predicted_gas, confidence = gas_sensor.predict_gas(readings)
                print(f"1. Enhanced Adaptive: {predicted_gas} (confidence: {confidence:.3f})")
                
                # Pattern analysis
                pattern_scores = gas_sensor.prediction_engine.analyze_sensor_patterns(readings)
                best_pattern = max(pattern_scores.keys(), key=lambda x: pattern_scores[x]) if pattern_scores else "None"
                best_pattern_score = pattern_scores.get(best_pattern, 0.0)
                print(f"2. Pattern Analysis: {best_pattern} (score: {best_pattern_score:.3f})")
                
                # Show detailed sensor responses
                print(f"\n🔬 SENSOR RESPONSES:")
                base_voltage = 1.6
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    if sensor in readings:
                        data = readings[sensor]
                        voltage_drop = base_voltage - data['raw_voltage']
                        print(f"  {sensor}: {data['raw_voltage']:.3f}V (drop: {voltage_drop:.3f}V) → {data['ppm']:.1f} PPM")

            # ENHANCED PREDICTION SYSTEM (36-42)
            elif choice == '36':
                # ENHANCED FIX OPSI 1 - FITUR UTAMA
                success = gas_sensor.run_enhanced_fix_opsi1()
                
                if success:
                    print(f"\n🎉 ENHANCED FIX BERHASIL!")
                    print(f"✅ Enhanced prediction system is now active")
                    print(f"✅ Model can now predict all gas types better")
                    print(f"✅ Enhanced features improve gas discrimination")
                    
                    # Test dengan current readings
                    print(f"\n🧪 Testing current prediction:")
                    readings = gas_sensor.read_sensors()
                    prediction, confidence = gas_sensor.predict_gas(readings)
                    print(f"   Prediction: {prediction}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Mode: {gas_sensor.prediction_mode}")
                    
                else:
                    print(f"\n❌ ENHANCED FIX FAILED")
                    print(f"💡 Possible reasons:")
                    print(f"   - Insufficient training data")
                    print(f"   - Poor data quality/separability")
                    print(f"   - High overlap between gas signatures")
                    print(f"\n💡 Recommendations:")
                    print(f"   1. Try Option 37 to analyze data quality")
                    print(f"   2. Consider systematic data collection (Opsi 2)")
                    print(f"   3. Check sensor responsivity (Option 18)")

            elif choice == '37':
                print("\n🔍 ANALYZING CURRENT TRAINING DATA QUALITY")
                print("="*50)
                
                if gas_sensor.data_analyzer.find_and_load_training_data():
                    gas_sensor.data_analyzer.analyze_data_quality()
                    
                    recommendation = gas_sensor.data_analyzer.get_recommendation()
                    print(f"\n💡 OVERALL RECOMMENDATION:")
                    print(f"   {recommendation}")
                    
                    print(f"\n🎯 NEXT STEPS:")
                    if "acceptable" in recommendation.lower():
                        print(f"   ✅ Data quality good - Run Enhanced Fix (Option 36)")
                    else:
                        print(f"   ⚠️ Data quality issues detected")
                        print(f"   💡 Consider collecting more data or systematic approach")
                else:
                    print("❌ No training data found to analyze")
                    print("💡 Collect training data first (Option 2)")

            elif choice == '38':
                print("\n⚔️ ENHANCED VS ORIGINAL PREDICTION COMPARISON")
                print("="*50)
                
                if hasattr(gas_sensor, 'enhanced_predictor') and gas_sensor.enhanced_predictor.is_enhanced_loaded:
                    readings = gas_sensor.read_sensors()
                    
                    # Test both prediction methods
                    if hasattr(gas_sensor, 'predict_gas_original'):
                        original_pred, original_conf = gas_sensor.predict_gas_original(readings)
                    else:
                        original_pred, original_conf = "N/A", 0.0
                    
                    enhanced_pred, enhanced_conf = gas_sensor.enhanced_predictor.predict_gas_enhanced(readings)
                    
                    print(f"\n📊 PREDICTION COMPARISON:")
                    print(f"   Original Model: {original_pred} (confidence: {original_conf:.3f})")
                    print(f"   Enhanced Model: {enhanced_pred} (confidence: {enhanced_conf:.3f})")
                    
                    if original_pred != enhanced_pred:
                        print(f"   ⚠️ Predictions DIFFER!")
                        print(f"   💡 Enhanced model may have better discrimination")
                    else:
                        print(f"   ✅ Predictions MATCH")
                        
                    if enhanced_conf > original_conf:
                        print(f"   📈 Enhanced model has HIGHER confidence")
                    
                    # Show sensor readings for context
                    print(f"\n🔬 Current sensor readings:")
                    for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                        ppm = readings[sensor]['ppm']
                        voltage = readings[sensor]['raw_voltage']
                        print(f"   {sensor}: {voltage:.3f}V → {ppm:.1f} PPM")
                        
                else:
                    print("❌ Enhanced model not loaded")
                    print("💡 Run Enhanced Fix first (Option 36)")

            elif choice == '39':
                print("\n🔄 SWITCH PREDICTION MODE")
                print("="*40)
                
                current_mode = getattr(gas_sensor, 'prediction_mode', 'original')
                print(f"Current mode: {current_mode.upper()}")
                
                if hasattr(gas_sensor, 'enhanced_predictor') and gas_sensor.enhanced_predictor.is_enhanced_loaded:
                    print("Available modes:")
                    print("1. Original prediction")
                    print("2. Enhanced prediction")
                    
                    mode_choice = input("Select mode (1-2): ").strip()
                    
                    if mode_choice == '1':
                        success = gas_sensor.switch_prediction_mode('original')
                        if success:
                            print("✅ Switched to ORIGINAL prediction mode")
                    elif mode_choice == '2':
                        success = gas_sensor.switch_prediction_mode('enhanced')
                        if success:
                            print("✅ Switched to ENHANCED prediction mode")
                else:
                    print("❌ Enhanced model not available")
                    print("💡 Run Enhanced Fix first (Option 36)")

            elif choice == '40':
                print("\n🔍 ENHANCED PREDICTION DIAGNOSTIC")
                print("="*50)
                
                # Current readings
                readings = gas_sensor.read_sensors()
                
                print("📊 CURRENT SENSOR STATUS:")
                total_response = 0
                for sensor in ['TGS2600', 'TGS2602', 'TGS2610']:
                    ppm = readings[sensor]['ppm']
                    voltage = readings[sensor]['raw_voltage']
                    status = "🟢 Active" if ppm > 10 else "🟡 Low" if ppm > 3 else "⚪ Minimal"
                    print(f"   {sensor}: {voltage:.3f}V → {ppm:.1f} PPM ({status})")
                    total_response += ppm
                
                print(f"\n🎯 PREDICTION ANALYSIS:")
                
                # Test predictions if available
                current_pred, current_conf = gas_sensor.predict_gas(readings)
                print(f"   Current Prediction: {current_pred}")
                print(f"   Confidence: {current_conf:.3f}")
                print(f"   Mode: {getattr(gas_sensor, 'prediction_mode', 'original').upper()}")
                
                # Enhanced feature analysis if available
                if hasattr(gas_sensor, 'enhanced_predictor') and gas_sensor.enhanced_predictor.is_enhanced_loaded:
                    features = gas_sensor.enhanced_predictor.extract_features_from_readings(readings)
                    if features:
                        print(f"\n🔬 ENHANCED FEATURES ANALYSIS:")
                        print(f"   Dominant sensor: {['TGS2600', 'TGS2602', 'TGS2610'][features.get('dominant_sensor', 0)]}")
                        print(f"   Total PPM: {features.get('total_ppm', 0):.1f}")
                        print(f"   Response intensity: {features.get('response_intensity', 0)}/4")
                        
                        # Pattern signatures
                        patterns = []
                        if features.get('alcohol_pattern', 0):
                            patterns.append('alcohol')
                        if features.get('toluene_pattern', 0):
                            patterns.append('toluene')
                        if features.get('hydrocarbon_pattern', 0):
                            patterns.append('hydrocarbon')
                        if features.get('ammonia_pattern', 0):
                            patterns.append('ammonia')
                        if features.get('clean_pattern', 0):
                            patterns.append('clean')
                        
                        print(f"   Detected patterns: {patterns if patterns else ['none']}")
                
                # Recommendations
                print(f"\n💡 DIAGNOSTIC RECOMMENDATIONS:")
                if total_response < 5:
                    print(f"   🎯 Very low sensor response - try gas exposure")
                    print(f"   🎯 Check sensitivity settings (Option 26)")
                elif total_response < 20:
                    print(f"   🔶 Moderate response - acceptable for some gases")
                else:
                    print(f"   ✅ Good sensor response detected")
                
                if current_conf < 0.4:
                    print(f"   ⚠️ Low prediction confidence")
                    print(f"   💡 Try Enhanced Fix if not done (Option 36)")
                elif current_conf < 0.7:
                    print(f"   🔶 Moderate confidence - acceptable")
                else:
                    print(f"   ✅ High confidence prediction")

            elif choice == '41':
                print("\n📋 ENHANCED MODEL INFORMATION")
                print("="*50)
                
                if hasattr(gas_sensor, 'enhanced_predictor') and gas_sensor.enhanced_predictor.is_enhanced_loaded:
                    try:
                        # Load metadata
                        with open('models/enhanced_model_metadata.json', 'r') as f:
                            metadata = json.load(f)
                        
                        print("🤖 ENHANCED MODEL DETAILS:")
                        print(f"   Version: {metadata.get('version', 'Unknown')}")
                        print(f"   Model Type: {metadata.get('model_type', 'Unknown')}")
                        print(f"   Features: {metadata.get('n_features', 'Unknown')}")
                        print(f"   Gas Types: {metadata.get('gas_types', [])}")
                        print(f"   Created: {metadata.get('timestamp', 'Unknown')}")
                        
                        print(f"\n🎯 ENHANCEMENTS APPLIED:")
                        enhancements = metadata.get('enhancements', [])
                        for enhancement in enhancements:
                            print(f"   ✅ {enhancement.replace('_', ' ').title()}")
                        
                        print(f"\n📊 FEATURE CATEGORIES:")
                        feature_names = metadata.get('feature_names', [])
                        
                        basic_features = [f for f in feature_names if '_ppm' in f or '_voltage' in f]
                        ratio_features = [f for f in feature_names if 'ratio_' in f]
                        pattern_features = [f for f in feature_names if '_pattern' in f]
                        other_features = [f for f in feature_names if f not in basic_features + ratio_features + pattern_features]
                        
                        print(f"   Basic Sensor: {len(basic_features)} features")
                        print(f"   Cross-Ratios: {len(ratio_features)} features")
                        print(f"   Gas Patterns: {len(pattern_features)} features")
                        print(f"   Other Enhanced: {len(other_features)} features")
                        
                    except Exception as e:
                        print(f"❌ Error loading enhanced model info: {e}")
                        print("💡 Enhanced model may be corrupted or missing metadata")
                else:
                    print("❌ Enhanced model not loaded")
                    print("💡 Run Enhanced Fix first (Option 36)")

            elif choice == '42':
                print("\n📋 QUICK COLLECTING GUIDE (3 MENIT OPTIMAL)")
                print("="*50)
                print("Based on your successful 3-minute collections:")
                
                print(f"\n🎯 OPTIMAL COLLECTING SEQUENCE:")
                gas_sequence = [
                    ('normal', 'Clean air - no gas sources nearby'),
                    ('alcohol', 'Alcohol spray 70% - 30-50cm distance'),
                    ('pertalite', 'Gasoline in small container + fan'),
                    ('toluene', 'Nail polish remover on cotton bud'),
                    ('ammonia', 'Floor cleaner or cut onion'),
                    ('butane', 'Lighter gas (no spark) - 50cm distance')
                ]
                
                for i, (gas, method) in enumerate(gas_sequence, 1):
                    print(f"\n{i}. {gas.upper()} (3 menit):")
                    print(f"   Method: {method}")
                    print(f"   Wait: 5 menit between gases (sensor recovery)")
                    print(f"   Target: Different sensor response pattern")
                
                print(f"\n⏱️ TOTAL TIME NEEDED:")
                total_collecting = len(gas_sequence) * 3
                total_waiting = (len(gas_sequence) - 1) * 5
                total_time = total_collecting + total_waiting + 10  # +10 for setup
                
                print(f"   Data collecting: {total_collecting} minutes")
                print(f"   Sensor recovery: {total_waiting} minutes")
                print(f"   Setup time: ~10 minutes")
                print(f"   TOTAL: ~{total_time} minutes ({total_time//60:.0f}h {total_time%60:.0f}m)")
                
                print(f"\n✅ SUCCESS CRITERIA PER GAS:")
                print(f"   normal: All PPM < 5")
                print(f"   alcohol: TGS2600 > others")
                print(f"   pertalite: TGS2602 & TGS2610 active")
                print(f"   toluene: TGS2602 >> others")
                print(f"   ammonia: TGS2602 moderate (different from toluene)")
                print(f"   butane: TGS2610 > others")

            elif choice == '10':
                print("👋 Exiting Enhanced Gas Sensor System...")
                # Stop monitoring if running
                if hasattr(gas_sensor, 'monitoring_collector') and gas_sensor.monitoring_collector.is_collecting:
                    gas_sensor.monitoring_collector.stop_monitoring()
                
                # Save all configurations
                if hasattr(gas_sensor, 'drift_manager'):
                    gas_sensor.drift_manager.save_drift_data()
                if hasattr(gas_sensor, 'sensitivity_manager'):
                    gas_sensor.sensitivity_manager.save_sensitivity_data()
                if hasattr(gas_sensor, 'adaptive_processor'):
                    gas_sensor.adaptive_processor.save_adaptive_config()
                
                print("✅ All configurations saved")
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