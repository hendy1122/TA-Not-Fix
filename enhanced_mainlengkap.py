#!/usr/bin/env python3
"""
Enhanced Gas Sensor Array System - COMPLETE VERSION 4.4 - ENHANCED PREDICTION
OPSI 1: Enhanced prediction tanpa collecting ulang + Original features tetap ada
DURASI COLLECTING: 3 menit (180 detik) - sesuai yang berhasil
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

# [SISIPKAN SEMUA CLASS YANG SUDAH ADA SEBELUMNYA DI SINI]
# AdaptiveFeatureProcessor, EnhancedPredictionEngine, AdvancedSensitivityManager, dll.
# TIDAK DIHAPUS, TETAP ADA SEMUA

# MODIFIKASI CLASS EnhancedDatasheetGasSensorArray
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
                'collecting_duration': 180,  # 3 menit
                'use_extended_mode': False,
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
                'collecting_duration': 180,  # 3 menit
                'use_extended_mode': False,
                'emergency_mode': False,
                'use_emergency_ppm': False,
                'use_advanced_sensitivity': True
            }
        }

        # Initialize semua managers (TETAP ADA SEMUA - TIDAK DIHAPUS)
        # [COPY SEMUA INITIALIZATION DARI KODE ASLI]
        
        # TAMBAHAN: Enhanced prediction system
        self.enhanced_predictor = None
        self.prediction_mode = 'original'  # 'original' atau 'enhanced'
        
        # Data analyzer untuk opsi 1
        self.data_analyzer = EnhancedDataAnalyzer(self.logger)
        
        # Enhanced model trainer
        self.enhanced_trainer = None

        self.logger.info("Enhanced Gas Sensor Array System v4.4 - OPSI 1 READY")

    # TAMBAH METHOD UNTUK ENHANCED PREDICTION SYSTEM
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
    
    # MODIFIKASI COLLECT TRAINING DATA UNTUK 3 MENIT
    def collect_training_data(self, gas_type, duration=180):  # Default 3 menit
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

    # [SISIPKAN SEMUA METHOD LAINNYA DARI KODE ASLI - TIDAK DIHAPUS]
    # voltage_to_resistance, temperature_compensation, humidity_compensation,
    # resistance_to_ppm, read_sensors, calibrate_sensors, dll.

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
        print("🧠 Enhanced Gas Sensor Array System - v4.4 OPSI 1")
        print("🎯 Enhanced Prediction + Original Features (3 menit collecting)")
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
        print("🚀 ENHANCED PREDICTION SYSTEM (OPSI 1):")
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

            # [TAMBAHKAN SEMUA HANDLING CHOICE LAINNYA DARI KODE ASLI]
            # elif choice == '4', '5', '6', dll. - SEMUA TETAP ADA

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
