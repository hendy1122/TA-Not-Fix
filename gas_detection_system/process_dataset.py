#!/usr/bin/env python3
"""
Dataset Processing Script
Memproses dataset dan melatih model untuk deteksi gas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_process_dataset(csv_file):
    """Load and process the merged dataset"""
    print(f"Loading dataset from {csv_file}...")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def train_model(df):
    """Train the gas detection model"""
    print("\nTraining gas detection model...")
    
    # Prepare features and labels
    feature_columns = ['TGS 2600', 'TGS 2602', 'TGS 2610']
    X = df[feature_columns].values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y_encoded.shape}")
    print(f"Unique labels: {label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, label_encoder

def save_model(model, label_encoder, model_file='gas_model.pkl', encoder_file='label_encoder.pkl'):
    """Save trained model and label encoder"""
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_file}")
        
        with open(encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to {encoder_file}")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def visualize_data(df):
    """Create visualizations of the dataset"""
    print("\nCreating data visualizations...")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Distribution of each sensor
        sensor_columns = ['TGS 2600', 'TGS 2602', 'TGS 2610']
        
        for i, sensor in enumerate(sensor_columns):
            row = i // 2
            col = i % 2
            
            for label in df['label'].unique():
                data = df[df['label'] == label][sensor]
                axes[row, col].hist(data, alpha=0.7, label=label, bins=20)
            
            axes[row, col].set_title(f'{sensor} Distribution')
            axes[row, col].set_xlabel('PPM')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Plot 4: Correlation heatmap
        correlation_data = df[sensor_columns].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', 
                   ax=axes[1, 1], center=0)
        axes[1, 1].set_title('Sensor Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'dataset_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    """Main function"""
    print("=== Gas Detection Dataset Processing ===")
    
    # Load dataset
    dataset_file = input("Enter dataset CSV filename (default: merged_dataset.csv): ").strip()
    if not dataset_file:
        dataset_file = "merged_dataset.csv"
    
    df = load_and_process_dataset(dataset_file)
    
    if df is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Create visualizations
    create_viz = input("Create data visualizations? (y/n, default: y): ").strip().lower()
    if create_viz != 'n':
        visualize_data(df)
    
    # Train model
    train_new_model = input("Train new model? (y/n, default: y): ").strip().lower()
    if train_new_model != 'n':
        model, label_encoder = train_model(df)
        save_model(model, label_encoder)
        print("\nModel training completed!")
    
    print("\nDataset processing completed!")

if __name__ == "__main__":
    main()
