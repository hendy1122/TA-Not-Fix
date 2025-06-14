#!/usr/bin/env python3
"""
Test script untuk validasi sensor gas TGS 2600, 2602, 2610
Compatible dengan main.py EnhancedDatasheetGasSensorArray
"""

import time
import sys
import json
from datetime import datetime

# Import your actual gas detection system
try:
    from main import EnhancedDatasheetGasSensorArray
    print("✅ Successfully imported EnhancedDatasheetGasSensorArray from main.py")
except ImportError as e:
    print(f"❌ Error importing main.py: {e}")
    print("Make sure main.py is in the same directory")
    sys.exit(1)

class SensorValidator:
    def __init__(self):
        self.test_results = {}
        
        # Initialize your gas sensor system
        try:
            self.gas_detector = EnhancedDatasheetGasSensorArray()
            print("✅ Gas sensor system initialized")
        except Exception as e:
            print(f"❌ Failed to initialize gas sensor system: {e}")
            sys.exit(1)
    
    def test_sensor_connectivity(self):
        """Test connectivity of all sensors"""
        print("\n🔍 Testing Sensor Connectivity...")
        
        try:
            # Test sensor reading using your main.py method
            readings = self.gas_detector.read_sensors()
            
            # Check each sensor
            for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
                if sensor_name in readings:
                    data = readings[sensor_name]
                    voltage = data['voltage']
                    resistance = data['resistance']
                    ppm = data['ppm']
                    
                    print(f"✅ {sensor_name}: Voltage={voltage:.3f}V, Resistance={resistance:.1f}Ω, PPM={ppm:.1f}")
                    
                    # Basic sanity checks
                    if voltage < 0.1 or voltage > 5.0:
                        print(f"   ⚠️  Warning: Unusual voltage reading for {sensor_name}")
                    if resistance < 1 or resistance > 1000000:
                        print(f"   ⚠️  Warning: Unusual resistance reading for {sensor_name}")
                else:
                    print(f"❌ {sensor_name}: No reading available")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Sensor connectivity error: {e}")
            return False
    
    def test_gas_detection_accuracy(self):
        """Test gas detection accuracy with known samples"""
        print("\n🧪 Testing Gas Detection Accuracy...")
        
        # Test clean air baseline
        print("📍 Test 1: Clean Air Baseline")
        clean_air_readings = self.gas_detector.read_sensors()
        predicted_gas, confidence = self.gas_detector.predict_gas(clean_air_readings)
        
        print(f"Clean air prediction: {predicted_gas} (confidence: {confidence:.2f})")
        
        # Calculate average PPM for clean air
        avg_clean_ppm = sum([clean_air_readings[sensor]['ppm'] for sensor in ['TGS2600', 'TGS2602', 'TGS2610']]) / 3
        print(f"Average clean air PPM: {avg_clean_ppm:.1f}")
        
        # Test with gas samples
        gas_tests = ['alcohol', 'pertalite', 'dexlite']
        test_results = {}
        
        for gas_type in gas_tests:
            input(f"\n🔥 Prepare {gas_type.upper()} sample, then press Enter...")
            
            print(f"📍 Testing {gas_type} detection...")
            readings = []
            
            for i in range(5):
                sensor_readings = self.gas_detector.read_sensors()
                predicted, conf = self.gas_detector.predict_gas(sensor_readings)
                
                avg_ppm = sum([sensor_readings[sensor]['ppm'] for sensor in ['TGS2600', 'TGS2602', 'TGS2610']]) / 3
                
                readings.append({
                    'predicted': predicted,
                    'confidence': conf,
                    'avg_ppm': avg_ppm,
                    'sensor_readings': sensor_readings
                })
                
                print(f"Reading {i+1}: {predicted} (conf: {conf:.2f}, avg PPM: {avg_ppm:.1f})")
                time.sleep(2)
            
            # Analyze results
            avg_confidence = sum([r['confidence'] for r in readings]) / len(readings)
            avg_ppm = sum([r['avg_ppm'] for r in readings]) / len(readings)
            most_predicted = max(set([r['predicted'] for r in readings]), 
                               key=[r['predicted'] for r in readings].count)
            
            test_results[gas_type] = {
                'most_predicted': most_predicted,
                'avg_confidence': avg_confidence,
                'avg_ppm': avg_ppm,
                'success': most_predicted == gas_type or avg_confidence > 0.5
            }
            
            print(f"Result: Most predicted: {most_predicted}, Avg confidence: {avg_confidence:.2f}")
            print(f"Success: {'✅ PASS' if test_results[gas_type]['success'] else '❌ FAIL'}")
        
        return test_results
    
    def test_calibration_status(self):
        """Test sensor calibration status"""
        print("\n⚙️ Testing Calibration Status...")
        
        calibration_ok = True
        
        for sensor_name in ['TGS2600', 'TGS2602', 'TGS2610']:
            config = self.gas_detector.sensor_config[sensor_name]
            R0 = config.get('R0')
            baseline_voltage = config.get('baseline_voltage')
            
            if R0 is not None and R0 > 0:
                print(f"✅ {sensor_name}: R0 = {R0:.1f}Ω")
                if baseline_voltage:
                    print(f"   Baseline voltage: {baseline_voltage:.3f}V")
            else:
                print(f"❌ {sensor_name}: Not calibrated (R0 not set)")
                calibration_ok = False
        
        return calibration_ok
    
    def test_model_status(self):
        """Test machine learning model status"""
        print("\n🤖 Testing ML Model Status...")
        
        if self.gas_detector.is_model_trained:
            print("✅ Machine learning model is loaded and ready")
            
            # Test prediction capability
            try:
                readings = self.gas_detector.read_sensors()
                predicted_gas, confidence = self.gas_detector.predict_gas(readings)
                print(f"✅ Model prediction working: {predicted_gas} (confidence: {confidence:.2f})")
                return True
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                return False
        else:
            print("⚠️  No trained model found")
            print("   Gas detection will use basic methods")
            print("   Train model from main.py for better accuracy")
            return False
    
    def run_full_validation(self):
        """Run complete sensor validation"""
        print("🚀 Starting Full Sensor Validation...")
        print("=" * 50)
        
        # Test 1: Connectivity
        connectivity_ok = self.test_sensor_connectivity()
        
        # Test 2: Calibration status
        calibration_ok = self.test_calibration_status()
        
        # Test 3: Model status
        model_ok = self.test_model_status()
        
        # Test 4: Detection accuracy (if everything else is OK)
        detection_results = None
        if connectivity_ok:
            detection_results = self.test_gas_detection_accuracy()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 SENSOR VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Hardware Connectivity: {'✅ PASS' if connectivity_ok else '❌ FAIL'}")
        print(f"Sensor Calibration: {'✅ PASS' if calibration_ok else '⚠️  NEEDS CALIBRATION'}")
        print(f"ML Model: {'✅ READY' if model_ok else '⚠️  BASIC MODE'}")
        
        if detection_results:
            detection_success = sum([r['success'] for r in detection_results.values()])
            detection_total = len(detection_results)
            print(f"Gas Detection: {detection_success}/{detection_total} ({'✅ PASS' if detection_success >= 2 else '❌ NEEDS IMPROVEMENT'})")
        
        overall_success = connectivity_ok and (calibration_ok or model_ok)
        print(f"\n🎯 OVERALL STATUS: {'✅ READY FOR MAPPING' if overall_success else '❌ NEEDS SETUP'}")
        
        if not overall_success:
            print("\n🔧 RECOMMENDED ACTIONS:")
            if not connectivity_ok:
                print("   1. Check hardware connections and power supply")
            if not calibration_ok:
                print("   2. Run sensor calibration from main.py")
            if not model_ok:
                print("   3. Train ML model from main.py for better accuracy")
        
        # Save results
        self.save_validation_results({
            'timestamp': datetime.now().isoformat(),
            'connectivity': connectivity_ok,
            'calibration': calibration_ok,
            'model': model_ok,
            'detection_results': detection_results,
            'overall_success': overall_success
        })
        
        return overall_success
    
    def save_validation_results(self, results):
        """Save validation results to file"""
        filename = f"logs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {filename}")

if __name__ == "__main__":
    print("🔬 USV Gas Sensor Validation Tool")
    print("Testing your EnhancedDatasheetGasSensorArray system")
    print("This will validate sensor connectivity and gas detection capability.\n")
    
    validator = SensorValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎉 Validation completed successfully!")
        print("Your gas detection system is ready for GPS mapping integration!")
    else:
        print("\n⚠️ System needs setup before mapping:")
        print("1. Fix any hardware issues")
        print("2. Run calibration and/or model training from main.py")
        print("3. Re-run this test to confirm readiness")
