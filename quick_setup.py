#!/usr/bin/env python3
"""
Quick Setup Script untuk USV Gas Mapping
Setup cepat dalam 3-4 hari - streamlined version
"""

import os
import subprocess
import sys
import json
from datetime import datetime

class QuickSetup:
    def __init__(self):
        self.setup_steps = [
            "🔧 Environment Setup",
            "📦 Dependencies Installation", 
            "🧪 Hardware Validation",
            "🗺️ Mapping System Test",
            "🚀 Integration Test"
        ]
        
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            print("❌ Python 3.7+ required")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 Installing dependencies...")
        
        packages = [
            "pymavlink==2.4.37",
            "dronekit==2.9.2", 
            "folium==0.14.0",
            "plotly==5.17.0",
            "pandas==2.1.4",
            "numpy==1.24.3",
            "scikit-learn",
            "joblib",
            "adafruit-circuitpython-ads1x15"
        ]
        
        try:
            for package in packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✅ All packages installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def create_directories(self):
        """Create required directories"""
        print("\n📁 Creating project directories...")
        
        dirs = ["logs", "maps", "data", "config", "models", "calibration"]
        
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ {dir_name}/")
        
        return True
    
    def check_main_py(self):
        """Check if main.py exists and is valid"""
        print("\n🔍 Checking main.py...")
        
        if not os.path.exists("main.py"):
            print("❌ main.py not found!")
            return False
        
        try:
            # Try to import to check validity
            spec = __import__("main")
            if hasattr(spec, 'EnhancedDatasheetGasSensorArray'):
                print("✅ main.py valid with EnhancedDatasheetGasSensorArray")
                return True
            else:
                print("❌ EnhancedDatasheetGasSensorArray not found in main.py")
                return False
        except Exception as e:
            print(f"❌ main.py import error: {e}")
            return False
    
    def check_hardware_readiness(self):
        """Check hardware readiness"""
        print("\n⚡ Checking hardware readiness...")
        
        # Check if this is running on Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo:
                    print("✅ Running on Raspberry Pi")
                    pi_ready = True
                else:
                    print("⚠️  Not running on Raspberry Pi")
                    pi_ready = False
        except:
            print("⚠️  Cannot determine hardware platform")
            pi_ready = False
        
        # Check I2C
        i2c_ready = os.path.exists('/dev/i2c-1')
        if i2c_ready:
            print("✅ I2C interface available")
        else:
            print("⚠️  I2C interface not found - check sudo raspi-config")
        
        # Check GPIO access
        gpio_ready = os.path.exists('/sys/class/gpio')
        if gpio_ready:
            print("✅ GPIO interface available")
        else:
            print("⚠️  GPIO interface not found")
        
        return pi_ready or (i2c_ready and gpio_ready)
    
    def create_quick_test_script(self):
        """Create quick test script"""
        print("\n📝 Creating quick test script...")
        
        test_script = '''#!/usr/bin/env python3
"""Quick integration test"""

def quick_test():
    print("🚀 Quick Integration Test")
    
    # Test 1: Import main.py
    try:
        from main import EnhancedDatasheetGasSensorArray
        print("✅ main.py import: OK")
    except Exception as e:
        print(f"❌ main.py import: {e}")
        return False
    
    # Test 2: Import usv_gas_mapping.py  
    try:
        from usv_gas_mapping import USVGasMapper
        print("✅ usv_gas_mapping.py import: OK")
    except Exception as e:
        print(f"❌ usv_gas_mapping.py import: {e}")
        return False
    
    # Test 3: Initialize systems
    try:
        gas_detector = EnhancedDatasheetGasSensorArray()
        print("✅ Gas detection system: OK")
    except Exception as e:
        print(f"⚠️  Gas detection system: {e}")
        print("   This may be normal if sensors not connected")
    
    try:
        mapper = USVGasMapper()
        print("✅ Mapping system: OK")
    except Exception as e:
        print(f"❌ Mapping system: {e}")
        return False
    
    print("\\n🎉 Quick test completed!")
    print("System ready for full testing")
    return True

if __name__ == "__main__":
    quick_test()
'''
        
        with open("quick_test.py", "w") as f:
            f.write(test_script)
        
        print("✅ quick_test.py created")
        return True
    
    def create_quick_config(self):
        """Create quick configuration"""
        print("\n⚙️ Creating configuration...")
        
        config = {
            "mavlink_connection": "udp:0.0.0.0:14550",
            "update_interval": 2.0,
            "gps_timeout": 10.0,
            "gas_sample_rate": 1.0,
            "map_update_interval": 5.0,
            "log_level": "INFO",
            "output_dir": "maps",
            "data_dir": "data",
            "quick_setup": True,
            "setup_date": datetime.now().isoformat()
        }
        
        with open("config/mapping_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration created")
        return True
    
    def run_quick_setup(self):
        """Run complete quick setup"""
        print("🚀 USV Gas Mapping - Quick Setup (3-4 Days)")
        print("=" * 60)
        print("This will setup everything for rapid implementation\n")
        
        setup_success = True
        
        # Step 1: Environment check
        print("STEP 1: Environment Check")
        if not self.check_python_version():
            setup_success = False
        
        # Step 2: Dependencies
        print("\nSTEP 2: Dependencies")
        if not self.install_dependencies():
            setup_success = False
        
        # Step 3: Directories
        print("\nSTEP 3: Project Structure")
        self.create_directories()
        
        # Step 4: Check main.py
        print("\nSTEP 4: main.py Validation")
        if not self.check_main_py():
            setup_success = False
        
        # Step 5: Hardware check
        print("\nSTEP 5: Hardware Check")
        hardware_ok = self.check_hardware_readiness()
        
        # Step 6: Configuration
        print("\nSTEP 6: Configuration")
        self.create_quick_config()
        
        # Step 7: Test script
        print("\nSTEP 7: Test Scripts")
        self.create_quick_test_script()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 QUICK SETUP SUMMARY")
        print("=" * 60)
        
        if setup_success:
            print("✅ Core setup completed successfully!")
        else:
            print("❌ Some setup issues found")
        
        if hardware_ok:
            print("✅ Hardware ready for sensor testing")
        else:
            print("⚠️  Hardware may need configuration")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Copy usv_gas_mapping.py to current directory")
        print("2. Copy test_sensors.py to current directory") 
        print("3. Copy test_mavlink.py to current directory")
        print("4. Run: python3 quick_test.py")
        print("5. Follow the 3-day implementation plan")
        
        return setup_success and hardware_ok

if __name__ == "__main__":
    setup = QuickSetup()
    setup.run_quick_setup()
