# Panduan Modifikasi File gas_sensor_fix.py

## 🎯 LANGKAH 1: Modifikasi Method collect_training_data()

**CARI baris ini di file Anda (sekitar baris 447):**
```python
def collect_training_data(self, gas_type, duration=60, samples_per_second=1):
    """Collect training data with extended range"""
    # Ensure we're in extended mode for training
    self.set_sensor_mode('extended')
    
    self.logger.info(f"Collecting training data for {gas_type} in EXTENDED mode")
    self.logger.info(f"Duration: {duration}s, Sampling rate: {samples_per_second} Hz")
    
    valid_gases = ['alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']  # ← UBAH BARIS INI
```

**GANTI baris `valid_gases = [...]` dengan:**
```python
    valid_gases = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']  # TAMBAH 'normal'
```

**CARI baris ini (masih di method yang sama, sekitar baris 454):**
```python
    if gas_type not in valid_gases:
        self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
        return None
    
    input(f"Prepare to spray {gas_type}. Press Enter to start...")  # ← GANTI BAGIAN INI
```

**GANTI bagian input tersebut dengan:**
```python
    if gas_type not in valid_gases:
        self.logger.error(f"Invalid gas type. Valid options: {valid_gases}")
        return None
    
    # Special instructions untuk normal data collection
    if gas_type == 'normal':
        input(f"Ensure sensors are in CLEAN AIR (no gas). Press Enter to start...")
        self.logger.info("Collecting NORMAL/CLEAN AIR data...")
    else:
        input(f"Prepare to spray {gas_type}. Press Enter to start...")
```

---

## 🎯 LANGKAH 2: Modifikasi Function main()

**CARI bagian menu di function main() (sekitar baris 700):**
```python
        print("1. Calibrate sensors")
        print("2. Collect training data (Extended mode - unlimited range)")
        print("3. Train machine learning model")
        print("4. Start monitoring - Datasheet mode (accurate detection)")
        print("5. Start monitoring - Extended mode (full range)")
        print("6. Test single reading")
        print("7. Switch sensor mode (Extended ↔ Datasheet)")
        print("8. View current sensor modes")
        print("9. Exit")  # ← UBAH BAGIAN INI
        print("-"*60)
```

**GANTI dengan:**
```python
        print("1. Calibrate sensors")
        print("2. Collect training data (Extended mode - unlimited range)")
        print("3. Train machine learning model")
        print("4. Start monitoring - Datasheet mode (accurate detection)")
        print("5. Start monitoring - Extended mode (full range)")
        print("6. Test single reading")
        print("7. Switch sensor mode (Extended ↔ Datasheet)")
        print("8. View current sensor modes")
        print("9. Collect NORMAL/Clean Air data")  # TAMBAH BARIS INI
        print("10. Exit")  # UBAH dari 9 ke 10
        print("-"*60)
```

**CARI baris input choice (sekitar baris 712):**
```python
        choice = input("Select option (1-9): ").strip()  # ← UBAH INI
```

**GANTI dengan:**
```python
        choice = input("Select option (1-10): ").strip()  # UBAH dari 1-9 ke 1-10
```

**CARI bagian elif choice == '2' (sekitar baris 720):**
```python
            elif choice == '2':
                gas_types = ['alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']  # ← UBAH INI
                print("Available gas types:", ', '.join(gas_types))
                gas_type = input("Enter gas type: ").strip().lower()
```

**GANTI dengan:**
```python
            elif choice == '2':
                gas_types = ['normal', 'alcohol', 'pertalite', 'pertamax', 'dexlite', 'biosolar']  # TAMBAH 'normal'
                print("Available gas types:", ', '.join(gas_types))
                print("⚠️  IMPORTANT: Collect 'normal' data first for baseline!")
                gas_type = input("Enter gas type: ").strip().lower()
```

**CARI bagian elif choice == '9' (ini adalah Exit yang lama):**
```python
            elif choice == '9':
                print("Exiting system...")
                break
```

**GANTI dengan (tambah option 9 baru dan ubah exit ke 10):**
```python
            elif choice == '9':  # NEW: Shortcut untuk collect normal data
                print("🌬️  Collecting NORMAL/Clean Air baseline data")
                print("Ensure sensors are in clean environment (no gas sources)")
                duration = int(input("Collection duration (seconds, default 120): ") or 120)
                gas_sensor.collect_training_data('normal', duration)
                
            elif choice == '10':  # Updated exit option
                print("Exiting system...")
                break
```

---

## 🎯 LANGKAH 3: Tambahkan Method Baru (Optional - Enhanced Prediction)

**CARI method predict_gas() di file Anda (sekitar baris 600), SETELAH method tersebut, TAMBAHKAN method baru:**

```python
    def enhanced_predict_gas(self, readings, confidence_threshold=0.6):
        """Enhanced prediction dengan confidence threshold untuk kondisi normal"""
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
                    feature_vector.append(0.0)
        
        features = np.array([feature_vector])
        features = np.nan_to_num(features, nan=0.0)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        # Enhanced logic: Jika confidence rendah dan prediction bukan 'normal',
        # kemungkinan kondisi tidak dikenali (might be normal)
        if confidence < confidence_threshold and prediction != 'normal':
            return "Uncertain - Possibly Normal", confidence
        
        return prediction, confidence
```

---

## 📝 RINGKASAN PERUBAHAN

### File yang dimodifikasi: `gas_sensor_fix.py`

### Baris yang diubah:
1. **~Line 452**: Tambah `'normal'` ke `valid_gases`
2. **~Line 458**: Tambah logic untuk instruksi normal data collection  
3. **~Line 708**: Tambah menu option 9 untuk normal data
4. **~Line 712**: Ubah input dari `(1-9)` ke `(1-10)`
5. **~Line 720**: Tambah `'normal'` ke gas_types dan warning
6. **~Line 785**: Tambah case `elif choice == '9'` untuk normal collection
7. **~Line 791**: Ubah exit dari choice 9 ke choice 10
8. **~Line 650**: (Optional) Tambah method `enhanced_predict_gas()`

### Total modifikasi: **7 tempat** di file yang sama

---

## ✅ CARA MUDAH IMPLEMENTASI

1. **Buka file** `gas_sensor_fix.py` dengan text editor
2. **Gunakan Ctrl+F** untuk mencari teks yang saya sebutkan di atas
3. **Ganti/tambah** sesuai instruksi
4. **Save** file
5. **Test** dengan menjalankan: `python gas_sensor_fix.py`

### 🚨 TIPS:
- **Backup** file original sebelum edit
- **Edit satu per satu** jangan sekaligus
- **Test** setelah setiap perubahan untuk memastikan tidak error
