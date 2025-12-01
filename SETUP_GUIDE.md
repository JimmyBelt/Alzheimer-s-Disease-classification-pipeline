# Setup Guide - Radiomics Pipeline

Complete installation and setup instructions for the Radiomics Pipeline for Alzheimer's Disease classification.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Testing Installation](#testing-installation)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 10 GB free space for code and models
- **GPU**: Not required (but can speed up XGBoost)

### Recommended for Large Datasets
- **RAM**: 32 GB
- **CPU**: 8+ cores
- **Storage**: SSD with 50+ GB free space

## 🚀 Installation Steps

### Step 1: Install Python

**Windows:**
```bash
# Download Python from python.org
# Make sure to check "Add Python to PATH" during installation
python --version  # Should be 3.8+
```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (using Homebrew)
brew install python3

# Verify installation
python3 --version
```

### Step 2: Create Project Directory

```bash
# Create and navigate to project directory
mkdir radiomics_pipeline
cd radiomics_pipeline
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Your prompt should now show (venv)
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scipy>=1.7.0

# Install medical imaging libraries
pip install SimpleITK>=2.1.0
pip install nibabel>=3.2.0
pip install pyradiomics>=3.0.1

# Install machine learning libraries
pip install scikit-learn>=1.0.0
pip install xgboost>=1.5.0

# Install visualization
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Install utilities
pip install tqdm>=4.62.0
pip install joblib>=1.1.0
pip install six>=1.16.0

# Optional: Install Boruta (advanced feature selection)
pip install boruta

# Or install all at once from requirements.txt
pip install -r requirements.txt
```

### Step 5: Download/Copy Pipeline Code

Copy all the pipeline Python files to your project directory:
- `config.py`
- `preprocessing.py`
- `feature_extraction.py`
- `feature_selection.py`
- `model_training.py`
- `pipeline.py`
- `predict.py`
- `main.py`
- `example_usage.py`

Create the `utils` directory and add:
- `utils/metrics.py`

### Step 6: Create Directory Structure

```bash
# Create necessary directories
mkdir data
mkdir models
mkdir results
mkdir logs

# Your structure should look like:
# radiomics_pipeline/
# ├── venv/
# ├── data/
# ├── models/
# ├── results/
# ├── logs/
# ├── utils/
# │   └── metrics.py
# ├── config.py
# ├── preprocessing.py
# ├── feature_extraction.py
# ├── feature_selection.py
# ├── model_training.py
# ├── pipeline.py
# ├── predict.py
# ├── main.py
# ├── example_usage.py
# └── requirements.txt
```

## 📁 Data Preparation

### Step 1: Organize Your Dataset

Your medical imaging data should follow this structure:

```
data/
├── AD/                          # Alzheimer's Disease subjects
│   ├── subject_001/
│   │   ├── T1/
│   │   │   └── T1.nii.gz       # T1-weighted MRI
│   │   └── label/
│   │       └── label.nii.gz    # Segmentation mask
│   ├── subject_002/
│   │   ├── T1/
│   │   │   └── T1.nii.gz
│   │   └── label/
│   │       └── label.nii.gz
│   └── ...
└── CN/                          # Cognitively Normal subjects
    ├── subject_101/
    │   ├── T1/
    │   │   └── T1.nii.gz
    │   └── label/
    │       └── label.nii.gz
    ├── subject_102/
    │   ├── T1/
    │   │   └── T1.nii.gz
    │   └── label/
    │       └── label.nii.gz
    └── ...
```

### Step 2: Validate Your Data

Create a simple validation script:

```python
# validate_data.py
from pathlib import Path
import nibabel as nib

def validate_dataset(dataset_path):
    """Validate dataset structure"""
    
    dataset_path = Path(dataset_path)
    
    issues = []
    valid_subjects = 0
    
    for diagnosis in ['AD', 'CN']:
        diag_path = dataset_path / diagnosis
        
        if not diag_path.exists():
            issues.append(f"Missing {diagnosis} directory")
            continue
        
        for subject_dir in diag_path.iterdir():
            if not subject_dir.is_dir():
                continue
            
            # Check image
            image_path = subject_dir / 'T1' / 'T1.nii.gz'
            if not image_path.exists():
                issues.append(f"Missing image: {subject_dir.name}")
                continue
            
            # Check mask
            mask_path = subject_dir / 'label' / 'label.nii.gz'
            if not mask_path.exists():
                issues.append(f"Missing mask: {subject_dir.name}")
                continue
            
            # Try loading
            try:
                img = nib.load(str(image_path))
                mask = nib.load(str(mask_path))
                
                if img.shape != mask.shape:
                    issues.append(f"Shape mismatch: {subject_dir.name}")
                    continue
                
                valid_subjects += 1
                
            except Exception as e:
                issues.append(f"Error loading {subject_dir.name}: {e}")
    
    print(f"\nValidation Results:")
    print(f"  Valid subjects: {valid_subjects}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    return valid_subjects, issues

if __name__ == '__main__':
    validate_dataset('data/')
```

Run validation:
```bash
python validate_data.py
```

## ⚙️ Configuration

### Basic Configuration

Edit `config.py` to set your paths:

```python
# In config.py, update DATA_DIR if needed
class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"  # Your dataset location
    # ... rest of config
```

### Quick Training Configuration

For initial testing, use quick mode:

```python
# In config.py or programmatically
config = Config()

# Reduce features for faster training
config.FEATURE_SELECTION['final_k_features'] = 10

# Reduce CV folds
config.TRAINING['cv_folds'] = 3

# Disable some models
config.MODELS['svm']['enabled'] = False
config.MODELS['gradient_boosting']['enabled'] = False
```

### Production Configuration

For full production training:

```python
config = Config()

# Use more features
config.FEATURE_SELECTION['final_k_features'] = 20

# More CV folds
config.TRAINING['cv_folds'] = 5

# Enable all models
for model_name in config.MODELS:
    config.MODELS[model_name]['enabled'] = True

# Enable Boruta
config.FEATURE_SELECTION['use_boruta'] = True
```

## 🧪 Testing Installation

### Test 1: Import All Modules

```python
# test_imports.py
print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy")
    
    import pandas as pd
    print("✓ Pandas")
    
    import SimpleITK as sitk
    print("✓ SimpleITK")
    
    import nibabel as nib
    print("✓ Nibabel")
    
    from radiomics import featureextractor
    print("✓ PyRadiomics")
    
    from sklearn.ensemble import RandomForestClassifier
    print("✓ Scikit-learn")
    
    import xgboost as xgb
    print("✓ XGBoost")
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib")
    
    import seaborn as sns
    print("✓ Seaborn")
    
    print("\n✓ All imports successful!")
    
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
```

Run test:
```bash
python test_imports.py
```

### Test 2: Test Pipeline Initialization

```python
# test_pipeline.py
from config import Config
from pipeline import RadiomicsPipeline

try:
    config = Config()
    pipeline = RadiomicsPipeline(config)
    print("✓ Pipeline initialized successfully!")
    print(f"  Config loaded: {config.TRAINING['cv_folds']} CV folds")
    print(f"  Models enabled: {config.get_enabled_models()}")
except Exception as e:
    print(f"✗ Pipeline initialization failed: {e}")
```

Run test:
```bash
python test_pipeline.py
```

### Test 3: Test Feature Extraction (Single Image)

```python
# test_extraction.py
from config import Config
from feature_extraction import RadiomicsExtractor
from pathlib import Path

config = Config()
extractor = RadiomicsExtractor(config)

# Update these paths to your test image
image_path = Path('data/CN/subject_001/T1/T1.nii.gz')
mask_path = Path('data/CN/subject_001/label/label.nii.gz')

if image_path.exists() and mask_path.exists():
    try:
        features = extractor.extract_from_paths(image_path, mask_path)
        print(f"✓ Feature extraction successful!")
        print(f"  Features extracted: {len(features)}")
        print(f"  First 5 features: {list(features.keys())[:5]}")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
else:
    print("✗ Test images not found. Update paths in test_extraction.py")
```

Run test:
```bash
python test_extraction.py
```

## 🔧 Troubleshooting

### Issue 1: PyRadiomics Installation Fails

**Solution:**
```bash
# Install build tools first
# Windows:
pip install wheel

# Linux:
sudo apt-get install python3-dev

# Then retry:
pip install pyradiomics
```

### Issue 2: SimpleITK Import Error

**Solution:**
```bash
pip uninstall SimpleITK
pip install SimpleITK --upgrade
```

### Issue 3: Memory Error During Training

**Solution:**
- Reduce number of subjects
- Use quick mode configuration
- Process in smaller batches
- Close other applications

### Issue 4: "No module named 'boruta'"

**Solution:**
```bash
# Boruta is optional
pip install boruta

# OR disable in config
config.FEATURE_SELECTION['use_boruta'] = False
```

### Issue 5: XGBoost Not Found

**Solution:**
```bash
pip install xgboost

# OR disable in config
config.MODELS['xgboost']['enabled'] = False
```

### Issue 6: NIfTI Loading Errors

**Solution:**
- Enable automatic fixing in config:
```python
config.PREPROCESSING['fix_affine'] = True
config.PREPROCESSING['validate_nifti'] = True
```

## 📚 Next Steps

After successful installation:

1. **Test with Small Dataset**
   ```bash
   python main.py --mode extract --data data/ --quick
   ```

2. **Run Complete Training**
   ```bash
   python main.py --mode train --data data/ --quick
   ```

3. **Test Prediction**
   ```bash
   python main.py --mode predict \
       --pipeline models/pipeline.pkl \
       --image path/to/test/image.nii.gz \
       --mask path/to/test/mask.nii.gz
   ```

4. **Explore Examples**
   ```bash
   python example_usage.py
   ```

5. **Read Documentation**
   - See README.md for complete usage guide
   - Check API documentation in each module

## 📞 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages in `logs/pipeline.log`
3. Verify your data structure matches expected format
4. Test with a small subset of data first
5. Check Python and package versions

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (no import errors)
- [ ] Directory structure created
- [ ] Pipeline code copied to project
- [ ] Dataset organized in correct structure
- [ ] Data validation passed
- [ ] Import test passed
- [ ] Pipeline initialization test passed
- [ ] Feature extraction test passed
- [ ] Configuration customized for your needs

If all checkboxes are complete, you're ready to start using the pipeline! 🎉

---

**Need more help?** Check README.md for detailed usage instructions.