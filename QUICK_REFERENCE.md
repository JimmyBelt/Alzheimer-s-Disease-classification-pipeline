# Quick Reference Guide

One-page cheat sheet for the Radiomics Pipeline.

## 🚀 Quick Start (3 Commands)

```bash
# 1. Train pipeline
python main.py --mode train --data /path/to/dataset --quick

# 2. Predict on new image
python main.py --mode predict --pipeline models/pipeline.pkl --image img.nii.gz --mask mask.nii.gz

# 3. See results
cat results/training_report.txt
```

## 📁 Expected Data Structure

```
dataset/
├── AD/
│   └── subject_*/T1/T1.nii.gz + label/label.nii.gz
└── CN/
    └── subject_*/T1/T1.nii.gz + label/label.nii.gz
```

## 🎯 Common Commands

### Training
```bash
# Full training (all models)
python main.py --mode train --data dataset/

# Quick training (faster, for testing)
python main.py --mode train --data dataset/ --quick

# Train specific models only
python main.py --mode train --data dataset/ --models random_forest xgboost

# Custom output directory
python main.py --mode train --data dataset/ --output ./my_results
```

### Prediction
```bash
# Single image
python main.py --mode predict --pipeline models/pipeline.pkl --image img.nii.gz --mask mask.nii.gz

# Batch prediction
python predict.py --pipeline models/pipeline.pkl --directory test_dataset/ --visualize
```

### Evaluation
```bash
# Evaluate on test set
python main.py --mode evaluate --pipeline models/pipeline.pkl --data test_dataset/
```

### Feature Extraction Only
```bash
# Extract features without training
python main.py --mode extract --data dataset/ --output features/
```

## ⚙️ Configuration Quick Edit

Edit `config.py`:

```python
# Feature selection
FEATURE_SELECTION = {
    'correlation_threshold': 0.8,      # Lower = more features
    'final_k_features': 15             # Final number of features
}

# Training
TRAINING = {
    'cv_folds': 5,                     # Cross-validation folds
    'scoring': 'roc_auc'               # 'accuracy', 'f1', 'roc_auc'
}

# Enable/disable models
MODELS['xgboost']['enabled'] = True    # Enable XGBoost
MODELS['svm']['enabled'] = False       # Disable SVM
```

## 🐍 Python API Quick Examples

### Example 1: Complete Training
```python
from config import Config
from pipeline import RadiomicsPipeline
from pathlib import Path

config = Config()
pipeline = RadiomicsPipeline(config)
results = pipeline.run_complete_pipeline(Path('dataset/'))
pipeline.save_pipeline()
```

### Example 2: Predict
```python
from predict import RadiomicsPredictor
from pathlib import Path

predictor = RadiomicsPredictor(Path('models/pipeline.pkl'))
result = predictor.predict_single(
    Path('image.nii.gz'),
    Path('mask.nii.gz')
)
print(f"{result['prediction_label']}: {result['confidence']:.2%}")
```

### Example 3: Feature Extraction
```python
from config import Config
from feature_extraction import RadiomicsExtractor

config = Config()
extractor = RadiomicsExtractor(config)
features_df, labels, info = extractor.extract_from_directory(Path('dataset/'))
features_df.to_csv('features.csv')
```

## 📊 Output Files Reference

```
results/
├── raw_features.csv              # All extracted features
├── model_comparison.csv          # Performance comparison
├── training_report.txt           # Detailed report
├── feature_selection_summary.png # Selection visualization
├── roc_curves.png               # ROC curves
├── confusion_matrix.png         # Confusion matrix
└── feature_importances.png      # Feature importance

models/
├── pipeline.pkl                 # Complete pipeline
└── pipeline_metadata.json       # Pipeline info

logs/
└── pipeline.log                 # Execution logs
```

## 🔍 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| Import error | `pip install -r requirements.txt` |
| Memory error | Use `--quick` flag or reduce dataset size |
| NIfTI error | Set `config.PREPROCESSING['fix_affine'] = True` |
| Boruta not found | `pip install boruta` or disable in config |
| XGBoost error | `pip install xgboost` or disable in config |
| Slow training | Use `--quick` or reduce `cv_folds` |

## 📈 Performance Tuning

### For Speed (Quick Training)
```python
config.FEATURE_SELECTION['final_k_features'] = 10
config.TRAINING['cv_folds'] = 3
config.MODELS['svm']['enabled'] = False
config.FEATURE_SELECTION['use_boruta'] = False
```

### For Accuracy (Best Performance)
```python
config.FEATURE_SELECTION['final_k_features'] = 20
config.TRAINING['cv_folds'] = 10
config.FEATURE_SELECTION['use_boruta'] = True
# Enable all models
```

## 🎨 Visualization Quick Access

```python
from utils.metrics import *

# Feature importances
plot_importances(model.feature_importances_, feature_names)

# Complete analysis
analyze_train_test_performance(model, X_train, X_test, y_train, y_test)

# Model comparison
compare_models_performance(models_results)
```

## 🔧 Custom Model Addition

```python
# 1. Add to config.py
MODELS = {
    'my_model': {
        'enabled': True,
        'param_grid': {'param1': [1, 2, 3]}
    }
}

# 2. Add to model_training.py
def get_model_instance(self, model_name):
    if model_name == 'my_model':
        return MyModel()
```

## 📝 Logging Levels

```python
# Set in config.py or programmatically
config.LOGGING['level'] = 'DEBUG'  # Show everything
config.LOGGING['level'] = 'INFO'   # Normal (default)
config.LOGGING['level'] = 'WARNING'  # Only warnings/errors
config.LOGGING['level'] = 'ERROR'  # Only errors
```

## 🎯 Feature Selection Methods

```python
# In config.py
FEATURE_SELECTION = {
    'correlation_threshold': 0.8,     # Remove correlated
    'univariate_method': 'mutual_info',  # or 'f_classif', 'chi2'
    'univariate_k': 50,               # Top K features
    'use_rfe': True,                  # Recursive elimination
    'rfe_n_features': 20,             # RFE target
    'use_boruta': True,               # Boruta algorithm
    'final_k_features': 15            # Final selection
}
```

## 🏥 Clinical Interpretation

```python
result = predictor.predict_single(image, mask)

# Confidence levels
if result['confidence'] > 0.8:
    print("High confidence - reliable prediction")
elif result['confidence'] > 0.6:
    print("Medium confidence - review recommended")
else:
    print("Low confidence - additional testing needed")

# Class probabilities
print(f"P(Alzheimer's): {result['probability_AD']:.1%}")
print(f"P(Normal): {result['probability_CN']:.1%}")
```

## 🔄 Common Workflows

### Workflow 1: Initial Setup
```bash
git clone <repo>
cd radiomics_pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python test_imports.py
```

### Workflow 2: Training
```bash
python main.py --mode extract --data dataset/  # Extract features
python main.py --mode train --data dataset/ --quick  # Quick training
# Review results, then:
python main.py --mode train --data dataset/  # Full training
```

### Workflow 3: Deployment
```bash
# Train on full dataset
python main.py --mode train --data full_dataset/

# Test on holdout set
python main.py --mode evaluate --pipeline models/pipeline.pkl --data holdout/

# Deploy for predictions
python main.py --mode predict --pipeline models/pipeline.pkl --image new.nii.gz --mask new_mask.nii.gz
```

## 📦 Module Reference

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `config.py` | Configuration | `Config` |
| `preprocessing.py` | Image preprocessing | `ImagePreprocessor` |
| `feature_extraction.py` | Feature extraction | `RadiomicsExtractor` |
| `feature_selection.py` | Feature selection | `FeatureSelector` |
| `model_training.py` | Model training | `ModelTrainer` |
| `pipeline.py` | Complete pipeline | `RadiomicsPipeline` |
| `predict.py` | Prediction | `RadiomicsPredictor` |
| `main.py` | CLI interface | `main()` |
| `utils/metrics.py` | Evaluation | Various plot functions |

## 🎓 Learning Resources

1. **Start here**: `SETUP_GUIDE.md`
2. **Usage**: `README.md`
3. **Examples**: `example_usage.py`
4. **What's new**: `IMPROVEMENTS.md`
5. **This guide**: `QUICK_REFERENCE.md`

## 💡 Tips & Tricks

### Tip 1: Test with Small Dataset First
```bash
# Create small test set (10 subjects)
mkdir test_small
cp -r dataset/AD/*[1-5] test_small/AD/
cp -r dataset/CN/*[1-5] test_small/CN/

# Quick test
python main.py --mode train --data test_small/ --quick
```

### Tip 2: Monitor Progress
```bash
# Watch log file in real-time
tail -f logs/pipeline.log
```

### Tip 3: Save Intermediate Results
```python
pipeline = RadiomicsPipeline(config)
pipeline.extract_features(dataset_path)
pipeline.features_df.to_csv('backup_features.csv')  # Save features
# Continue later...
```

### Tip 4: Compare Configurations
```bash
# Train with different configs
python main.py --mode train --data dataset/ --output results_config1/
# Edit config.py
python main.py --mode train --data dataset/ --output results_config2/
# Compare results_config1/ and results_config2/
```

## 🆘 Getting Help

```bash
# CLI help
python main.py --help
python predict.py --help

# Check logs
cat logs/pipeline.log

# Test installation
python test_imports.py
python test_pipeline.py
```

## ✅ Pre-flight Checklist

Before training:
- [ ] Data structure correct
- [ ] Paths updated in config
- [ ] Requirements installed
- [ ] Test imports pass
- [ ] Validation script run

Before prediction:
- [ ] Pipeline trained and saved
- [ ] New images in correct format
- [ ] Masks available

Before deployment:
- [ ] Evaluated on test set
- [ ] Performance acceptable
- [ ] Documentation updated
- [ ] Error handling tested

---

**Quick Help:** For detailed information, see `README.md` and `SETUP_GUIDE.md`