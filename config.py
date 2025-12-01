"""
    Configuration Module for Radiomics Pipeline
"""
import os
from pathlib import Path

class Config:
    """Configuration class for the radiomics analysis pipeline"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    LOGS_DIR = BASE_DIR / "logs"


    # Create a directory that does not exist
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Feature Extraction Settings
    RADIOMICS_SETTINGS = {
        "correctMask": False,
        "minimumROIDimensions": 2,
        "binWidth":25,
        "resamplePixelSpacing": None,   # No resample spacing (Possible Change)
        "interpolator": 'sitkBSpline',
        "padDistance": 5
    }

    IMAGE_TYPES = {
        'Original': True,
        'Wavelet': True,
        'LoG': True,
        'Square': True,
        'SquareRoot': True,
        'Logarithm':True,
        'Exponential':True,
        'Gradient': True,
        'LBP2D':True,
        'LBP3D': True
    }

    # Preprocessing Settings
    PREPROCESSING = {
        'validate_nifti':True,
        'fix_affine':True,
        'intensity_normalization':True,
        'check_spacing': True,
        'check_orientation': True
    }

    # Feature Selection Settings
    FEATURE_SELECTION = {
        # Correlation-base filtering
        'correlation_threshold': 0.8,
        'correlation_method': 'pearson',    # perason, spearman

        # Univariant Section
        'univariant_method': 'mutual_info', # mutual_info, f_classif, chi2
        'univariante_k': 50,  # Number of features to select

        # Multivariante Section
        'use_rfe': True,        # Recursive Feature Elimination
        'rfe_n_feature': 20,
        'rfe_step': 5,

        'use_boruta': True,     # Boruta algorithm
        'boruta_max_iter': 100,
        'boruta_perc': 90,

        # Variance threshold
        'variance_threshold': 0.01,

        # Final Selection
        'final_k_features': 15
    }

    MODELS = {
        'random_forest':{
            'enabled': True,
            'param_grid':{
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini','entropy']
            }
        },
        'xgboost': {
            'enabled': True,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 5],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        },
        'svm': {
            'enabled': True,
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'gradient_boosting': {
            'enabled': True,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'logistic_regression': {
        'enabled': True,
        'param_grid': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 200, 500],
            'l1_ratio': [None, 0.3, 0.5, 0.7]  # Only used with 'elasticnet'
        }
    }
    }

    # Training Setting
    TRAINING = {
        'test_size': 0.15,
        'validation_size': 0.15,  # From training set
        'random_state': 42,
        'cv_folds': 5,
        'cv_method': 'stratified',  # 'stratified' or 'standard'
        'scoring': 'roc_auc',  # 'accuracy', 'roc_auc', 'f1'
        'n_jobs': -1  # Use all cores
    }
    
    SCALING = {
        'method': 'minmax',       # minmax, standard, robust
        'feature_range': (0,1)
    }

    # Logging
    LOGGING = {
        'level': 'INFO',       # DEBUG, INFO, WARNING, ERROR
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': LOGS_DIR / "pipeline.log" 
    }

    # Clinical labels
    LABELS = {
        'AD': 1,
        'CN': 0,
        'control': 0
    }

    # Expected folder structure
    EXPECTED_STRUCTURE = {
        'image_folder': 'T1',
        'image_file': 'T1.nii.gz',
        'label_folder': 'label',
        'label_file': 'label.nii.gz'
    }


    @classmethod
    def update_data_path(cls, new_path):
        """Update the data directory path"""
        cls.DATA_DIR = Path(new_path)
        if not cls.DATA_DIR.exists():
            raise ValueError(f'Data directory does not exsit: {new_path}')
        
    @classmethod
    def get_model_config(cls, model_name):
        """"Get configuration for a  specific model"""
        if model_name not in cls.MODELS:
            raise ValueError(f" Unknown model: {model_name}")
        return cls.MODELS[model_name]
    
    @classmethod
    def get_enabled_models(cls):
        """Get list of enable models"""
        return [name for name, config in cls.MODELS.items() if config['enabled']]