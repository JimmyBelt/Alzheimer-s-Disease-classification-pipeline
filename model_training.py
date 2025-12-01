""" 
    Model training module with multiple classifiers and typerparameter optimization
"""
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABE = True
except ImportError:
    XGBOOST_AVAILABE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)


logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate multiple classification models"""

    def __init__(self, config):
        self.config = config
        self.training_config = config.TRAINING
        self.models_config = config.MODELS
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_history = {}

    def get_model_instance(self, model_name: str):
        """Get a fresh instance of specific model"""

        random_state = self.training_config['random_state']

        if model_name == 'random_forest':
            return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        elif model_name == 'xgboost':
            if not XGBOOST_AVAILABE:
                raise ValueError("XGBoost is not installed")
            return xgb.XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        elif model_name == 'svm':
            return SVC(random_state=random_state, probability=True)
        
        elif model_name == 'logistic_regression':
            return LogisticRegression(
                random_state=random_state,
                max_iter=1e+3,
                n_jobs=-1
            )
        
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=random_state)
        
        else: 
            raise ValueError(f"Unknown model: {model_name}")
        
    def train_single_model(self, model_name:str, X_train: np.ndarray,
                           y_train: np.ndarray) -> GridSearchCV:
        """ 
        Train a single model with hyperparameter optimization
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Fitted GridSearchCV object
        """

        logger.info(f"Training {model_name} ...")

        # Get model config
        model_config = self.models_config[model_name]

        if not model_config['enabled']:
            logger.info(f"Skipping {model_name} (disable)")
            return None
        
        # Get model istance
        model = self.get_model_instance(model_name=model_name)

        # Setup cross-validation
        if self.training_config['cv_method'] == 'stratified':
            cv = StratifiedKFold(
                n_splits=self.training_config['cv_folds'],
                shuffle=True,
                random_state=self.training_config['random_state']
            )

        else:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=self.training_config['cv_folds'],
                shuffle=True,
                random_state=self.training_config['random_state']
            )

        # Grid Search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=model_config['param_grid'],
            cv=cv,
            scoring=self.training_config['scoring'],
            n_jobs=self.training_config['n_jobs'],
            verbose=0,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"{model_name} - Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"{model_name} - Best params: {grid_search.best_params_}")

        # Store results
        self.trained_models[model_name] = grid_search
        self.training_history[model_name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_params': pd.DataFrame(grid_search.cv_results_)
        }

        return grid_search
        
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """ 
        Train all enable models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """

        enabled_models = self.config.get_enabled_models()
        logger.info(f"Training {len(enabled_models)} models: {enabled_models}")
        
        for model_name in enabled_models:
            try:
                self.train_single_model(model_name=model_name, X_train=X_train, y_train=y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}; {str(e)}")

        # Select best model
        self.select_best_model()

        return self.trained_models
    
    def select_best_model(self):
        """Select the bet model based on CV parameters"""

        if not self.trained_models:
            logger.warning("No trained models availabe")
            return
        
        best_score = -np.inf
        best_name = None

        for name, model in self.trained_models.items():
            if model is not None and model.best_score_ > best_score:
                best_score = model.best_score_
                best_name = name

        if best_name is not None:
            self.best_model = self.trained_models[best_name]
            self.best_model_name = best_name
            logger.info(f"Best model {best_name} (CV score: {best_score:.4f})")

    def evaluate_model(self, model, X:np.ndarray, y:np.ndarray,
                       set_name: str = 'test') -> Dict:
        """ 
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            set_name: Name of the dataset ('train', 'validation', 'test')
            
        Returns:
            Dictionary of metrics
        """

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model,'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }

        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_prob)

        logger.info(f"\n{set_name.upper()} SET PERFORMANCE:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """ 
        Evaluate all trained models on train and test sets
        
        Args:
            X_train: training features dataset
            y_train: training true labels
            X_test: test features dataset
            y_test: test true labels
             
        Returns:
            DataFrame with comparisson of all models
        """

        results = []

        for name, model in self.trained_models.items():
            if model is None:
                continue

            logger.info(f"\nEvaluation {name} ...")

            train_metrics = self.evaluate_model(model=model,X=X_train, 
                                                y=y_train, set_name='train')
            test_metrics = self.evaluate_model(model=model, X=X_test,
                                               y=y_test, set_name='test')
            
            results.append({
                'Model': name,
                'Train_Accuracy': train_metrics['accuracy'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Train_F1': train_metrics['f1'],
                'Test_F1': test_metrics['f1'],
                'Train_ROC_AUC': train_metrics.get('roc_auc', np.nan),
                'Test_ROC_AUC': test_metrics.get('roc_auc', np.nan),
                'Overfit_Gap': train_metrics['accuracy'] - test_metrics['accuracy']
            })

        results_df = pd.DataFrame(data=results).sort_values('Test_ROC_AUC', ascending=False)
        return results_df
        
    def plot_confusion_matrix(self, model, X: np.ndarray, y: np.ndarray,
                             title: str = 'Confusion Matrix',
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""

        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray,
                       save_path: Optional[str] = None):
        """Plot ROC curves for all models"""

        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, model in self.trained_models.items():
            if model is None or not hasattr(model, 'predict_proba'):
                continue
            
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importances(self, model, feature_names: List[str],
                                top_k: int = 20,
                                save_path: Optional[str] = None):
        """Plot feature importances for tree-based models"""

        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_k:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_k} Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, model_name: Optional[str] = None, 
                   filepath: Optional[Path] = None):
        """Save trained model to disk"""

        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)
        
        if model is None:
            logger.error(f"Model {model_name} not found")
            return
        
        if filepath is None:
            filepath = self.config.MODELS_DIR / f"{model_name}_model.pkl"
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model from disk"""

        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def save_training_report(self, results_df: pd.DataFrame, 
                            filepath: Optional[Path] = None):
        """Save comprehensive training report"""
        
        if filepath is None:
            filepath = self.config.RESULTS_DIR / 'training_report.txt'
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RADIOMICS CLASSIFICATION - TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MODEL COMPARISON:\n")
            f.write(results_df.to_string())
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("DETAILED MODEL RESULTS:\n")
            f.write("=" * 80 + "\n\n")
            
            for name, history in self.training_history.items():
                f.write(f"\n{name.upper()}:\n")
                f.write(f"  Best CV Score: {history['best_score']:.4f}\n")
                f.write(f"  Best Parameters: {history['best_params']}\n")
                f.write("-" * 40 + "\n")
        
        logger.info(f"Training report saved to {filepath}")
            
            
    






