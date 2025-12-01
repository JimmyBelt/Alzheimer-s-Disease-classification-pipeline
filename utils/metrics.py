""" 
    Utility functions for model evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from typing import Optional, List

def plot_importances(importances: np.ndarray, feature_names: List[str],
                     top_k: int = 20, title: str = 'Feature Importances',
                     save_path: Optional[str] = None):
    """ 
    Plot feature importances
    
    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        top_k: Number of top features to display
        title: Plot title
        save_path: Path to save figure
    """

    # Sort by importance
    indices = np.argsort(importances)[-top_k:]

    plt.figure(figsize=(10,8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def analyze_train_test_performance(model, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray,
                                   save_path: Optional[str] = None):
    """ 
    Comprhensive analysis of model performance on train and test sets
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        save_path: Path to save figure

    """

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
    
    else:
        y_train_prob = None
        y_test_prob = None

    # Create figure
    plt.figure(figsize=(16,10))

    # 1. Confusion Matrices
    ax1 = plt.subplot(2,3,1)
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Train Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predict Label')

    ax2 = plt.subplot(2,3,2)
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax2)
    ax2.set_title('Test Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predict Label')

    # 2. ROC Curves
    if y_train_prob is not None and y_test_prob is not None:
        ax3 = plt.subplot(2,3,3)

        # Train ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        roc_auc_train = auc(fpr_train, tpr_train)
        ax3.plot(fpr_train, tpr_train, label = f"Train (AUC = {roc_auc_train:.3f})",
                 linewidth=2, color='blue')
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
        roc_auc_test = auc(fpr_test, tpr_test)
        ax3.plot(fpr_test, tpr_test, label=f'Test (AUC = {roc_auc_test:.3f})',
                 linewidth=2, color='orange')
        
        ax3.plot([0,1],[0,1], 'k--', label='Random', linewidth=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 3. Metrics Comparison
    ax4 = plt.subplot(2,3,4)

    metrics_train = {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred, zero_division=0),
        'Recall': recall_score(y_train, y_train_pred, zero_division=0),
        'F1': f1_score(y_train, y_train_pred, zero_division=0)
    }

    metrics_test = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'F1': f1_score(y_test, y_test_pred, zero_division=0)
    }

    x = np.arange(len(metrics_train))
    width = 0.35

    ax4.bar(x-width/2, list(metrics_train.values()), width,
            label='Train', color='steelblue')
    ax4.bar(x+width/2, list(metrics_test.values()), width,
            label='Test', color='coral')
    
    ax4.set_ylabel('Score')
    ax4.set_title('Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_train.keys())
    ax4.legend()
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y') 

    # 4. Precision-Recall Curves
    if y_train_prob is not None and y_test_prob is not None:
        ax5 = plt.subplot(2,3,5)

        # Train PR
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
        pr_auc_train = auc(recall_train, precision_train)
        ax5.plot(recall_train, precision_train,
                 lable=f'Train (AUC = {pr_auc_train:.3f})',
                linewidth=2, color='steelblue')
        
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision-Recall Curves')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 5. Prediction Distribution
    ax6 = plt.subplot(2,3,6)

    if y_train_prob is not None and y_test_prob is not None:
        ax6.hist(y_train_prob[y_train == 0], bins=20, alpha=0.5,
                 label='Train CN', color='lightblue')
        ax6.hist(y_train_prob[y_train == 1], bins=20, alpha=0.5,
                 label='Train AD', color='lightcoral')
        ax6.hist(y_test_prob[y_test == 0], bins=20, alpha=0.5,
                 lable='Test CN', color='blue', histtype='step', linewidth=2)
        ax6.hist(y_test_prob[y_test == 1], bins=20, alpha=0.5,
                 label='Test AD', color='red', histtype='step', linewidth=2)
        
        ax6.set_xlabel('prediction Probability')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Prediction Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE METRICS")
    print("="*60)

    print("\nTRAINING SET:")
    print(classification_report(y_train, y_train_pred,
                                target_names=['CN', 'AD']))
    if y_train_prob is not None:
        print(f"ROC AUC: {roc_auc_score(y_train, y_train_prob):.4f}")

    print("\nTEST SET:")
    print(classification_report(y_test, y_test_pred,
                                target_names=['CN', 'AD']))
    
    if y_test_prob is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, y_test_prob):.4f}")

    print("\n OVERFITTING ANALYSIS")
    print(f"Accuracy gap: {metrics_train['Accuracy'] - metrics_test['Accuracy']:.4f}")
    print(f"F! gap: {metrics_train['F1'] - metrics_test['F1']:.4f}")

    if metrics_train['Accuracy'] - metrics_test['Accuracy'] > 0.1:
        print("WARNING: Model may be overfitting (accuracy gap > 0.1)")
    else:
        print("Model generalization appears good")

    print("="*60 + "\n")

def plot_learning_curves(model, X_train: np.ndarray, y_train: np.ndarray,
                         cv: int=5, save_path: Optional[str]=None):
    """ 
    Plot learning curves to diagnose bias/variance
    
    Args:
        model: Model to evaluate
        X_train: training features
        y_train: Training labels
        cv: Number of cross-validation folds
        save_path: Path to save figure
    """

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.1,1.0,10),
        scoring='roc_auc'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10,6))

    plt.plot(train_sizes, train_mean, label='Training score',
             color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, val_mean, label='Validation score',
             color='red', marker='s')
    plt.fill_between(train_sizes, val_mean-val_std,
                     val_mean + val_std, alpha=0.15, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC AUC Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                           n_bins: int=10, save_path: Optional[str] = None):
    """ 
    PLot calibration curve to assess probability predictions
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities 
        n_bins: Number of bins for calibration
        save_path: Path to save figure
    """

    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )

    plt.figure(figsize=(8,8))

    plt.plot(mean_predicted_value, fraction_of_positives, 's-',
             label='Model', linewidth=2, markersize=8)
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration', linewidth=2)

    plt.xlabel('Mean Predicted Posibility')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dp1=300, bbox_inches='tight')

    plt.show()

def compare_models_performance(models_results: dict, save_path: Optional[str]=None):
    """
    Compare multiple models performance
    
    Args:
        models_results: Dict with model names as keys and metrics dicts as values
        save_path: Path to save figure
    """

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    n_models = len(models_results)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, 2, figsize=(15,5))

    # Bar plot
    x = np.arange(n_metrics)
    width = 0.8/n_models

    for i, (model_name, results) in enumerate(models_results.items()):
        values = [results.get(m, 0) for m in metrics]
        axes[0].bar(x+i*width, values, width, label=model_name)

    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x+width*(n_models-1)/2)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(122, projection='polar')
    
    for model_name, results in models_results.items():
        values = [results.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1.0])
    ax.set_title('Model Performance Radar')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
        









    