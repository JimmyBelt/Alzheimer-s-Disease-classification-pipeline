""" 
    Feature selection module with univariante and multivariante methods
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2,
    RFE, RFECV, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Advanced feature selection with multiple methods"""

    def __init__(self, config):
        self.config = config
        self.fs_config = config.FEATURE_SELECTION
        self.selected_features_ = None
        self.feature_scores_ = {}
        self.selection_history_ = []

    def remove_low_variance_features(self, X: pd.DataFrame,
                                     threshold: Optional[float]=None)->pd.DataFrame:
        """ 
        Remove features with low variance
        
        Args:
            X: Feature dataframe
            threshold: Variance threshold (uses config if None)
            
        Returns:
            Filtered dataframe
        """

        if threshold is None:
            threshold = self. fs_config['variance_threshold']

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        selected_cols = X.columns[selector.get_support()]
        removed = len(X.columns) - len(selected_cols)

        logger.info(f"Variance threshold: Removed {removed} features with variance < {threshold}")

        self.selection_history_.append({
            'method': 'variance_threshold',
            'n_features': len(selected_cols),
            'removed': removed
        })

        return X[selected_cols]
    
    def removed_correlated_features(self, X: pd.DataFrame, y:np.ndarray,
                                    threshold: Optional[float]=None,
                                    method: str = 'pearson')->pd.DataFrame:
        """ 
        Remove highly correlated features (univariante correlation)
        
        Args:
            X: Feature dataframe
            y: Target array
            threshold: Correlation threshold
            method: 'pearson' or 'spearman'
            
        Returns:
            Filtered dataframe
        """

        if threshold is None:
            threshold = self.fs_config['correlation_threshold']
        
        # Calculate correlation matrix
        corr_matrix =  X.corr(method=method).abs()

        # Find upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater that threshold
        to_drop = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > threshold):
                # Keep the feature with higher correlation to target
                correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
                correlated_features.append(column)

                # Calculate correlation with target for each
                target_corrs = {}
                for feat in correlated_features:
                    if feat not in to_drop:
                        target_corrs[feat] = abs(np.corrcoef(X[feat], y)[0, 1])

                # Keep the one with highest target correlation
                best_feature = max(target_corrs, key=target_corrs.get)
                for feat in correlated_features:
                    if feat != best_feature and feat not in to_drop:
                        to_drop.append(feat)
        
        logger.info(f"Correlation filter: Removed {len(to_drop)} features with correlation > {threshold}")
        self.selection_history_.append({
            'method': 'correlation_filter',
            'threshold': threshold,
            'n_features': len(X.columns) - len(to_drop),
            'removed': len(to_drop)
        })

        return X.drop(columns=to_drop)
    
    def select_univariante(self, X: pd.DataFrame, y:np.ndarray,
                           method: Optional[str]=None,
                           k: Optional[int]=None) -> pd.DataFrame:
        """ 
        Univariante feature selection
        
        Args:
            X: Feature dataframe
            y: Target array
            method: 'mutual_info', 'f_classif', 'chi2'
            k: Number of features to select
        
        Returns:
            Selected features dataframe
        """

        if method is None:
            method = self.fs_config['univariant_method']
        if k is None:
            k = min(self.fs_config['univariante_k'], X.shape[1])
        
        # Choose scoring function
        score_func_map = {
            'mutual_info': mutual_info_classif,
            'f_classif': f_classif,
            'chi2': chi2
        }

        if method not in score_func_map:
            raise ValueError(f"Unknown methos: {method}")
        
        score_func = score_func_map[method]

        # For chi2, ensure all values are non-negative
        if method == 'chi2':
            X_temp = X-X.min() + 1e-10
        else:
            X_temp = X

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_temp, y)

        selected_cols = X.columns[selector.get_support()]
        scores = selector.scores_

        # Store scores
        self.feature_scores_[method] = dict(zip(X.columns, scores))

        logger.info(f"Univariante selection ({method}): Selected {len(selected_cols)} features")
        self. selection_history_.append({
            'method': f'univariante_{method}',
            'n_features': len(selected_cols),
            'k': k
        })

        return X[selected_cols]
    
    def select_rfe(self, X: pd.DataFrame, y: np.ndarray,
                   n_features: Optional[int] = None,
                   step: int = 1,
                   cv: Optional[int] = None) -> pd.DataFrame:
        """ 
        Recursive feature Elimination (RFE)
        
        Args:
            X: Feature dataframe
            y: Target array
            n_features: Number of features to select (None for RFECV)
            step: Features to remove at each iteration
            cv: Number of CV folds (None for no CV)
            
        Returns:
            Selected features dataframe
        """

        if not self.fs_config['use_rfe']:
            return X
        
        if n_features is None:
            n_features = self.fs_config['rfe_n_features']
        if step is None:
            step = self.fs_config['rfe_step']

        # Use Random Forest as estimator
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.config.TRAINING['random_state'],
            n_jobs = -1
        )

        if cv is not None:
            # Use RFECV for cross-validated selection
            selector = RFECV(
                estimator=estimator,
                step=step,
                cv=cv,
                scoring='roc_auc',
                n_jobs = -1,
                min_features_to_select=max(5, n_features // 2)
            )
        else:
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=step
            )
        
        selector.fit(X,y)
        selected_cols = X.columns[selector.get_support()]

        # Store rankings
        self.feature_scores_['rfe_ranking'] = dict(zip(X.columns, selector.ranking_))

        logger.info(f"RFE: Selected {len(selected_cols)} features")
        self.selection_history_.append({
            'method': 'rfe',
            'n_features': len(selected_cols),
            'cv': cv is not None
        })

        return X[selected_cols]
    
    def select_boruta(self, X: pd.DataFrame, y: np.ndarray,
                      max_iter: Optional[int]=None,
                      perc: Optional[int]=None)-> pd.DataFrame:
        """
        Boruta algorithm for feature selection (multivariante)
        Indentifies all relevant features
        
        Args:
            X: Feature dataframe
            y: Target array
            max_iter: Maximum iterations
            perc: Percentile for importance threshold
            
        Returns:
            Selected features dataframe
        """

        if not self.fs_config['use_boruta']:
            return X
        
        try:
            from boruta import BorutaPy
        except ImportError:
            logger.warning("Boruta not installed. Skipping Boruta selection. Install with: pip install boruta")
            return X
        
        if max_iter is None:
            max_iter = self.fs_config['boruta_max_iter']
        if perc is None:
            perc = self.fs_config['boruta_perc']
        
        # Use Random Forest as estimator
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=self.config.TRAINING['random_state']
        )

        # Initialize Boruta
        boruta_selector = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            max_iter=max_iter,
            perc=perc,
            random_state=self.config.TRAINING['random_state']
        )

        # Fit Boruta
        boruta_selector.fit(X.values, y)

        # Get selected features
        selected_cols = X.columns[boruta_selector.support_]

        # Store rankings
        self.feature_scores_['boruta_ranking'] = dict(zip(X.columns, boruta_selector.ranking_))
        logger.info(f"Boruta: Selected {len(selected_cols)} features as confirmed importance")
        self.selection_history_.append({
            'method': 'boruta',
            'n_features': len(selected_cols),
            'max_iter': max_iter
        })

        return X[selected_cols]
    
    def select_from_model(self, X: pd.DataFrame, y: np.ndarray,
                          threshold: str = 'median') -> pd.DataFrame:
        """ 
        Feature selecting using model importance (multivariante)
        
        Args:
            X: Feature dataframe
            y: Target array
            threshold: Threshold for selection ('median', 'mean', or float)
            
        Returns:
            Selected features dataframe   
        """

        from sklearn.feature_selection import SelectFromModel

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=self.config.TRAINING['random_state'],
            n_jobs=-1
        )
        rf.fit(X,y)

        # Selected features
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        selected_cols = X.columns[selector.get_support()]

        # Store importance
        self.feature_scores_['rf_importance'] = dict(zip(X.columns, rf.feature_importances_))

        logger.info(f"Model-based selection: Selected {len(selected_cols)} features")
        self.selection_history_.append({
            'method': 'select_from_model',
            'n_features': len(selected_cols),
            'threshold': threshold
        })

        return X[selected_cols]
    
    def comprehensive_selection(self, X: pd.DataFrame, y: np.ndarray,
                                final_k: Optional[int]=None) -> Tuple[pd.DataFrame, List[str]]:
        """ 
        Complete feature selection pipeline with multiple methods
        
        Args:
            X: Feature datframe
            y: Target array
            final_k: Final number of features to select
            
        Returns:
            Tuple of (selected features dataframe, list of selected feature names)
        """

        logger.info(f"Starting compehensive feature selection with {X.shape[1]} features")

        if final_k is None:
            final_k = self.fs_config['final_k_features']

        def ensure_numeric(df, name):
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            empty_cols = df_numeric.columns[df_numeric.isna().all()]
            if len(empty_cols) > 0:
                df_numeric = df_numeric.drop(columns=empty_cols)
            df_numeric = df_numeric.fillna(0)
            return df_numeric

        X = ensure_numeric(X, "X")
        
        # Step 1: Remove low variance features
        X_filtered = self.remove_low_variance_features(X)
        logger.info(f"After variance filter: {X_filtered.shape[1]} features")

        # Step 2: Remove correlated features (univariante)
        X_filtered = self.removed_correlated_features(X_filtered, y)
        logger.info(f"After correlation filter: {X_filtered.shape[1]} features")

        # Step 3: Univariante selection
        X_filtered = self.select_univariante(X_filtered, y, k=min(100, X_filtered.shape[1]))
        logger.info(f"After univariante selection: {X_filtered.shape[1]} features")

        # Step 4: Multivariate - RFE
        if self.fs_config['use_rfe'] and X_filtered.shape[1] > final_k:
            X_filtered = self.select_rfe(X_filtered, y, n_features=min(50, X_filtered.shape[1]))
            logger.info(f"After RFE: {X_filtered.shape[1]} features")
        
        # Step 5: Multivariante - Boruta (if requested and available)
        if self.fs_config['use_boruta'] and X_filtered.shape[1] > final_k:
            X_filtered = self.select_boruta(X_filtered, y)
            logger.info(f" After Boruta: {X_filtered.shape[1]} features")

        # Step 6: Final selection using model importance
        if X_filtered.shape[1] > final_k:
            # Use mutula information for final ranking
            scores = mutual_info_classif(X_filtered, y, random_state=self.config.TRAINING['random_state'])
            top_indices = np.argsort(scores)[-final_k:]
            selected_features = X_filtered.columns[top_indices].tolist()
            X_final = X_filtered[selected_features]

        else:
            X_final = X_filtered
            selected_features = X_filtered.columns.tolist()

        self.selected_features_ = selected_features
        logger.info(f"Final selection: {len(selected_features)} features")

        return X_final, selected_features
    
    def plot_selection_summary(self, save_path: Optional[str]=None):
        """Plot summary of feature selection process"""

        if not self.selection_history_:
            logger.warning("No selection history availabe")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = [step['method'] for step in self.selection_history_]
        n_features = [step['n_features'] for step in self.selection_history_]

        ax.plot(range(len(methods)), n_features, marker='o', linewidth=2, markersize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_xlabel('Selection Step')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Selection Pipeline')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Selection summary saved to {save_path}")
        
        plt.show()

    def pltot_feature_scores(self, method: str='mutual_info',
                             top_k: int=20,
                             save_path:Optional[str] =None):
        """Plot top feature scores from a specific method"""
        
        if method not in self.feature_scores_:
            logger.warning(f"No scores available for method: {method}")
            return
    
        scores = self.feature_scores_[method]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        features, values = zip(*sorted_scores)

        fig, ax = plt.subplots(fisize=(12,8))
        ax.barh(range(len(features)), values)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Score')
        ax.set_title(f"Top {top_k} features by {method}")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Features scores plot saved to {save_path}")

        plt.show()



 


