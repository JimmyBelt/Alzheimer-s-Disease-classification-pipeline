""" 
    Complete radiomics pipeline integeting all components
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import json
from datetime import datetime

from config import Config
from feature_extraction import RadiomicsExtractor
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class RadiomicsPipeline:
    """Complete radiomics analysis pipeline"""

    def __init__(self, config: Optional[Config] = None):
        """ 
        Initialize pipeline with configuration
        
        Args:
            config: Configuration object (uses default if None)
            
        """

        self.config = config if config else Config()
        self.setup_logging()

        # Components
        self.extractor = RadiomicsExtractor(self.config)
        self.preprocessor = ImagePreprocessor(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.trainer = ModelTrainer(self.config)

        # Data Storage
        self.features_df = None
        self.labels = None
        self.subject_info = None

        # Processed data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Transformations
        self.scaler = None
        self.selected_features = None

        # Metadata
        self.pipeline_metadata = {
            'creation_date': datetime.now().isoformat(),
            'config': {}
        }
    

    def setup_logging(self):
        """Setup loggigng configuration"""

        log_config = self.config.LOGGING

        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
            logging.FileHandler(log_config['file']),
            logging.StreamHandler()
            ]
        )

        logger.info("="*80)
        logger.info("RADIOMICS PIPELINE INITIALIZED")
        logger.info("="*80)


    def extract_features(self, dataset_path:Path):
        """ 
        Extract features from dataset
        
        Args:
            dataset_path: Path to dataset directory
        """

        logger.info(f"Starting feature extraction from: {dataset_path}")

        self.features_df, self.labels, self.subject_info = \
            self.extractor.extract_from_dictionary(dataset_path)
        
        logger.info(f"Extracted {self.features_df.shape[1]} features from "
                    f"{self.features_df.shape[0]} subjects")
        
        # Save raw features
        self.save_features(self.config.RESULTS_DIR / 'raw_features.csv')

        return self.features_df, self.labels
    

    def split_data(self):
        """ Split data into train, validation and test sets"""

        logger.info("Splitting data into train/validation/test sets")

        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.features_df,
            self.labels,
            test_size = self.config.TRAINING['test_size'],
            random_state=self.config.TRAINING['random_state'],
            stratify=self.labels
        )

        # Second split: Train vs Validation
        val_size = self.config.TRAINING['validation_size']
        val_size_adjusted = val_size / (1 - self.config.TRAINING['test_size'])

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.TRAINING['random_state'],
            stratify=y_temp
            )

        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Validation set: {self.X_val.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")

        logger.info(f"Train class distribution: {np.bincount(self.y_train)}")
        logger.info(f"Val class distribution: {np.bincount(self.y_val)}")
        logger.info(f"Test class distribution: {np.bincount(self.y_test)}")


    def scale_features(self):
        """Scale features using configured method"""

        method = self.config.SCALING['method']
        logger.info(f"Scaling features using {method} method")
        
        # # CRITICAL: Filter out diagnostic features BEFORE scaling
        # logger.info("Filtering diagnostic features before scaling...")
        # diagnostic_cols = [col for col in self.X_train.columns if col.startswith('diagnostics_')]
        
        # if diagnostic_cols:
        #     logger.info(f"Removing {len(diagnostic_cols)} diagnostic features")
        #     self.X_train = self.X_train.drop(columns=diagnostic_cols)
        #     self.X_val = self.X_val.drop(columns=diagnostic_cols)
        #     self.X_test = self.X_test.drop(columns=diagnostic_cols)
        # else:
        #     logger.info("No diagnostic features found in data")

        if self.X_train.empty:
            raise ValueError("X_train is empty after numeric conversion. Check feature extraction.")

        if method == 'minmax':
            self.scaler = MinMaxScaler(
                feature_range=self.config.SCALING['feature_range']
            )
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data only
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        # Transform validation and test data
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )

        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )        

        logger.info("Feature scaling completed")


    def select_features(self):
        """Perform comprenshive feature selection"""

        logger.info("Starting feature selection")

        X_selected, self.selected_features = \
            self.feature_selector.comprehensive_selection(self.X_train, self.y_train)

        # Update dataset with selected features
        self.X_train = X_selected
        self.X_val = self.X_val[self.selected_features]
        self.X_test = self.X_test[self.selected_features]

        logger.info(f"Feature selection completed: {len(self.selected_features)} feature selected")

        # Save feature selection summary
        self.feature_selector.plot_selection_summary(
            save_path = self.config.RESULTS_DIR / 'features_selection_summary.png'
        )
        return self.selected_features
    

    def train_models(self):
        """Train all enabled models"""

        logger.info("Starting model training")

        self.trainer.train_all_models(
            self.X_train.values,
            self.y_train
        )

        logger.info(f"Training completed. Best model: {self.trainer.best_model_name}")


    def evaluate_models(self):
        """Evaluate all trained models"""

        logger.info("Evaluating models on all datasets")

        results_df = self.trainer.evaluate_all_models(
            self.X_train.values, self.y_train,
            self.X_test.values, self.y_test
        )

        # Save results
        results_df.to_csv(
            self.config.RESULTS_DIR / 'model_comparison.csv',
            index=False
        )

        # Save training report
        self.trainer.save_training_report(results_df)

        # Plot ROC curve
        self.trainer.plot_roc_curves(
            self.X_test.values, self.y_test,
            save_path=self.config.RESULTS_DIR / 'roc_curves.png'
        )

        # Plot confusion matrix for best model
        self.trainer.plot_confusion_matrix(
            self.trainer.best_model,
            self.X_test.values, self.y_test,
            title=f"Confusion Matrix - {self.trainer.best_model_name}",
            save_path=self.config.RESULTS_DIR / 'confusion_matrix.png'
        )
        
        # Plot feature importances if available
        if hasattr(self.trainer.best_model.best_estimator_, 'feature_importance_'):
            self.trainer.plot_feature_importances(
                self.trainer.best_model.best_estimator_,
                self.selected_features,
                save_path=self.config.RESULTS_DIR / "feature_importance.png"
            )

        return results_df
    

    def run_complete_pipeline(self, dataset_path: Path):
        """ 
        Run the complete pipeline from data to trained models
        
        Args:
            dataset_path: Path to dataset directory
        """

        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("="*80 + "\n")

        try:
            # Step 1: Extract features
            logger.info("STEP 1: Feature extraction")
            self.extract_features(dataset_path)

            # Step 2: Split data
            logger.info("\nSTEP 2: Data Splitting")
            self.split_data()

            # Step 3: Select features
            logger.info("\nSTEP 3: Feature Selection")
            self.select_features()

            # Step 4: Scale features
            logger.info("\nSTEP 4: Feature Scaling")
            self.scale_features()

            # Step 5: Train models
            logger.info("\nSTEP 5: Model Training")
            self.train_models()

            # Step 6: Evaluate models
            logger.info("\nSTEP 6: Model Evaluation")
            results_df = self.evaluate_models()

            # Step 7: Save pipeline
            logger.info("\nSTEP 7: Saving Pipeline")
            self.save_pipeline()

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return results_df

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


    def save_pipeline(self, filepath: Optional[Path] =None):
        """ 
        Save complete pipeline (models, scalers, selected features)
        
        Args:
            filepath: Path to save pipeline (user default if None)
        """       

        if filepath is None:
            filepath = self.config.MODELS_DIR / 'pipeline.pkl'
        
        pipeline_data = {
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'best_model': self.trainer.best_model,
            'best_model_name': self.trainer.best_model_name,
            'all_models': self.trainer.trained_models,
            'feature_selector': self.feature_selector,
            'config': self.config,
            'metadata': self.pipeline_metadata
        }

        joblib.dump(pipeline_data, filepath)
        logger.info(f"Complete pipeline saved to {filepath}")

        # Save metadata as JSON
        metadata_path = filepath.parent / 'pipeline_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'creation_date': self.pipeline_metadata['creation_date'],
                'best_model': self.trainer.best_model_name,
                'n_features': len(self.selected_features),
                'selected_features': self.selected_features,
                'n_samples_train': len(self.y_train),
                'n_samples_val': len(self.y_val),
                'n_samples_test': len(self.y_test)
            }, f, indent=2)

        logger.info(f"Pipeline metadata saved to {metadata_path}")


    def load_pipeline(self, filepath: Path):
        """ 
        Load complete pipeline
        
        Args:
            filepath Path to saved pipeline
        """

        logger.info(f"Loading pipeline from {filepath}")

        pipeline_data = joblib.load(filepath)

        self.scaler = pipeline_data['scaler']
        self.selected_features = pipeline_data['selected_features']
        self.trainer.best_model = pipeline_data['best_model']
        self.trainer.best_model_name = pipeline_data['best_model_name']
        self.trainer.trained_models = pipeline_data['all_models']
        self.feature_selector = pipeline_data['feature_selector']
        self.pipeline_metadata = pipeline_data['metadata']

        logger.info("Pipeline loaded successfullly")
        logger.info(f"Best model: {self.trainer.best_model_name}")
        logger.info(f"Selected features: {len(self.selected_features)}")

    
    def predict_new_image(self, image_path: Path, mask_path: Path) -> Dict:
        """ 
        Predict on a new image
        
        Args:
            image_path: Path to image
            mask_path: Path to mask
        
        Returns:
            Dictionary with prediction results
        """

        if self.trainer.best_model is None:
            raise ValueError("No trained model available. Train or load a model first.")
        
        logger.info(f"Predicting on new image: {image_path.name}")
        
        # Preprocess
        image, mask = self.preprocessor.preprocess_image_pair(image_path, mask_path)
        
        # Extract features
        features = self.extractor.extract_features_single(image, mask)
        
        # # CRITICAL: Filter out diagnostic features (same as training)
        # features_filtered = {k: v for k, v in features.items() 
        #                     if not k.startswith("diagnostics_")}
        
        features_df = pd.DataFrame([features])
        
        # Check if all required features are present
        missing_features = set(self.selected_features) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with zeros")
            for feat in missing_features:
                features_df[feat] = 0
        
        # Select only the features used in training
        features_df = features_df[self.selected_features]
        
        # Scale
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.trainer.best_model.predict(features_scaled)[0]
        probability = self.trainer.best_model.predict_proba(features_scaled)[0]  
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'AD' if prediction == 1 else 'CN',
            'probability_CN': float(probability[0]),
            'probability_AD': float(probability[1]),
            'confidence': float(max(probability)),
            'model_used': self.trainer.best_model_name
        }
        
        logger.info(f"Prediction: {result['prediction_label']} "
                    f"(confidence: {result['confidence']:.2%})")
        
        return result
    

    def save_features(self, filepath: Path):
        """ Save features to CSV"""

        if self.features_df is not None:
            self.features_df.to_csv(filepath)
            logger.info(f"Features saved to {filepath}")
    

    def generate_report(self, output_path: Optional[Path] = None):
        """Generate comprehensive HTML report"""
        
        if output_path is None:
            output_path = self.config.RESULTS_DIR / 'pipeline_report.html'
        
        # Generate a comprensive HTML report
        logger.info("Generating pipeline report...")

        report_text = f"""
        RADIOMICS PIPELINE REPORT
        ==========================

        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Dateset Information:
        - Total samples: {len(self.labels)}
        - Training samples: {len(self.y_train)}
        - Validation samples: {len(self.y_val)}
        - Test samples: {len(self.y_test)}
        
        Feature Extraction:
        - Total features extracted: {self.features_df.shape[1]}
        - Selected features: {len(self.selected_features)}
        
        Best Model:
        - Model: {self.trainer.best_model_name}
        - CV Score: {self.trainer.best_model.best_score_:.4f}
        
        Results saved to: {self.config.RESULTS_DIR}        
        """

        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(report_text)

        logger.info(f"Report saved to {output_path.with_suffix('.txt')}")

