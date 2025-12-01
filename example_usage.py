""" 
    Example usage of the radiomics pipeline
    This scrip demostrates common use cases
"""

from pathlib import Path
from config import Config
from pipeline import RadiomicsPipeline
from predict import RadiomicsPredictior
import numpy as np

def example_1_complete_training():
    """Example 1: Complete training pipeline"""

    print("\n" + "="*80)
    print("EXAMPLE 1: Complete Training Pipeline")
    print("="*80 + "\n")

    # Setup configuration
    config = Config()

    # Optional: Customize settings fr fast training
    config.FEATURE_SELECTION['final_k_features'] = 10 # This number could be changed
    config.TRAINING['cv_folds'] = 3 # Reduced to 3 for speed

    # Disable some models for faster training (not mandatory)
    config.MODELS['svm']['enabled'] = False
    config.MODELS['logistic_regression']['enabled'] = False

    # Initialize pipeline
    pipeline = RadiomicsPipeline(config)

    # Run complete pipeline
    dataset_path = Path("path/to/dataset")      # CHange this

    try:
        _ = pipeline.run_complete_pipeline(dataset_path=dataset_path)

        print("\n Training completed successfully")
        print(f"Best model: {pipeline.trainer.best_model_name}")
        print(f"Results saved to: {config.RESULTS_DIR}")

        return pipeline
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return None
    
def example_2_predict_new_image():
    """Example 2: Predict on a new image using trained pipeline"""

    print("\n" + "="*81)
    print("EXAMPLE 2: Predict on New Image")
    print("="*80 + "\n")

    # Load trained pipeline
    pipeline_path = Path('models/pipeline.pkl')

    if not pipeline_path.exists():
        print("No trained pipeline found. Run example_1 first!!!")
        return
    predictor = RadiomicsPredictior(pipeline_path=pipeline_path)

    # Predict on new image
    image_path = Path('path/to/new/image.nii.gz')   # Change this
    mask_path = Path("path/to/new/mask.nii.gz")     # Change this

    try:
        result = predictor.predict_single(image_path, mask_path, save_results=True)

        print("\nPrediction completed!")
        print(f" DIagnosis: {result['prediction_labl']}")
        print(f" Confidence: {result['confidence']:.2%}")
        print(f" P(CN): {result['probability_CN']:.2%}")
        print(f" P(AD): {result['probability_AD']:.2%}")
        print(f" Model: {result['model_used']}")

        # Interpretation
        if result['confidence'] > 0.8:
            print("\nHigh confidence prediction")
        elif result['confidence'] > 0.6:
            print("\n Medium confidence prediction")
        else:
            print("\nLow confidence prediction - review recommended")

        return result
    
    except Exception as e:
        print(f"\nPrediction failed: {str(e)}")
        return None
    
def example_3_batch_prediction():
    """Example 3: Batch prediction on multiple images"""

    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Prediction")
    print("="*80 + "\n")

    pipeline_path = Path('models/pipeline.pkl')

    if not pipeline_path.exists():
        print("No trianed pipeline found. Run exaple_1 first!!")
        return
    
    predictor = RadiomicsPredictior(pipeline_path)

    # Define list of images to process
    data_list = [
        {
            'subject_id': 'patient_001',
            'image_path': 'path/to/patient_001/T1/T1.nii.gz',
            'mask_path': 'path/to/patient_001/label/label.nii.gz'
        },
        {
            'subject_id': 'patient_002',
            'image_path': 'path/to/patient_002/T1/T1.nii.gz',
            'mask_path': 'path/to/patient_002/label/label.nii.gz'
        },
        # Continue with the rest of patients
    ]

    try:
        results_df = predictor.predict_batch(data_list, save_results=True)

        print("\n✓ Batch prediction completed!")
        print(f"  Processed: {len(results_df)} subjects")
        print("  Results saved to: batch_predictions.csv")
        
        # Show summary
        print("\nPrediction Summary:")
        print(results_df[['subject_id', 'prediction_label', 'confidence']].to_string())

        # Generate report
        report = predictor.get_prediction_report(results_df)
        print(report)

    except Exception as e:
        print(f"\nBatch prediction failed: {str(e)}")

def example_4_custom_feature_selection():
    """Example 4: Training with custom feature selection"""

    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Feature Selection")
    print("="*80 +"\n")

    config = Config()

    # Custom feature selection
    config.FEATURE_SELECTION = {
        'correlation_threshold': 0.85,  # More strict correlation
        'univariate_method': 'f_classif',  # Use f-test instead of mutual info
        'univariate_k': 100,
        'use_rfe': True,
        'rfe_n_features': 30,
        'use_boruta': True,  # Enable Boruta
        'boruta_max_iter': 50,
        'final_k_features': 20  # More features for final model
    }

    pipeline = RadiomicsPipeline(config)
    dataset_path = Path('path/to/the/dataset')     # CHange this

    try: 
        # Run Pipeline
        _ = pipeline.run_complete_pipeline(dataset_path=dataset_path)

        print("\nCustom feature selection completed!!")
        print(f" Selected features: {len(pipeline.selected_features)}")
        print(f" Feature names: {pipeline.selected_features[:5]}...")

        return pipeline
    
    except Exception as e:
        print(f"\n Error: {str(e)}")
        return None
    
def example_5_model_comparison():
    """Example 5: Compare multiple models"""

    print("\n" + "="*81)
    print("EXAMPLE 5: Model Comparison")
    print("="*81 + "\n")

    config = Config()

    # Enable all models for comparison
    for model_name in config.MODELS:
        config.MODELS[model_name]['enabled'] = True

    pipeline = RadiomicsPipeline(config)

    dataset_path = Path('path/to/the/dataset')   # CHange this path

    try:
        # Run pipelien
        results = pipeline.run_complete_pipeline(dataset_path=dataset_path)

        print("\nModel comparison completed!")
        print("\nModel Performanece Comparison:")
        print(results.to_string(index=False))

        # Find best models
        best_by_auc = results.loc[results['Test_ROC_AUC'].idxmax()]
        best_by_f1 = results.loc[results['Test_F1'].idxmax()]

        print(f"\n Best by ROC-AUC: {best_by_auc['Model']} ({best_by_auc['Test_RO_AUC']:.4f})")
        print(f" Best by F1: {best_by_f1['Model']} ({best_by_f1['Test_F1']:.4f})")

        return pipeline
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def example_6_feature_extraction_only():
    """Example 6: Extract features without training"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Feature Extraction Only")
    print("="*80 + "\n")

    from feature_extraction import RadiomicsExtractor

    config = Config
    extractor = RadiomicsExtractor(config)

    dataset_path = Path('path/to/your/dataset')         # Change this

    try:
        # Extract features
        features_df, labels, subject_info = extractor.extract_from_dictionary(dataset_path)
        print("\nFeature extraction completed!")
        print(f" Subjects: {features_df.shape[0]}")
        print(f" Features: {features_df.shape[1]}")
        print(f" Class distribution: AD={np.sum(labels==1)}, CN={np.sum(labels==0)}")

        # Save features
        output_dir = Path('extracted_features')
        output_dir.mkdir(exist_ok=True)

        features_df.to_csv(output_dir / 'features.csv')
        np.save(output_dir / 'labels.npy', labels)

        print(f"\n Features saved to: {output_dir}")

        # Aalyze feature distribution
        extractor.analyze_feature_distribution(
            features_df,
            labels,
            save_path=output_dir / 'feature_distribution.png'
        )
        return features_df, labels
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None, None
    
def example_7_load_and_continue():
    """Example 7: Load existing pipeline and continue analysis"""

    print("\n"+"="*81)
    print("EXAMPLE 7: Load and Continue Analysis")
    print("="*81+"\n")

    pipeline_path = Path('models/pipeline.pkl')

    if not pipeline_path.exists():
        print("No trained pipeline found")
        return
    
    # Load pipeline
    config = Config()
    pipeline = RadiomicsPipeline(config)
    pipeline.load_pipeline(pipeline_path)

    print("Pipeline loaded successfully!!")
    print(f" Best model: {pipeline.trainer.best_model_name}")
    print(f" Selected features: {len(pipeline.selected_features)}")

    # Access trained models
    for model_name, model in pipeline.trainer.trained_models.items():
        if model is not None:
            print(f"\n {model_name}:")
            print(f" CV Score: {model.best_score_:.4f}")
            print(f" Best params: {model.best_params_}")

    # Use for prediction
    print("\n Pipeline ready for predictions!")

    return pipeline

def example_8_quick_evaluation():
    """Example 8: Quick evaluation on test set"""

    print("\n" + "="*80)
    print("EXAMPLE 8: Quick Evaluation")
    print("="*80 + "\n")

    pipeline_path = Path('models/pipeline.pkl')
    test_data_path = Path('path/to/test/dataset')       #Change this

    if not pipeline_path.exists():
        print("No trained pipeline found!")
        return
    
    config = Config()
    pipeline = RadiomicsPipeline(config)
    pipeline.load_pipeline(pipeline_path)

    try:
        # Extract features from test data
        features_df, labels, _ = pipeline.extractor.extract_from_dictionary(test_data_path)

        # Scale and select features
        features_selected = features_df[pipeline.selected_features]
        features_scaled = pipeline.scaler.transform(features_selected)

        # Evaluate 
        metrics = pipeline.trainer.evaluate_model(
            pipeline.trainer.best_model,
            features_scaled, 
            labels,
            set_name='evaluation'
        )

        print("\n Evaluation completed")

        # Visualization
        pipeline.trainer.plot_confusion_matrix(
            pipeline.trainer.best_model,
            features_scaled,
            labels,
            title='Test Set Performance'
        )

        return metrics
    
    except Exception as e:
        print(f"\nError: {str(e)}")

def main():
    """Run examples"""
    
    print("\n" + "="*81)
    print("RADIOMICS PIPELINE -USAGE EXAMPLES")
    print("="*81 + "\n")

    examples = {
        '1': ('Complete Training Pipeline', example_1_complete_training),
        '2': ('Predict on New Image', example_2_predict_new_image),
        '3': ('Batch Prediction', example_3_batch_prediction),
        '4': ('Custom Feature Selection', example_4_custom_feature_selection),
        '5': ('Model Comparison', example_5_model_comparison),
        '6': ('Feature Extraction Only', example_6_feature_extraction_only),
        '7': ('Load and Continue Analysis', example_7_load_and_continue),
        '8': ('Quick Evaluation', example_8_quick_evaluation)
    }

    print("\nAvailabe examples")
    for key, (name,_) in examples.items():
        print(f" {key}. {name}")

    choice = input("\nSelect example to run(1-8, or 'all' for documentation):").strip()

    if choice == 'all':
        print("\n" + "="*80)
        print("Running all examples would require dataset paths.")
        print("Please edit the paths in each function and run individually")
        print("="*80)
        return
    if choice in examples:
        name, func = examples[choice]
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"{'='*80}")
        func()
    else:
        print("Invalid choice...")

    if __name__ == '__main__':
        # Quick start guide
        print("""
    QUICK START GUIDE
    =================
    
    Before running examples, update the following paths in this file:
    
    1. Dataset path: 'path/to/your/dataset'
    2. Image paths for prediction examples
    
    Expected dataset structure:
    dataset/
    ├── AD/
    │   └── subject_*/
    │       ├── T1/T1.nii.gz
    │       └── label/label.nii.gz
    └── CN/
        └── subject_*/
            ├── T1/T1.nii.gz
            └── label/label.nii.gz
    
    First-time users:
    1. Run Example 1 (Complete Training) - this trains the full pipeline
    2. Run Example 2 (Predict) - to test prediction on new images
    3. Explore other examples as needed
    
    For production use:
    - Use main.py for command-line interface
    - Integrate modules into your GUI application
    """)
    
    main()
















