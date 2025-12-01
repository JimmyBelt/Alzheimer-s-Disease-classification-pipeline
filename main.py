"""
    Main script for running the Radiomics Pipeline
"""

import argparse
import logging
from pathlib import Path
import sys

from config import Config
from pipeline import RadiomicsPipeline

def setup_argparse():
    """Setup comand-line argument parse"""
    parser = argparse.ArgumentParser(
        description='Radiomics Pipeline for Alzheimer\'s Disease Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
"""
Examples:
    # Train pipeline on dataset
    python main.py --mode train --data /path/to/dataset

    # Predict on new images
    python main.py --mode predict --pipelines models/pipeline.pkl --image image.nii.gz --mask mask.nii.gz

    # Evaluate existing pipeline
    python main.py --mode evaluate --pipeline models/pipeline.pkl --data /path/to/dataset
""")
    
    parser.add_argument('--mode', type = str, required = True,
                        choices=['train','predict','evaluate','extract'],
                        help='Operation mode')
    
    parser.add_argument('--data', type = str, 
                        help='Path to dataset directory (for train/evaluate/extract modes)')
    
    parser.add_argument('--pipeline', type=str,
                        help='Path to saved pipeline (for predict/evaluate modes)')
    
    parser.add_argument('--image', type=str,
                        help='Path to image file (for predict mode)')
    
    parser.add_argument('--mask', type=str,
                        help='Path to mask file (for predict mode)')
    
    parser.add_argument('--output', type=str,
                        help='Output directory for results')
    
    parser.add_argument('--config', type=str,
                        help='Path to custom configuration file')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level',
                        choices=['DEBUG','INFO','WARNING','ERROR'])
    
    parser.add_argument('--models', type=str,nargs='+',
                        help='Specific models to train (e.g., random_forest xgboost)')
    
    parser.add_argument('--no-preprocessing', action='store_true',
                        help='Skip Image Preprocessing')
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced hyperparameters search')
    
    return parser

def train_mode(args, config):
    """Execute training mode"""
    logging.info("Starting training mode")

    if not args.data:
        raise ValueError('--data is required for training mode')
    
    dataset_path = Path(args.data)
    
    if not dataset_path.exists():
        raise ValueError(f'Dataset path does not exist: {dataset_path}')
    
    # Configuration modification for quick mode
    if args.quick:
        logging.info("Quick mode enabled - reduced hyperparameter search space")

        for model_name in config.MODELS:
            if config.MODELS[model_name]['enabled']:
                #Reduced grid search space
                param_grid = config.MODELS[model_name]['param_grid']

                for param in param_grid:
                    if isinstance(param_grid[param], list) and len(param_grid[param]) > 3:
                        param_grid[param] = param_grid[param][:3] #Take the 3 first parameters checked in the previous condition
    
    if args.models:
        for model_name in config.MODELS:
            config.MODELS[model_name]['enabled'] = model_name in args.models
        logging.info(f'Training only: {args.models}')
    
    


    # Initialize and run pipeline
    pipeline = RadiomicsPipeline(config)
    results = pipeline.run_complete_pipeline(dataset_path)

    # Print summary
    print("\n" + "="*80)
    print("TRAIN SUMMARY")
    print("="*80)
    print(results.to_string())
    print("\n" + "="*80)
    print(f"Best Model: {pipeline.trainer.best_model_name}")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*80)

    return pipeline

def predict_mode(args, config):
    """Execute prediction mode"""
    logging.info("Starting prediction mode")

    if not args.pipeline:
        raise ValueError("--pipeline is required for prediction mode")
    
    if not args.image or not args.mask:
        raise ValueError('--image and --mask are required for prediction mode')
    
    
    
    from predict import RadiomicsPredictor

    # Load Pipeline
    predictor = RadiomicsPredictor(Path(args.pipeline))

    # Predict 
    results = predictor. predict_single(
        Path(args.image),
        Path(args.mask),
        save_results = True
    )

    # Print results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"Diagnosis: {results['prediction_label']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Probability CN: {results['probability_CN']:.2%}")
    print(f"Probability AD: {results['probability_AD']:.2f}")
    print(f"Model: {results['model_used']}")
    print("="*80+"\n")

def evaluate_mode(args, config):
    """Execute evaluate mode"""
    logging.info("Starting evaluation mode")

    if not args.pipeline:
        raise ValueError("--pipeline is required for evaluation mode")
    
    if not args.data:
        raise ValueError("--data is required for evaluation mode")
    

    # Load pipeline
    pipeline = RadiomicsPipeline(config)
    pipeline.load_pipeline(Path(args.pipeline))

    # Extract feature for train data
    logging.info("Extracting features from test dataset ...")
    features_df, labels, _ = pipeline.extractor.extract_from_dictionary(Path(args.data))

    # Scale features
    features_scaled = pipeline.scaler.transform(features_df[pipeline.selected_features])

    # Evaluate
    from model_training import ModelTrainer
    trainer = ModelTrainer(config)
    trainer.best_model = pipeline.trainer.best_model
    trainer.best_model_name = pipeline.trainer.best_model_name

    metrics = trainer.evaluate_model(
        pipeline.trainer.best_model,
        features_scaled,
        labels,
        set_name='evaluation' 
    )

    # Print Results
    print("\n"+"="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {pipeline.trainer.best_model_name}")
    print(f"Dataset: {args.data}")
    print(f"Samples: {len(labels)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precission: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("="*80 + "\n")


    # Plot Results
    trainer.plot_confusion_matrix(
        pipeline.trainer.best_model,
        features_scaled,
        labels,
        title = "Evaluation Confusion Matrix"
        )
    
def extract_mode(args, config):
    """Execute feature extracting only mode"""
    logging.info("Starting feature extraction mode")

    if not args.data:
        raise ValueError("--data is required for extraction mode")
    
    from feature_extraction import RadiomicsExtractor

    extractor = RadiomicsExtractor(config)
    features_df, labels, _ = extractor.extract_from_dictionary(Path(args.data))

    # Save features
    output_dir = Path(args.output) if args.output else config.RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    features_path = output_dir / 'exacted_features.csv'
    features_df.to_csv(features_path)

    labels_path = output_dir/"labels.csv"

    import pandas as pd

    pd.DataFrame({'labels': labels}).to_csv(labels_path, index=False)

    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"Samples: {features_df.shape[0]}")
    print(f"Features: {features_df.shape[1]}")
    print(f"Classes distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Features saved to: {features_path}")
    print(f"Labels saved to: {labels_path}")
    print("="*80 + "\n")

def main():
    """Main function"""
    parser = setup_argparse()
    args = parser.parse_args()

    # Load configuration
    if args.config:
        # TODO: Implement custom config loading
        logging.warning("Custom config loading not yet implemeted, using default")

    config = Config()

    # Update output directory if specified
    if args.output:
        config.RESULTS_DIR = Path(args.output)
        config.RESULTS_DIR.mkdir(parents=True,exist_ok=True)

    # Update logging level
    config.LOGGING['level'] = args.log_level

    # Setup logging
    logging.basicConfig(
        level = getattr(logging, args.log_level),
        format = config.LOGGING['format'],
        handlers=[
            logging.FileHandler(config.LOGGING['file']),
            logging.StreamHandler()
        ]
    )

    try:
        #Execute base on mode
        if args.mode == "train":
            train_mode(args=args,config=config)

        elif args.mode == "predict":
            predict_mode(args=args, config=config)
        
        elif args.mode == "evaluate":
            evaluate_mode(args=args, config=config)
        
        elif args.mode == "extract":
            extract_mode(args=args, config=config)
        
        else:
            parser.error(f"Unknown mode: {args.mode}")

        logging.info("Operation completed successfully")
        sys.exit(0)

    except Exception as e:
        logging.error(f"Operation failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    import numpy as np 
    main()






    






