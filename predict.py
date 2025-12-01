""" 
    Prediction module for classifiying new medical images
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from pipeline import RadiomicsPipeline

logger = logging.getLogger(__name__)

class RadiomicsPredictor:
    """Predic on new medical images using trained pipeline"""

    def __init__(self, pipeline_path: Path):
        """ 
        Initialize predictor with trained pipeline
        
        Args:
            pipeline_path: Path to saved pipeline file
        
        """

        self.pipeline = RadiomicsPipeline()
        self.pipeline.load_pipeline(pipeline_path)
        logger.info("Predictor initialized with trained ipeline")

    def predict_single(self, image_path: Path, mask_path: Path,
                       save_results: bool = False,
                       output_path: Optional[Path] = None) -> Dict:
        """ 
        Predict on a sigle image-mask pair
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            save_results: Whether to save results to file
            output_path: Path to save results (optional)
            
        Returns:
            Dictionary with prediction results
        """

        result = self.pipeline.predict_new_image(image_path, mask_path)

        # Add metadata
        result['image_path'] = str(image_path)
        result['mask_path'] = str(mask_path)
        result['timestamp'] = datetime.now().isoformat()

        if save_results:
            if output_path is None:
                output_path = Path(f"prediction_{image_path.stem}.json")

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")

        return result
    
    def predict_batch(self, data_list: List[Dict],
                      save_results: bool = True,
                      output_path: Optional[Path] = None) -> pd.DataFrame:
        """ 
        Predict on multiple images
        
        Args:
            data_list: List of dicts with 'image_path' and 'mask_path'
            save_results: Whether to save results
            output_path: Path to save results CSV

        Returns:
            DataFrame with predictions

        """

        results = []

        logger.info(f"Processing {len(data_list)} images...")

        for i, data in enumerate(data_list):
            try:
                image_path = Path(data['image_path'])
                mask_path = Path(data['mask_path'])
                subject_id = data.get('subject_id', f'subject_{i}')

                result = self.predict_single(image_path=image_path, mask_path=mask_path)
                result['subject_id'] = subject_id
                results.append(result)

                logger.info(f"[{i+1}/{len(data_list)}] {subject_id}: "
                            f"{result['prediction_label']} "
                            f"({result['confidence']:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {data.get('subject_id', i)}: {str(e)}")
                results.append({
                    'subject_id': data.get('subject_id', f'subject_{i}'),
                    'error': str(e),
                    'prediction': None
                })
            
            # COnvert to DataFrame
            results_df = pd.DataFrame(results)

            if save_results:
                if output_path is None:
                    output_path = Path('batch_predictions.csv')

                results_df.to_csv(output_path, index=False)
                logger.info(f"Batch results saved to {output_path}")
            
            return
        
    def predcit_from_dictionary(self, dictionary:Path,
                                image_pattern: str = '**/T1/T1.nii.gz',
                                mask_pattern: str = '**/label/label.nii.gz',
                                save_results: bool = True) -> pd.DataFrame:
        """ 
        Predict on all images in a dictionary structure

        Args:
            dictionary: root dictionary containing images
            image_pattern: Glob pattern for image files
            mask_pattern: GLob pattern for mask files
            save_results: Whether to save results

        Returns:
            DataFrame with predictions
        """

        # Find all images
        image_paths = sorted(dictionary.glob(image_pattern))

        logger.info(f"Found {len(image_paths)} images in {dictionary}")

        data_list = []

        for image_path in image_paths:
            # Construct corresponding mask path
            subject_dir = image_path.parent.parent
            mask_path = subject_dir / 'label' / 'label.nii.gz'

            if mask_path.exists():
                data_list.append({
                    'subject_id': subject_dir.name,
                    'image_path': str(image_path),
                    'mask_path': str(mask_path)
                })
            
            else:
                logger.warning(f"Mask not found for {image_path}")

            return self.predict_batch(data_list, save_results=save_results)

    def get_prediction_report(self, results_df: pd.DataFrame) -> str:
        """ 
        Generate a summary report from prediction results
        
        Args:
            results_df: DataFrame with prediction results
            
        Returns:
            Report string
        """

        # Filter successful predictions
        valid_results = results_df[results_df['prediction'].notna()]
        
        n_total = len(results_df)
        n_valid = len(valid_results)
        n_errors = n_total-n_valid

        if n_valid > 0:
            n_ad = (valid_results['prediction'] == 1).sum()
            n_cn = (valid_results['prediction'] == 0).sum()
            avg_confidence = valid_results['confidence'].mean()

        else:
            n_ad = n_cn=avg_confidence = 0
        
        report = f"""
        PREDICTION REPORT
        =================
        
        Total images: {n_total}
        Successfully processed: {n_valid}
        Errors: {n_errors}
        
        Predictions:
        - Alzheimer's Disease (AD): {n_ad} ({n_ad/n_valid*100:.1f}%)
        - Cognitively Normal (CN): {n_cn} ({n_cn/n_valid*100:.1f}%)
        
        Average confidence: {avg_confidence:.2%}
        
        Model used: {results_df['model_used'].iloc[0] if n_valid > 0 else 'N/A'}
        """

        return report
    
    def visualize_predictions(self, results_df: pd.DataFrame,
                              save_path: Optional[Path] = None):
        """ 
        Create visualizationsof prediction results
        
        Args:
            results_df: DataFrame with predictions
            save_path: Path to save figure
            
        """

        import matplotlib.pyplot as plt

        valid_results = results_df[results_df['prediction'].notna()]

        if len(valid_results) == 0:
            logger.warning("No valid predictions to visualizate")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        
        # 1. Prediction distribution
        pred_counts = valid_results['prediction_label'].value_counts()
        axes[0].bar(pred_counts.index, pred_counts.values, color=['lightblue', 'coral'])
        axes[0].set_xlabel('Diagnosis')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Prediction Distribution')
        
        # 2. Confidence distribution
        axes[1].hist(valid_results['confidence'], bins=20, color='skyblue', edgecolor='black')
        axes[1].axvline(valid_results['confidence'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {valid_results["confidence"].mean():.2f}')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution')
        axes[1].legend()
        
        # 3. Probability scatter
        axes[2].scatter(valid_results['probability_CN'], 
                       valid_results['probability_AD'],
                       c=valid_results['prediction'],
                       cmap='coolwarm', alpha=0.6, s=50)
        axes[2].plot([0, 1], [1, 0], 'k--', alpha=0.3)
        axes[2].set_xlabel('P(CN)')
        axes[2].set_ylabel('P(AD)')
        axes[2].set_title('Prediction Probabilities')
        axes[2].set_xlim([0, 1])
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prediction Alzheimer\'s Disease from medical images'
    )
    parser.add_argument('--pipeline', type=str, required=True,
                       help='Path to trained pipeline file')
    parser.add_argument('--image', type=str,
                       help='Path to image file (for single prediction)')
    parser.add_argument('--mask', type=str,
                       help='Path to mask file (for single prediction)')
    parser.add_argument('--directory', type=str,
                       help='Directory containing images (for batch prediction)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    args = parser.parse_args()

    # Initialize predictior
    predictor = RadiomicsPredictor(Path(args.pipeline))

    # Single prediction
    if args.image and args.mask:
        result = predictor.predict_single(
            Path(args.image),
            Path(args.mask),
            save_results=True,
            output_path=Path(args.output)
        )

        print("\nPrediction Result:")
        print(f"  Diagnosis: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  P(CN): {result['probability_CN']:.2%}")
        print(f"  P(AD): {result['probability_AD']:.2%}")

    # Batch prediction
    elif args.directory:
        results_df = predictor.predict_from_directory(
            Path(args.directory),
            save_results=True
        )
        
        # Print report
        report = predictor.get_prediction_report(results_df)
        print(report)
        
        # Visualize if requested
        if args.visualize:
            predictor.visualize_predictions(
                results_df,
                save_path=Path(args.output).with_suffix('.png')
            )
    
    else:
        parser.error("Either --image and --mask, or --directory must be provided")


if __name__ == '__main__':
    main()
