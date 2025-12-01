""" 
Feature extraction module using PyRadiomics (SOUP)
"""

import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from radiomics import featureextractor
import six
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RadiomicsExtractor:
    """Extract radiomics features from medical images"""

    def __init__(self, config):
        self.config = config
        self.extractor = None
        self.initialize_extractor()

    def initialize_extractor(self):
        """Initialize PyRadiomics feature extractor"""
        # Supress PyRadiomics warnings
        logging.getLogger("radiomics").setLevel(logging.ERROR)
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        # Initialize extractor with settings
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            **self.config.RADIOMICS_SETTINGS
        )

        # Enable all feature classes
        self.extractor.enableAllFeatures()

        # Enable/disable image types
        for image_type, enabled in self.config.IMAGE_TYPES.items():
            self.extractor.enableImageTypeByName(image_type, enabled=enabled)

        logger.info("Radiomics extractor initialized")

    def extract_features_single(self, image: sitk.Image,
                                mask: sitk.Image) -> Dict:
        """ 
        Extract features from a single image-mask pair
        
        Args:
            image: SimpleITK image
            mask: SimpleITK mask
        
        Retuns:
            Dictionary of extracted features
        """
        try:
            result = self.extractor.execute(image, mask)

            # Filter out diagnosis features
            features = {}
            for key, value in six.iteritems(result):
                if isinstance(value, (np.integer, np.floating)):
                    features[key] = float(value)
                else:
                    features[key] = value
                
            return features

        except Exception as e:
            logger.error(f"Error extractiong features: {str(e)}")
            raise

    def extract_from_paths(self, image_path: Path,
                           mask_path:Path) -> Dict:
        """ 
        Extract features from file paths
        
        Args:
            image_path: Path to image file
            mask_path: Path to mask file
        
        Returns:
            Dictionary of extracted features
        """
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        return self.extract_features_single(image, mask)
    
    def batch_extract(self, data_list: List[Dict],
                      use_preprocessing:bool = True) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """ 
        Extract features from multiples subjects
        
        Args:
            data_list: List of dictionaries with 'image_path', 'mask_path', 'label' 
            use_preprocessing: Whether to apply preprocessing
        
        Returns:
            Tuple of (features_df, labels_array, subject_info_dict)
        """
        from preprocessing import ImagePreprocessor

        if use_preprocessing:
            preprocessor = ImagePreprocessor(self.config)

        features_dict = {}
        labels = []
        subject_info = {}
        failed_subject = []

        logger.info(f"Extracting features from {len(data_list)} subjects...")

        for idx, data in enumerate(tqdm(data_list, desc = "Extracting features")):
            try:
                subject_id = data.get("subject_id", f"subject_{idx}")
                image_path = Path(data['image_path'])
                mask_path = Path(data['mask_path'])
                label = data['label']

                # Preprocessing if request
                if use_preprocessing:
                    image, mask = preprocessor.preprocess_image_pair(
                        image_path, mask_path
                    )
                else:
                    image = sitk.ReadImage(str(image_path))
                    mask = sitk.ReadImage(str(mask_path))

                # Extract features
                features = self.extract_features_single(image, mask)

                # Store results
                features_dict[str(idx)] = features
                labels.append(label)
                subject_info[str(idx)] = {
                    'subject_id': subject_id,
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                    'label': label
                }    

            except Exception as e:
                logger.error(f"Failed to process {data.get('subject_id', idx)}: {str(e)}")
                failed_subject.append(data.get('subject_id',idx))

        if failed_subject:
            logger.warning(f"Failed to process {len(failed_subject)} subjects: {failed_subject}")
        
        # Convert to DF
        features_df = pd.DataFrame(data=features_dict).T
        labels_array = np.array(labels)

        logger.info(f"Successfully extracted features from {features_df.shape[0]} subjects")
        logger.info(f"Number of features: {features_df.shape[1]}")
        logger.info(f"Class distribution: {np.bincount(labels_array)}")

        return features_df, labels_array, subject_info
    
    def extract_from_dictionary(self, dataset_path: Path)-> Tuple[pd.DataFrame, np.array, Dict]:
        """ 
        Extract features from a structured dictionary
        
        Expected structure:
        dataset_path/
            diagnosis_1/
                subjet_1/
                    T1/T1.nii.gz
                    label/label.nii.gz
                subject_2/
                    ...
            diagnosis_2/
                ...
        
        Args:
            dataset_path: Path to dataset dictionary
        
        Returns:
            Tuple of (features_df, labels_array, subject_info_dict)
        """

        data_list = []
        structure = self.config.EXPECTED_STRUCTURE

        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Iterate through diagnosis folders
        for diagnosis_dir in dataset_path.iterdir():
            if not diagnosis_dir.is_dir():
                continue
                
            diagnosis = diagnosis_dir.name
            label = self.config.LABELS.get(diagnosis, self. config.LABELS.get(diagnosis.upper(), -1))

            if label == -1:
                logger.warning(f"Unknown diagnosis: {diagnosis}, skipping ...")
                continue

            logger.info(f"Processing {diagnosis} dataser (label={label})")

            # Iterate through subjects
            for subject_dir in diagnosis_dir.iterdir():
                if not subject_dir.is_dir():
                    continue

                subject_id = f"{diagnosis}_{subject_dir.name}"

                # Build paths
                image_path = subject_dir / structure['image_folder'] / structure['image_file']
                mask_path = subject_dir / structure['label_folder'] / structure['label_file']

                # Validate paths
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                if not mask_path.exists():
                    logger.warning(f"Mask not found: {mask_path}")
                    continue

                data_list.append({
                    'subject_id': subject_id,
                    'image_path':str(image_path),
                    'mask_path': str(mask_path),
                    'label': label
                })
        
        logger.info(f"Found {len(data_list)} valid subjects")

        return self.batch_extract(data_list=data_list)
    
    def get_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """ 
        Group features by type (image type and feature class)
        
        Args: 
             feature_names: List of feature names
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        
        groups = {}

        # Image type
        image_types = ['original', 'wavelet', 'log', 'square', 'squareroot',
                       'logarithm', 'exponential', 'gradient', 'lbp-3d']
        
        for img_type in image_types:
            groups[img_type] = [f for f in feature_names if f.lower().startswith(img_type)]
            
        # Feature classes
        feature_classes = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape']

        for feat_class in feature_classes:
            groups[feat_class] = [f for f in feature_names if feat_class in f.lower()]

        return groups
    
    def analyze_feature_distribution(self, features_df: pd.DataFrame,
                                     lables: np.ndarray,
                                     save_path: Optional[Path] = None):
        """ 
        Analyze and visualize feature distributions
        
        Args:
            features_df: Feature dataFrame
            labels: Label array
            save_path: Path to save analysis
        """

        import matplotlib.pyplot as plt

        # Group features
        groups = self.get_feature_groups(features_df.columns.tolist())

        # Count features per group
        group_counts = {k: len(v) for k, v in groups.items() if v}

        # Plot
        fig, axes = plt.subplots(1, 2, figsize = (15,5))

        # Feature counts by image type
        image_type_counts = {k: v for k, v in group_counts.items()
                             if k in ['original', 'wavelet', 'log', 'square',
                                      'squareroot', 'logarithm', 'exponential',
                                      'gradient', 'lbp-3d']}
        
        axes[0].bar(image_type_counts.keys(), image_type_counts.values())
        axes[0].set_xlabel('Image Type')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Features by Image Type')
        axes[0].tick_params(axis='x', rotation=45)

        # Feature counts by feature class
        feature_class_counts = {k: v for k, v in group_counts.items()
                                if k in ['firstorder', 'glcm', 'glrlm', 'glszm',
                                         'gldm', 'ngtdm', 'shape']}
        
        axes[1].bar(feature_class_counts.keys(), feature_class_counts.values())
        axes[1].set_xlabel('Feature Class')
        axes[1].set_ylabl('Number of Features')
        axes[1].set_title('Features by Feature Class')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        # Log statistics
        logger.info("\nFeature extraction statistics:")
        logger.info(f"Total features: {features_df.shape[1]}")
        for group, count in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                logger.info(f"  {group}: {count} features")
        


