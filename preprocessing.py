""" 
Preprocessing module for medical image
Handles validation, normalization and fixing of NIfTI files
"""

import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles preprocessing of medical images"""

    def __init__(self, config):
        self.config = config
        self.preprocessing_settings = config.PREPROCESSING

    def validate_nifti(self, image_path: Path) -> Dict:
        """ 
        Validate NIfTI file and return information about any issues

        Args:
            image_path: Path to NIFTI file

        Returns:
            Dictionary with validation results
        """

        validation_results={
            'valid': True,
            'warnings': [],
            'errors': []
        }

        try:
            img = nib.load(str(image_path))

            # Check affine matrix
            affine = img.affine
            if not self._is_affine_orthonormal(affine):
                validation_results['warnings'].append('Non-orthonormal affine matrix')

            # Check data integrity
            data = img.get_fdata()
            if np.any(np.isnan(data)):
                validation_results['warnings'].append('NaN values present')
            if np.any(np.isinf(data)):
                validation_results['warnings'].append('Infinite values present')

            # Check dimensions
            if len(data.shape) != 3:
                validation_results['error'].append(f'Expected 3D image, got {len(data.shape)}D')
                validation_results['valid'] = False
            
            # Check spacing
            pixdim = img.header['pixdim'][1:4]
            if np.any(pixdim<=0):
                validation_results['warnings'].append('Invalid pixel dimensions')

        except Exception as e:
            validation_results['error']. append(f'Failed to load image: {str(e)}')
            validation_results['valid'] = False

        return validation_results
    
    def fix_nifti(self, image_path: Path, output_path: Optional[Path]=None)->Path:
        """ 
        Fix common issues in NIfTI files
        
        Args:
            image_path: Path to input NIfTI file
            output_path: Path for fixed file (if None, overwrites original)
            
        Returns:
            Path to fixed file
        """

        try:
            img = nib.load(str(image_path))
            affine = img.affine
            data = img.get_fdata()

            #Fix NaN and Inf values
            data = np.nan_to_num(data, nan=0.0, posinf=data[~np.isinf(data)].max(),
                                 neginf=data[~np.isinf(data)].min())
            
            # Fix affine matrix if needed
            if not self._is_affine_orthonormal(affine):
                logger.warning(f"Fixing non-orthonormal affine in {image_path}")
                affine = self._fix_affine_matrix(affine)
            
            # Create fixed image
            fixed_img = nib.Nifti1Image(data, affine,img.header)

            # Save
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_fixed.nii.gz"
            
            nib.save(fixed_img, str(output_path))
            logger.info(f"Fixed image saved to {output_path}")

            return output_path
        
        except Exception as e:
            logger.error(f"Error fixing {image_path}: {str(e)}")
            raise

    def normalize_intensity(self, image_path: Path, method: str = 'zscore',
                            mask_path: Optional[Path] = None) -> sitk.Image:
        """ 
        Normalize image intensity
        
        Args:
            image_path: Path to image
            method: Normalized method ('zscore', 'minmax', 'percentile')
            mask_path: Optional mask to compute statistics only with ROI
            
        Returns:
            Normalized SimpleITK image
        """

        img = sitk.ReadImage(str(image_path))
        img_array = sitk.GetArrayFromImage(img)

        # Get mask if provided
        if mask_path is not None:
            mask = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask)
            roi_values = img_array[mask_array > 0]
        else:
            roi_values = img_array[img_array > 0]
        
        if method == 'zscore':
            mean = np.mean(roi_values)
            std = np.std(roi_values)
            normalized = (img_array - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            min_val = np.min(roi_values)
            max_val = np.max(roi_values)
            normalized = (img_array- min_val) / (max_val -min_val + 1e-8)
        
        elif method == 'percentile':
            p1, p99 = np.percentile(roi_values, [1, 99])
            normalized = np.clip((img_array - p1) / (p99 - p1 +1e-8), 0, 1)
        
        else:
            raise ValueError(f"Unknow normalization method: {method}")
        
        # Create normalized images
        normalized_img = sitk.GetImageFromArray(normalized)
        normalized_img.CopyInformation(img)

        return normalized_img
    
    def check_image_mask_aligment(self, image_path: Path, mask_path: Path) -> bool:
        """ 
        Check if image and mask are properly aligned
        
        Args:
            image_path: Path to image
            mask_image: Path to mask
            
        Returns:
            True if aligned, False otherwise
        """

        try:
            img = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            # Check dimentions
            if img.GetSize()!= mask.GetSize():
                logger.error(f"Size mismatch: Image {img.GetSize()} vs Mask {mask.GetSize()}")
                return False
            
            # Check spacing
            if not np.allclose(img.GetSpacing(), mask.GetSpacing(), rtol = 1e-3):
                logger.warning(f"Spacing mismatch: Image {img.GetSpacing()} vs Mask {mask.GetSpacing()}")

            if not np.allclose(img.GetOrigin(), mask.GetOrigin(), rtol=1e-3):
                logger.warning(f"Origin mismatch: Image {img.GetOrigin()} vs Mask {mask.GetOrigin()}")
            
            # Check direction
            if not np.allclose(img.GetOrigin(), mask.GetDirection(), rtol=1e-3):
                logger.warning("Direction mismatch")
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking aligment: {str(e)}")
            return False
        
    def resample_to_reference(self, moving_image_path: Path,
                              reference_image_path: Path,
                              output_path: Path,
                              interpolator: str = 'linear') -> sitk.Image:
        """" 
        Resample moving image to mathc reference image geometry
        
        Args:
            moving_image_path: Image to resample
            reference_image_path: Reference image
            output_path: Where to save resampled image
            interpolator: 'linear', 'nearest', 'bspline'
            
        Returns:
            Resampled image
        """

        moving = sitk.ReadImage(str(moving_image_path))
        reference = sitk.ReadImage(str(reference_image_path))

        # Choose interpolator
        interp_map = {
            'linear': sitk.sitkLinear,
            'nearest': sitk.sitkNearestNeighbor,
            'bspline': sitk.sitkBSpline
        }

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interp_map.get(interpolator, sitk.sitkLinear))
        resampler.SetDefaultPixelValue(0)

        resampled = resampler.Exectute(moving)

        sitk.WriteImage(resampled, str(output_path))
        logger.info(f"Resampled image saved to {output_path}")

        return resampled
    
    def preprocess_image_pair(self, image_path: Path, mask_path:Path,
                              temp_dir: Optional[Path] = None) -> Tuple[sitk.Image, sitk.Image]:
        """ 
        Complete preprocessing pipeline for image-mask pair
        
        Args:
            image_path: Path to image
            masl_path: Path to mask
            temp_dir: Dictionary for temporary files
            
        Returns:
            Tuple of (preprocessed_image, preprocessed_mask)
        """

        logger.info(f"Preprocessing {image_path.name}")

        # Validate
        if self.preprocessing_settings['validate_nifti']:
            img_validation = self.validate_nifti(image_path=image_path)
            mask_validarion = self.validate_nifti(image_path=mask_path)

            if not img_validation['valid'] or not mask_validarion['valid']:
                raise ValueError("Invalid NIfTI files")
            
            # Fix if needed
            if self.preprocessing_settings['fix_affine']:
                if img_validation['warnings']:
                    image_path = self.fix_nifti(image_path, temp_dir / f"fixed_{image_path.name}" if temp_dir else None)
                if img_validation['warnings']:
                    mask_path = self.fix_nifti(mask_path, temp_dir / f"fixed_{mask_path.name}" if temp_dir else None)

        # Check alignment
        if self.preprocessing_settings['check_orientation']:
            if not self.check_image_mask_aligment(image_path=image_path, mask_path=mask_path):
                logger.warning("Image and mask may not be properly aligned")

        # Normalize intensity
        if self.preprocessing_settings['intensity_normalization']:
            logger.info("Image and mask intensity normalization")

            img = self.normalize_intensity(image_path=image_path, method='percentile', 
                                            mask_path=mask_path)
            
            # Save normalized image to temp location if needed
            if temp_dir:
                norm_path = temp_dir / f"norm_{image_path.name}"
                sitk.WriteImage(img, str(norm_path))
                img = sitk.ReadImage(str(norm_path))

            else:
                img = sitk.ReadImage(str(image_path))
        else:
            img = sitk.ReadImage(str(image_path))

        mask = sitk.ReadImage(str(mask_path))

        return img, mask
    
    @staticmethod
    def _is_affine_orthonormal(affine: np.ndarray, tolerance: float = 1e-4) -> bool:
        """Check is affine matrix is orthonormal"""
        rotation = affine[:3, :3]
        product = np.dot(rotation, rotation.T)
        identity = np.eye(3)

        return np.allclose(product, identity, atol = tolerance)
    
    @staticmethod
    def _fix_affine_matrix(affine: np.ndarray) -> np.ndarray:
        """Fix non-orthonormal affine matrix using QR descomposition"""
        fixed_affine = np.eye(4)
        fixed_affine[:3,:3] = np.linalg.qr(affine[:3,:3])[0]
        fixed_affine[:3,3] = affine[:3,3]

        return fixed_affine

            


        


        



        


