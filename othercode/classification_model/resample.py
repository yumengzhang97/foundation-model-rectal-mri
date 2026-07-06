import pandas as pd
import SimpleITK as sitk
import glob
import os
import numpy as np

path = r"/home/chaimeleon/persistent-home/classificationmodel_original_sagittal"
train_paths = glob.glob(os.path.join(path, 'train', '*.nii.gz'))
val_paths = glob.glob(os.path.join(path, 'val', '*.nii.gz'))
test_paths = glob.glob(os.path.join(path, 'test', '*.nii.gz'))
image_paths =train_paths + val_paths + test_paths


def resample_image(img, target_spacing, is_label=False):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
  
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    if is_label:
        # Use nearest neighbor interpolation for segmentation masks
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Use linear interpolation or other suitable methods for images
        resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(img)

def generate_path(path, mask_chosen):
    filename = os.path.basename(path)
    name_without_ext = filename.replace('.nii.gz', '')
    
    if 'train' in path:
        data_type = 'train'
    elif 'val' in path:
        data_type = 'val'
    elif 'test' in path:
        data_type = 'test'

   
    path_mask = os.path.join(
        '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal',
        'total_work',
        data_type,
        name_without_ext,
        mask_chosen,
    )
    return path_mask


# Define the target spacing
new_spacing = (3.56, 0.43, 0.43)  # Adjust as needed
save_path = r'/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/resampled'


# Read the original image and segmentation mask
for img_path in image_paths:
    #find male cases in train and val set
    basename = os.path.basename(img_path)
    prefix = basename[:4] #id
 
    original_image = sitk.ReadImage(img_path)
    path_colon = generate_path(img_path, 'colon.nii.gz')
    colon_image = sitk.ReadImage(path_colon)
        
    # Resample the original image
    resampled_image = resample_image(original_image, new_spacing, is_label=False)      
    # Resample the segmentation mask
    resampled_colon = resample_image(colon_image, new_spacing, is_label=True) 

    new_folder = os.path.join(save_path, basename)
    os.makedirs(new_folder, exist_ok=True)
        
    save_path1 = os.path.join(new_folder, 'resampled_image.nii.gz')
    save_path3 = os.path.join(new_folder, 'colon.nii.gz')
        
    # Save the resampled image and mask
    sitk.WriteImage(resampled_image, save_path1)
    sitk.WriteImage(resampled_colon, save_path3)

print('done')