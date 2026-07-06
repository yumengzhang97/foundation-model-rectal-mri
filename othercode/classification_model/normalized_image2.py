import os
import SimpleITK as sitk
import numpy as np

def robust_normalize_per_image(
    img: sitk.Image,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5
) -> sitk.Image:
    """
    Perform per-image robust intensity normalization:
    1) Convert to NumPy
    2) Compute lower/upper percentiles
    3) Clip intensities
    4) Z-score normalize using mean/std of clipped intensities
    5) Convert back to SimpleITK Image
    """

    img_array = sitk.GetArrayFromImage(img)  # shape: [z, y, x] in SITK (assuming 3D)
    
    # Flatten to compute percentiles over entire volume
    flat = img_array.flatten()
    
    # Compute the lower/upper intensities at the given percentiles
    lower_val = np.percentile(flat, lower_percentile)
    upper_val = np.percentile(flat, upper_percentile)
    
    # Clip
    clipped = np.clip(img_array, lower_val, upper_val)
    
    # Mean and std of the clipped data
    mean_val = clipped.mean()
    std_val  = clipped.std()
    
    # Prevent divide-by-zero
    if std_val < 1e-8:
        std_val = 1e-8
    
    # Z-score normalization
    normalized = (clipped - mean_val) / std_val
    
    # Convert back to SimpleITK Image (maintain original metadata)
    out_img = sitk.GetImageFromArray(normalized)
    out_img.CopyInformation(img)
    
    return out_img

# Example usage for a single image:
# img = sitk.ReadImage("some_image.nii.gz")
# norm_img = robust_normalize_per_image(img)

# If you want to do this for a folder of images:
def normalize_folder_per_image(input_folder, output_folder):
    """Normalize each image in 'input_folder' separately and save to 'output_folder'."""
    os.makedirs(output_folder, exist_ok=True)
    
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith((".nii", ".nii.gz")):
            continue
        img_path = os.path.join(input_folder, fname)
        img = sitk.ReadImage(img_path)
        
        # Normalize
        norm_img = robust_normalize_per_image(img, 2.5, 97.5)
        
        # Save
        out_path = os.path.join(output_folder, fname)
        sitk.WriteImage(norm_img, out_path)
        print(f"Saved normalized image to {out_path}")

input_folder = r'/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/test'

output_folder = r"/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/normalized/test"  # Replace with your output directory

normalize_folder_per_image(input_folder, output_folder)