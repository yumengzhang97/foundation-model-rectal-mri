import os
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


input_dir = r"/home/chaimeleon/persistent-home/classification_model_originalimage/train_data"
output_root_dir = r"/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/train"


if __name__ == "__main__":

    os.environ["TOTALSEG_HOME_DIR"] = "/home/chaimeleon/persistent-home/.totalsegmentator"

    for file in os.listdir(input_dir):

        input_image_path = os.path.join(input_dir, file)
            
        image_name = os.path.splitext(os.path.splitext(file)[0])[0]  # .nii.gz
        output_folder = os.path.join(output_root_dir, image_name)

        os.makedirs(output_folder, exist_ok=True)

        try:

            # option 1: provide input and output as file paths
            totalsegmentator(input_image_path, output_folder, task="total_mr")
                    
            # option 2: provide input and output as nifti image objects
            input_img = nib.load(input_image_path)
            output_img = totalsegmentator(input_image_path)
            nib.save(output_img, output_folder)
                        
            print(f"start to segment: {input_image_path} -> {output_folder}")

        except Exception as e:

            print(f"error{e}: can not segment{input_image_path}")
                
    print("finish")
