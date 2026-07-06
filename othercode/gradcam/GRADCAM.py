import subprocess, sys

required  = {'wheel','setuptools','torch==2.1.2','torchvision==0.16.2','torchaudio==2.1.2','mmdet','mmcv==2.1.0','medicalmultitaskmodeling','monai','tensorboard','SimpleITK','torch==2.1.2','torchvision==0.16.2','torchaudio==2.1.2'}

subprocess.check_call([sys.executable, '-m', 'pip', 'install','--no-index','--find-links','/home/chaimeleon/persistent-home/UMedPt/FM/FM', *required])

import os
import torch
from monai.data import DataLoader, Dataset,ArrayDataset,decollate_batch
from monai.utils import set_determinism
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

set_determinism(6339391)
os.environ['MMM_LICENSE_ACCEPTED'] = 'i accept'

from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.labelstudio_ext.NativeBlocks import MMM_MODELS

Prefix = 'FM_axial_harmo'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### MODEL #####
# stolen directly from their source code and rewritten to fit the current context
class SmartHead(torch.nn.Module): # Simple MLP head
    def __init__(self, hidden_dim, out_dim, dropout):
        super(SmartHead, self).__init__()
        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(
                max(out_dim * 4, hidden_dim),
                max(out_dim * 4, hidden_dim // 2),
            ),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.classification_head(x)

# # combining the grouper and classifier for convenience
class CombinedModel(torch.nn.Module):
    def __init__(self, grouper, classifier): # supercase_indices should be used at runtime
        super(CombinedModel, self).__init__()                  # hardcoded for simplicity at the intilization stage
        self.grouper = grouper                                 # this will restrict parallel processing
        self.classifier = classifier                           # at this stage, I don't care about speed

    def forward(self, hidden_vector,supercase_indices):
        grouper_vector, weights = self.grouper(hidden_vector, supercase_indices)
        output = self.classifier(grouper_vector)
        return output, weights


class GradCAMFriendlyModel(torch.nn.Module):
    def __init__(self, encoder, squeezer, grouper, classifier):
        super().__init__()
        self.encoder = encoder
        self.squeezer = squeezer
        self.grouper = grouper
        self.classifier = classifier
        self.flatten = torch.nn.Flatten(1)

    def forward(self, x):
        print("x input shape", x.shape) #1，1，192，192，36
        batch_size, img_depth = x.shape[0], x.shape[4]
        x = x.permute(0, 4, 1, 2, 3) #1,36,1,192,192 from (B, C, H, W, D) into (B, D, C, H, W)
        # print("img_permuted",x.shape)

        x = x.reshape(batch_size * img_depth, x.shape[2], x.shape[3], x.shape[4]) #36，1，192，192
        # from (B, D, C, H, W) to (B*D, C, H, W)
        # Repeat single channel to 3 channel image.
        x = x.repeat(1, 3, 1, 1)
        print("x after reshape",x.shape) #36，3，192，192

        supercase_indices = (
            torch.arange(batch_size).repeat_interleave(img_depth).to(x.device)
        )

        feat_pyramid = self.encoder(x)
        squeezed = self.squeezer(feat_pyramid)[1] 
        hidden_vector = self.flatten(squeezed)

        group_vec, _ = self.grouper(hidden_vector, supercase_indices)
        output = self.classifier(group_vec)
        return output

MMM_MODELS = {"encoder-1.0.4.pt.zip": "/home/chaimeleon/persistent-home/UMedPt/FM/encoder-1.0.4.pt.zip"}
fmEncoder = NativeBlocks(MMM_MODELS[DEFAULT_MODEL])
embedding_dim=512

fmGrouper1 = Grouper(Grouper.Config(module_name="grouper", version="weighted"),
                                embedding_dim=embedding_dim,)
fmClassifier1 = SmartHead(hidden_dim=embedding_dim,out_dim=1,dropout=0.5)
model1 = CombinedModel(fmGrouper1, fmClassifier1).to(device)

# checkpoint1 = torch.load(r"/home/chaimeleon/persistent-home/classificationmodel_harmonizednew/axial_harmonize/rectum_centrecrop2/models_FM/FM_axial_harmoevi-checkpoint_latest.pth")
checkpoint1 = torch.load(r"/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/normalized/models_FM/FM_192evi-checkpoint_latest.pth")
model1.load_state_dict(checkpoint1["model_state_dict"])


fmGrouper2 = Grouper(Grouper.Config(module_name="grouper", version="weighted"),
                        embedding_dim=embedding_dim,)
fmClassifier2 = SmartHead(hidden_dim=embedding_dim,out_dim=1,dropout=0.5)
model2 = CombinedModel(fmGrouper2, fmClassifier2).to(device)

# checkpoint2 = torch.load(r"/home/chaimeleon/persistent-home/classificationmodel_harmonizednew/axial_harmonize/rectum_centrecrop2/models_FM/FM_axial_harmomfi-checkpoint_latest.pth")
checkpoint2 = torch.load(r"/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/normalized/models_FM/FM_192mfi-checkpoint_latest.pth")
# model2.load_state_dict(checkpoint2["model_state_dict"])


model_cam1 = GradCAMFriendlyModel(encoder=fmEncoder["encoder"], squeezer=fmEncoder["squeezer"], grouper=model1.grouper, classifier=model1.classifier).to(device)
# model_cam2 = GradCAMFriendlyModel(encoder=fmEncoder["encoder"], squeezer=fmEncoder["squeezer"], grouper=model2.grouper, classifier=model2.classifier).to(device)
model_cam1.eval()
# model_cam2.eval()

for name, param in model1.named_parameters():
    print(f"{name}: {param.shape}")


#### DATA PART ######
def app_int(risk):
    risk = str(risk).strip().lower()
    if risk == 'true':
        return 1
    else:
        return 0

class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        # print(csv_file)
        self.subject_id = self.data['Subjects'].tolist()
        self.labels1 = self.data['EVI'].apply(app_int).tolist()
        self.labels2 = self.data['MFI'].apply(app_int).tolist()

        # print(len(self.subject_id))

    def __len__(self):
        return len(self.subject_id)

    def __getitem__(self, index):

        image = os.path.join(self.image_dir, self.subject_id[index]+'.nii.gz')
        evi_label = self.labels1[index] #EVI
        mfi_label = self.labels2[index] #MFI

        labels = {'EVI':evi_label, 'MFI': mfi_label}
        
        return {'image': image,'label':labels}

test_dir = r"/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/normalized/test/"
test_labels_dir = r'/home/chaimeleon/persistent-home/classification_model_originalimage/totalsegment_work/resampled/centrecrop_correct2/normalized/test_labels.csv'

# test_dataset = CustomDataset(image_dir=test_dir, csv_file=test_labels_dir)
# sample = test_dataset[0]
# print("img_path:", sample["image"])
# print("labels:", sample["label"])
# label_EVI = torch.Tensor(sample["label"]['EVI']).to(device)
# label_EVI = torch.Tensor(sample["label"]['MFI']).to(device)

# img_np =sitk.GetArrayFromImage(sitk.ReadImage(sample["image"]))
# img_new = img_np.transpose(2,1,0)[None, ...][None, ...] #1,1,192,192,36
# img_new = torch.Tensor(img_new).to(device)
# print("img_new",img_new.shape)

# target_filename = '48d0879e1b7b2264fdabfd93c79841812daecd3b56bf3b2cec8374ab2557f830' 
# target_filename = '3d522523ec38961ef45a8a998cac955c433ec6938de08adbd28e3a60c18094c3'
# target_filename = 'cebd40831e1574532187720dfbf13dcaae8fcd218945630a5c0847e8c1c0a8aa'
# target_filename = '9538eff66d146fefc3437137a4ac6467c8c53b7fd44cb87401042e7eb3bdd3de'
# target_filename = '9ed9a6635b671eb5da8d8eff030b02b662c531cd46e0fc6f1b30aa4636a91a7a'
# target_filename = 'e111418b82285066489ef354189414d2603ae31e2dd1b16d60eea1e4be6bf556'
target_filename = '9e85100d6217e4d9d800f3d1c618be1beac6621ec53fc2e0204ed5d863492601'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(test_labels_dir)
row = df[df['Subjects'] == target_filename].iloc[0]
label_EVI = torch.tensor(1 if str(row['EVI']).strip().lower() == 'true' else 0).to(device)
label_MFI = torch.tensor(1 if str(row['MFI']).strip().lower() == 'true' else 0).to(device)

# ==== ???? ====
image_path = f"{test_dir}/{target_filename}.nii.gz"
img_np = sitk.GetArrayFromImage(sitk.ReadImage(image_path))  # shape: [36, 192, 192]
img_new = img_np.transpose(2,1,0)[None, ...][None, ...] #1,1,192,192,36
img_new = torch.Tensor(img_new).to(device)
print("img_new",img_new.shape)
print("EVI label:", label_EVI.item(), "MFI label:", label_MFI.item())


with torch.enable_grad():
    output1 = model_cam1(img_new.to(device))
    print("output1 type:", type(output1))

    #output2 = model_cam2(img_new.to(device))
    pred_EVI = int((output1 > 0).item())
    #pred_MFI = int((output2 > 0).item())

print("output1.shape:", output1.shape)
#print("output2.shape:", output2.shape)
print(f"predicted label 1:{pred_EVI}")
#print(f"predicted label 2:{pred_MFI}")

targets1 = [ClassifierOutputTarget(pred_EVI)]
#targets2 = [ClassifierOutputTarget(pred_MFI)]
print(f"targets for gradcam: {targets1}")
#print(f"targets for gradcam: {targets2}")

# print("encoder model features printing...")
# print(fmEncoder["encoder"].model.wrapped_model.features)

target_layer1 = model_cam1.encoder.model.wrapped_model.features[7][0].norm1 #.norm2
#target_layer2 = model_cam2.encoder.model.wrapped_model.features[7][0].norm1

def reshape_transform(tensor):
    print(f"Original tensor shape: {tensor.shape}")  # 36，6，6，1024
    # Permute to [C, D, H, W]
    # tensor = tensor.permute(3, 0, 1, 2)  # (C, D, H, W) 
    tensor = tensor.permute(3, 1, 2, 0)  # (C, D, H, W) 
    print(f"Permuted tensor shape: {tensor.shape}") 
    # Add batch dimension
    return tensor.unsqueeze(0)  # (1, C, D, H, W)
    # return tensor

cam1 = GradCAM(
    model=model_cam1,
    target_layers=[target_layer1],
    reshape_transform=reshape_transform
)

#cam2 = GradCAM(
#    model=model_cam2,
#    target_layers=[target_layer2],
#    reshape_transform=reshape_transform
#)

# Run cam
grayscale_cam1 = cam1(img_new, [ClassifierOutputTarget(0)]) #INDEX=0 #1,1,192,192,36
#grayscale_cam2 = cam2(img_new, [ClassifierOutputTarget(0)])
print(f"Grayscale CAM shape: {grayscale_cam1.shape}")
 #1，192，192,36

import matplotlib.pyplot as plt

# Middle slice
mode = "EVI"  # ?? "MFI"
slice_range = range(13, 26)

target_prefix = target_filename[:4]
save_dir = os.path.join(
    r"/home/chaimeleon/persistent-home/Complete_set/gradcam/cam_output",
    f"axial_ori{target_prefix}",
    mode
)
os.makedirs(save_dir, exist_ok=True)

for slice_index in slice_range:

    input_slice = img_new[0, 0, :, :, slice_index].cpu().numpy() #1,1,192,192,36
    input_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)
    input_rgb = np.stack([input_norm] * 3, axis=-1).astype(np.float32) #192，192，3

    # Prepare CAM1
    grayscale_cam_slice1 = grayscale_cam1[0, :, :, slice_index] #192,192
    grayscale_cam_slice1 = (grayscale_cam_slice1 - np.min(grayscale_cam_slice1))/(np.max(grayscale_cam_slice1) - np.min(grayscale_cam_slice1))
    grayscale_cam_uint8_1 = np.uint8(255 * grayscale_cam_slice1) #192,192
    # grayscale_cam_uint8 = np.stack([grayscale_cam_uint8] * 3, axis=-1) #192,192,3

    # Overlay
    overlay1 = show_cam_on_image(input_rgb, grayscale_cam_uint8_1, use_rgb=True)
    #overlay2 = show_cam_on_image(input_rgb, grayscale_cam_uint8_2, use_rgb=True)

    # Plot
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.rot90(input_slice), cmap='gray')
    plt.title(f"Original Slice {slice_index}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(overlay1))
    plt.title("Grad-CAM EVI")
    plt.axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f"70norm1_slice{slice_index}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


print("save done")

# plt.show()

