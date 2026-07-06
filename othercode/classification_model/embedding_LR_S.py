import subprocess, sys

required  = {'wheel','setuptools','torch==2.1.2','torchvision==0.16.2','torchaudio==2.1.2','mmdet','mmcv==2.1.0','medicalmultitaskmodeling','monai','tensorboard','SimpleITK','torch==2.1.2','torchvision==0.16.2','torchaudio==2.1.2'}

#
subprocess.check_call([sys.executable, '-m', 'pip', 'install','--no-index','--find-links','/home/chaimeleon/persistent-home/UMedPt/FM/FM', *required])

# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM wheel setuptools
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM mmdet 
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM mmcv==2.1.0 
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM medicalmultitaskmodeling 
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM monai
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM tensorboard
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM SimpleITK
# !pip install --no-index --find-links /home/chaimeleon/persistent-home/Packages/FM torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

Patch_size = (192,192,24)

writer_loc ='/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/logs_FM'
save_model_loc = '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/models_FM'
csv_location = '/home/chaimeleon/persistent-home/Complete_set/Experiments/Results/'

# model saving directory
train_dir = r"/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train2"
val_dir = r"/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train_val2"
    
train_label = r'/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train_labels.csv'
val_label = r'/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train_val_labels.csv'


base_dir = '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/embedding_LR/'

Batch_size = 1
num_epochs = 200
num_worker =8

Prefix = 'SAG_EM_LR192' #Increased Pos weight
# prior_name ='gland'

#train
# ___RUN_Inference___ = False
# ___From_Scratch___ = True
# ___Export_results___ = False

# #test
___RUN_Inference___ = True
___From_Scratch___ = False
___Export_results___ = False

accumulation_steps =24 #32*8=256 img size 26*8=208<224
#EXTRA PRECAUTION

# if ___RUN_Inference___:
#     ___From_Scratch___ =False
#     ___Export_results___=True


import os
import torch
from monai.data import DataLoader, Dataset,ArrayDataset,decollate_batch
import monai.losses as loss
from monai.utils import set_determinism
import pandas as pd
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandFlipd,
    RandZoomd,
    NormalizeIntensityd,
    AsDiscrete,
    RandAffined,
    RandBiasFieldd,
    RandGibbsNoised,
    ToTensord,
    RepeatChanneld,
    ScaleIntensityRangePercentilesd,
    ConcatItemsd
)

import random
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from glob import glob
import shutil
from torch.utils.tensorboard import SummaryWriter
set_determinism(6339391)
#GLOBALS FOR LAZINESS
#post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
#post_label = Compose([AsDiscrete(to_onehot=2)])
#torch._logging.set_logs(recompiles=True)
# Compulsory to accept the license
os.environ['MMM_LICENSE_ACCEPTED'] = 'i accept'


from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.labelstudio_ext.NativeBlocks import MMM_MODELS

number_of_steps= int(212/Batch_size)
print(os.path.basename(__file__),Prefix,Patch_size,Batch_size,number_of_steps)

def FromScratch():
    writer_path = (f"{writer_loc}/{Prefix}")
    if os.path.exists(writer_path):
        print('Deleting Previous Logs')
        shutil.rmtree(writer_path)
    # models_path = (f'{save_model_loc}/{Prefix}checkpoint_best.pth')
    # if os.path.exists(models_path):
    #     os.remove(models_path)
    # if os.path.exists(models_path.replace("best","latest")):    
    #     os.remove(models_path.replace("best","latest"))

def app_int(risk):
    risk = str(risk).strip().lower()
    if risk == 'true':
        return 1
    else:
        return 0

class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        data = pd.read_csv(csv_file)
        # print(csv_file)

        subject_ids = data['Subjects'].tolist()
        labels1 = data['EVI'].apply(app_int).tolist()
        labels2 = data['MFI'].apply(app_int).tolist()

        # print(len(self.subject_id))
        # Filter out cases where .nii.gz does not actually exist
        self.subject_id = []
        self.labels1 = []
        self.labels2 = []

        for sid, l1, l2 in zip(subject_ids, labels1, labels2):
            file_path = os.path.join(self.image_dir, sid + '.nii.gz')
            if os.path.isfile(file_path):
                self.subject_id.append(sid)
                self.labels1.append(l1)
                self.labels2.append(l2)
            else:
                # Optionally print a warning or keep a log
                print(f"Skipping {file_path} (not found).")

        print(f"Loaded {len(self.subject_id)} subjects with existing .nii.gz files.")


    def __len__(self):
        return len(self.subject_id)

    def __getitem__(self, index):

        image = os.path.join(self.image_dir, self.subject_id[index]+'.nii.gz')
        evi_label = self.labels1[index] #EVI
        mfi_label = self.labels2[index] #MFI

        # evi_label = 1 if label1 == "True" else 0
        # mfi_label = 1 if label2 == "True" else 0

        labels = {'EVI':evi_label, 'MFI': mfi_label}
        
        return {'image': image,'label':labels}
    

class WarmupReduceLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, plateau_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        super(WarmupReduceLROnPlateau, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, metrics=None):
        if self.last_epoch < self.warmup_epochs:
            self.last_epoch += 1
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.plateau_scheduler.step(metrics)
    
    def get_last_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return self.plateau_scheduler.get_last_lr()

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

# combining the grouper and classifier for convenience
class CombinedModel(torch.nn.Module):
    def __init__(self, grouper, classifier): # supercase_indices should be used at runtime
        super(CombinedModel, self).__init__()                  # hardcoded for simplicity at the intilization stage
        self.grouper = grouper                                 # this will restrict parallel processing
        self.classifier = classifier                           # at this stage, I don't care about speed

    def forward(self, hidden_vector,supercase_indices, return_embedding=False):
        grouper_vector, weights = self.grouper(hidden_vector, supercase_indices)
        output = self.classifier(grouper_vector)

        if return_embedding:
            return output, weights, grouper_vector
        else:
            return output, weights

    
def SetupModel(device):
    MMM_MODELS = {"encoder-1.0.4.pt.zip": "/home/chaimeleon/persistent-home/UMedPt/FM/encoder-1.0.4.pt.zip"}
    fmEncoder = NativeBlocks(MMM_MODELS[DEFAULT_MODEL])
    embedding_dim=512

    # Keep only the encoder and squeezer
    fmEncoder =torch.nn.ModuleDict({
        "encoder": fmEncoder["encoder"],
        "squeezer": fmEncoder["squeezer"]
    }).to(device)

    fmGrouper1 = Grouper(Grouper.Config(module_name="grouper", version="weighted"),
                        embedding_dim=embedding_dim,)
    fmClassifier1 = SmartHead(hidden_dim=embedding_dim,out_dim=1,dropout=0.5)
   
    Weighted_classifier1 = CombinedModel(fmGrouper1, fmClassifier1).to(device)


    fmGrouper2 = Grouper(Grouper.Config(module_name="grouper", version="weighted"),
                        embedding_dim=embedding_dim,)
    fmClassifier2 = SmartHead(hidden_dim=embedding_dim,out_dim=1,dropout=0.5)
    
    Weighted_classifier2 = CombinedModel(fmGrouper2, fmClassifier2).to(device)

    # Define optimizer and loss function
    optimizer1 = torch.optim.AdamW(Weighted_classifier1.parameters(), lr=5e-4, weight_decay=1e-4)
    # Define optimizer and loss function
    optimizer2 = torch.optim.AdamW(Weighted_classifier2.parameters(), lr=5e-4, weight_decay=1e-4)

    loss_function1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.15).to(device))#EVI negative/positive in train
    loss_function2 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.7).to(device))#MFI negative/positive in train

 
    
    scaler = None# torch.cuda.amp.GradScaler(enabled=False)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)
    print('model Setup Complete')
    return Weighted_classifier1,Weighted_classifier2,fmEncoder, loss_function1,loss_function2, optimizer1,optimizer2,scaler , scheduler1,scheduler2




def LoadCheckpoint(checkpointname,model,optimizer,scalar):
    checkpoint = torch.load(checkpointname,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scalar is not None:
        scalar.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_score = checkpoint['best_val_score']

    return epoch,best_val_score



    

def Setup_Data(train_dir,val_dir,Batch_size,num_worker):   

    train_transforms1 = Compose([
        LoadImaged(keys=(['image']),),
        EnsureChannelFirstd(keys=(['image']),),
        SpatialPadd(keys=(['image']), spatial_size=Patch_size, method='symmetric'),
        RandZoomd(keys=(['image']), prob=0.2, min_zoom=0.9, max_zoom=1.1),
        RandAffined(keys=(['image']), prob=0.5, rotate_range=(0, 0, np.pi/15), shear_range=(0.1, 0.1, 0.1),scale_range =(0.1,0.1,0.1), padding_mode='border'),
        RandFlipd(keys=(['image']), prob=0.2, spatial_axis=0),
        RandGaussianNoised(keys=(['image']), prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=(['image']), prob=0.1, sigma_x=(0.5, 1), sigma_y=(0.5, 1), sigma_z=(0.5, 1)),
        RandScaleIntensityd(keys=(['image']), prob=0.2, factors=[0.8, 1.2]),
        RandAdjustContrastd(keys=(['image']), prob=0.2, gamma=(0.8, 1.2)),
        RandBiasFieldd(keys=(['image']),prob=0.1,coeff_range=(0.1,0.2)),
        RandGibbsNoised(keys=(['image']),prob=0.1,alpha=(0.6,0.8)),
        ScaleIntensityRangePercentilesd(keys=(['image']),lower=2.5,upper=97.5,b_min=0,b_max=1),
        RepeatChanneld(keys=(['image']),repeats=3),
        ConcatItemsd(keys=(['image']),name='image'),
        ToTensord(keys=(['image','label']),)
    ])

    val_transforms = Compose([
        LoadImaged(keys=(['image']),),
        EnsureChannelFirstd(keys=(['image']),),
        SpatialPadd(keys=(['image']), spatial_size=Patch_size, method='symmetric'),
        #RandZoomd(keys=(['image']), prob=0.2, min_zoom=0.9, max_zoom=1.1),
        #RandAffined(keys=(['image']), prob=0.2, rotate_range=(np.pi/15, np.pi/15, np.pi/15), shear_range=(0.1, 0.1, 0.1),scale_range =(0.1,0.1,0.1), padding_mode='zeros'),
        #RandFlipd(keys=(['image']), prob=0.2, spatial_axis=0),
        #Rand3DElasticd(keys=(['image']), prob=0.2, sigma_range=(5,7),magnitude_range = (50,150),padding_mode ='zeros'),
        #RandGaussianNoised(keys=(['image']), prob=0.2, mean=0.0, std=0.1),
        #RandGaussianSmoothd(keys=(['image']), prob=0.2, sigma_x=(0.5, 1), sigma_y=(0.5, 1), sigma_z=(0.5, 1)),
        #RandScaleIntensityd(keys=(['image']), prob=0.2, factors=[0.8, 1.2]),
        #RandAdjustContrastd(keys=(['image']), prob=0.2, gamma=(0.8, 1.2)),
        ScaleIntensityRangePercentilesd(keys=(['image']),lower=2.5,upper=97.5,b_min=0,b_max=1),
        RepeatChanneld(keys=(['image']),repeats=3),
        ConcatItemsd(keys=(['image']),name='image'),
        ToTensord(keys=(['image','label']),)
    ])


    # Create the dataset
    dataset = CustomDataset(train_dir,train_label)
    print('Total=',len(dataset))
    train_dataset = ArrayDataset(dataset,train_transforms1)

    dataset2 = CustomDataset(val_dir,val_label)
    print('Total=',len(dataset2))
    val_dataset = ArrayDataset(dataset2,val_transforms)



    # Create the Training DataLoader
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=num_worker,drop_last = True, pin_memory=True, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=int(num_worker/2),drop_last = False, pin_memory=True, persistent_workers = True)
    
    
    return train_loader,val_loader








#torch.set_float32_matmul_precision('medium')





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







def train_model(device,model1,model2,fmEncoder,best_loss1,best_loss2,currepoch,num_epochs, loss_function1,loss_function2, optimizer1,optimizer2, scaler,train_loader,val_loader,writer,Prefix,scheduler1,scheduler2):
    print('Starting Train Loop')
    
    best_score1 = -1
    best_score2 = -1
    flatten_layer = torch.nn.Flatten(1)
    for epoch in range(currepoch,num_epochs):

   
        loop = tqdm(enumerate(train_loader), total=(number_of_steps))
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")

        model1.train()
        model2.train()
        RunningLoss1 = 0
        RunningLoss2 = 0
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)
        count=0
        for indx,batch in loop:
            if indx>=number_of_steps: break
            img = batch['image'].to(device,non_blocking=True)
            evi_labels = batch['label']['EVI'].to(device,dtype=torch.float,non_blocking=True)
            mfi_labels = batch['label']['MFI'].to(device,dtype=torch.float,non_blocking=True)
            
            batchSize = img.size(0)
            # print('img shape', img.shape)

            # Permute the dimensions of the entire batch
            # Original shape: (batchSize, channel, h, w, d)
            # Permuted shape: (batchSize, d, channel, h, w)
            img_permuted = img.permute(0, 4, 1, 2, 3)
            # print('img_permuted shape', img_permuted.shape)
            # Reshape the permuted tensor to match the desired shape
            # Original shape: (batchSize, d, channel, h, w)
            # Reshaped shape: (batchSize * d, channel, h, w)
            img = img_permuted.reshape(batchSize * Patch_size[2], 3, Patch_size[0], Patch_size[1])
            

            # Create supercase_indices tensor
            supercase_indices = torch.arange(batchSize).repeat_interleave(Patch_size[2]).to(device)
               
            with torch.autocast(device_type='cuda'):
                # Forward pass
                with torch.no_grad():
                    feature_pyramid: list[torch.Tensor] = fmEncoder["encoder"](img)
                    hidden_vector = flatten_layer(fmEncoder["squeezer"](feature_pyramid)[1])


                output1, weights1 = model1(hidden_vector,supercase_indices)
                output2, weights2 = model2(hidden_vector,supercase_indices)
                    
                # Compute loss
                loss1 = loss_function1(output1,evi_labels.unsqueeze(1))
                loss1 = loss1/accumulation_steps if accumulation_steps>1 else loss1

                loss2 = loss_function2(output2,mfi_labels.unsqueeze(1))
                loss2 = loss2/accumulation_steps if accumulation_steps>1 else loss2

            if scaler is None:
                # Accumuating Gradients
                loss1.backward()
                loss2.backward()
                
                RunningLoss1 += loss1.item()
                RunningLoss2 += loss2.item()
                # Update weights once per batch
                if (indx+1)% accumulation_steps ==0:
                    optimizer1.step()
                    optimizer1.zero_grad(set_to_none=True)
                    
                    optimizer2.step()
                    optimizer2.zero_grad(set_to_none=True)
                    count+=1

            else:
                scaler.scale(loss1).backward()
                scaler.scale(loss2).backward()
                # Accumulate gradients
                if (indx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer1)
                    torch.nn.utils.clip_grad_norm_(model1.parameters(), 2)
                    scaler.step(optimizer1)
                    scaler.update()
                    optimizer1.zero_grad(set_to_none=True)

                    scaler.unscale_(optimizer2)
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), 2)
                    scaler.step(optimizer2)
                    scaler.update()
                    optimizer2.zero_grad(set_to_none=True)
                    count+=1

                RunningLoss1 += loss1.item()
                RunningLoss2 += loss2.item()
            
        print(count)
        writer.add_scalar("Loss/train1", RunningLoss1/(indx+1), epoch)
        writer.add_scalar("Loss/train2", RunningLoss2/(indx+1), epoch)

        if (epoch+1) % 10 == 0:
            print('Saving Latest')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': best_loss1,
            }, f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': best_loss2,
            }, f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth')
        
        val_loss1,val_loss2, metrics_evi, metrics_mfi  = validation(model1,model2,fmEncoder,val_loader,device,loss_function1,loss_function2)
        print(f'Lossevi: {RunningLoss1/(indx+1):.4f},Lossmfi: {RunningLoss2/(indx+1):.4f}, Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, LR_evi: {scheduler1.get_last_lr()[0]:.8f}, LR_mfi: {scheduler2.get_last_lr()[0]:.8f}, Best1: {best_loss1:.4f}, Best2: {best_loss2:.4f}')
        print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')

        
        writer.add_scalar("Loss/valevi", val_loss1, epoch)
        writer.add_scalar("Loss/valmfi", val_loss2, epoch)
        writer.add_scalar("Acc/Val_evi",metrics_evi["balanced_acc"],epoch)
        writer.add_scalar("AUC/Val_evi",metrics_evi["auc"],epoch)
        writer.add_scalar("Acc/Val_mfi",metrics_mfi["balanced_acc"],epoch)
        writer.add_scalar("AUC/Val_mfi",metrics_mfi["auc"],epoch)

        scheduler1.step()#(val_loss)
        scheduler2.step()#(val_loss)

        if val_loss1 < best_loss1:
            print('Saving Best loss',val_loss1,'################################################')
            best_loss1 = val_loss1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': metrics_evi["fScore"],
            }, f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth')

        if val_loss2 < best_loss2:
            print('Saving Best loss',val_loss2,'################################################')
            best_loss2 = val_loss2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': metrics_mfi["fScore"],
            }, f'{save_model_loc}/{Prefix}mfi-checkpoint_best.pth')

        if metrics_evi["fScore"] > best_score1:
            print('Saving Best',metrics_evi["fScore"],'################################################')
            best_score1 = metrics_evi["fScore"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': metrics_evi["fScore"],
            }, f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth')

        if metrics_mfi["fScore"] > best_score2:
            print('Saving Best',metrics_mfi["fScore"],'################################################')
            best_score2 = metrics_mfi["fScore"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                'best_val_score': metrics_mfi["fScore"],
            }, f'{save_model_loc}/{Prefix}mfi-checkpoint_bestScr.pth')
    #print total GPU memmory, used GPU memmory and free GPU memmory
        # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    

    #epoch end



import sklearn.metrics as skm
def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5)
    
    auc = skm.roc_auc_score(y_true, y_pred.float())
    balanced_acc = skm.balanced_accuracy_score(y_true, y_pred_binary)
    sensitivity = skm.recall_score(y_true, y_pred_binary)
    specificity = skm.recall_score(y_true, y_pred_binary, pos_label=0)
    f1 = skm.f1_score(y_true, y_pred_binary)
    
    fScore = 0.4 * auc + 0.2 * balanced_acc + 0.2 * sensitivity + 0.2 * specificity 
    return {
        'fScore': fScore,
        'auc': auc,
        'balanced_acc': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1
    }

# @torch.no_grad()
def validation(model1,model2,fmEncoder,val_loader,device,loss_function1,loss_function2, csv_prefix=''):
    model1.eval()
    model2.eval()

    embedding_list = []
    evi_label_list = []
    mfi_label_list = []

    all_output_evi = []
    all_output_mfi = []
    all_label_evi = []
    all_label_mfi = []
    all_embeddings = []

    all_image_names = []
    RunningLoss1 = 0
    RunningLoss2 = 0
    flatten_layer = torch.nn.Flatten(1)
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(device, non_blocking=True)
            evi_labels = batch['label']['EVI'].to(device,dtype=torch.float,non_blocking=True)
            mfi_labels = batch['label']['MFI'].to(device,dtype=torch.float,non_blocking=True)

            if ___Export_results___:
            # Extract image names from MONAI's meta tensor
                image_names = [mt.meta['filename_or_obj'] for mt in img[:]]
                all_image_names.extend(image_names)

            batchSize = img.size(0)

            # Permute the dimensions of the entire batch
            # Original shape: (batchSize, channel, h, w, d)
            # Permuted shape: (batchSize, d, channel, h, w)
            img_permuted = img.permute(0, 4, 1, 2, 3)

            # Reshape the permuted tensor to match the desired shape
            # Original shape: (batchSize, d, channel, h, w)
            # Reshaped shape: (batchSize * d, channel, h, w)
            img = img_permuted.reshape(batchSize * Patch_size[2], 3, Patch_size[0], Patch_size[1])


            # Create supercase_indices tensor
            supercase_indices = torch.arange(batchSize).repeat_interleave(Patch_size[2]).to(device)


            with torch.autocast(device_type='cuda'):
                # Forward pass
                with torch.no_grad():
                    feature_pyramid: list[torch.Tensor] = fmEncoder["encoder"](img)
                    hidden_vector = flatten_layer(fmEncoder["squeezer"](feature_pyramid)[1])
                    # print("hidden_vector's shape:", hidden_vector.shape)

                    # Convert tensor to NumPy array for logistic regression
                    # embedding_np = hidden_vector.cpu().numpy()
                    evi_labels_np = evi_labels.cpu().numpy()
                    mfi_labels_np = mfi_labels.cpu().numpy()

                    # # Store embeddings and both labels
                    # embedding_list.append(embedding_np)
                    evi_label_list.append(evi_labels_np)
                    mfi_label_list.append(mfi_labels_np)

                    o1, weights1, emb1 = model1(hidden_vector,supercase_indices, return_embedding=True)
                    o2, weights1, emb2 = model2(hidden_vector,supercase_indices, return_embedding=True)
                    all_embeddings.append(emb1.cpu().numpy())
                    # all_embeddings.append(emb2.cpu().numpy())
                    
                    # Compute loss
                    loss1 = loss_function1(o1,evi_labels.unsqueeze(1))
                    loss2 = loss_function2(o2,mfi_labels.unsqueeze(1))
            
            o1 = torch.sigmoid(o1)
            o2 = torch.sigmoid(o2)
            all_output_evi.append(o1.cpu())
            all_output_mfi.append(o2.cpu())
            all_label_evi.append(evi_labels.cpu())
            all_label_mfi.append(mfi_labels.cpu())

            RunningLoss1 += loss1.item()
            RunningLoss2 += loss2.item()

    # Stack into NumPy arrays
    embeddings = np.vstack(all_embeddings)  # Shape: (Total_Samples, 512)
    evi_labels = np.hstack(evi_label_list)  # Shape: (Total_Samples,)
    mfi_labels = np.hstack(mfi_label_list)  # Shape: (Total_Samples,)

    embeddings_path = os.path.join(base_dir, f"{csv_prefix}_embeddings.npy")
    evi_labels_path = os.path.join(base_dir, f"{csv_prefix}_evi_labels.npy")
    mfi_labels_path = os.path.join(base_dir, f"{csv_prefix}_mfi_labels.npy")

    # Save to disk for later use
    np.save(embeddings_path, embeddings)
    np.save(evi_labels_path, evi_labels)
    np.save(mfi_labels_path, mfi_labels)

    print(f"Saved embeddings: {embeddings.shape}, EVI labels: {evi_labels.shape}, MFI labels: {mfi_labels.shape}")

    all_outputs_evi = torch.cat(all_output_evi)
    all_outputs_mfi = torch.cat(all_output_mfi)
    all_labels_evi = torch.cat(all_label_evi)
    all_labels_mfi = torch.cat(all_label_mfi)

    metrics_evi = calculate_metrics(all_labels_evi, all_outputs_evi)
    metrics_mfi = calculate_metrics(all_labels_mfi, all_outputs_mfi)


    if ___Export_results___:
    # Save predictions along with image names
        predictions_df = pd.DataFrame({
            'image_name': all_image_names,
            'predicted_probability_evi': all_outputs_evi.squeeze().numpy(),
            'predicted_probability_mfi': all_outputs_mfi.squeeze().numpy(),
            'ground_truth_evi': all_labels_evi.squeeze().numpy(),
            'ground_truth_mfi': all_labels_mfi.squeeze().numpy(),
        })
        global csv_location
        predictions_df.to_csv(os.path.join(csv_location,f'{csv_prefix}_{Prefix}.csv'), index=False)

    return RunningLoss1 / len(val_loader),RunningLoss2 / len(val_loader), metrics_evi, metrics_mfi


def RunInference():
        global val_dir,val_label
        print('Running Inference')
        val_dir = val_dir.replace('/train_val2','/test2')
        val_label = val_label.replace('train_val_labels.csv','test_labels.csv')
        
        train_loader,val_loader = Setup_Data(train_dir,val_dir,Batch_size,num_worker)
        model1,model2,fmEncoder, loss_function1,loss_function2, optimizer1,optimizer2, scaler, scheduler1,scheduler2=SetupModel(device)
        
        # if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth'):
        #     try:
        #         currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth',model1,optimizer1,scaler)
        #         currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_best.pth',model2,optimizer2,scaler)
        #     except Exception as e: 
        #         print(e)
        #         print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth')
        # print(f'Best LOSS model-evi : Value{best_loss1}, Epoch{currepoch}')
        # print(f'Best LOSS model-mfi : Value{best_loss2}, Epoch{currepoch}')

        val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,fmEncoder,val_loader,device,loss_function1,loss_function2,csv_prefix='test')
        # print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        # print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        # print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')

        # model1,model2,fmEncoder, loss_function1,loss_function2, optimizer1,optimizer2, scaler, scheduler1,scheduler2=SetupModel(device)
        # if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth'):
        #     try:
        #         currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth',model1,optimizer1,scaler)
        #         currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_bestScr.pth',model2,optimizer2,scaler)   
        #     except Exception as e: 
        #         print(e)
        #         print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth')
        # print(f'Best SCORE model-evi : Value{best_loss1}, Epoch{currepoch}')
        # print(f'Best SCORE model-mfi: Value{best_loss2}, Epoch{currepoch}')

        # val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,fmEncoder,val_loader,device,loss_function1,loss_function2,csv_prefix='best_scr')
        # print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        # print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        # print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')

        # model1,model2,fmEncoder, loss_function1,loss_function2, optimizer1,optimizer2, scaler, scheduler1,scheduler2=SetupModel(device)
        # if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth'):
        #     try:
        #         currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth',model1,optimizer1,scaler)
        #         currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth',model2,optimizer2,scaler)
        #     except Exception as e: 
        #         print(e)
        #         print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth')
        # print(f'Final : Value{best_loss2}, Epoch{currepoch}')
        # val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,fmEncoder,val_loader,device,loss_function1,loss_function2,csv_prefix='latest')
        # print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        # print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        # print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')

        print('Inference Complete')



if __name__ == '__main__':

    if not ___RUN_Inference___:

        if ___From_Scratch___:
            FromScratch()

        writer = SummaryWriter(f"{writer_loc}/{Prefix}")

        train_loader,val_loader = Setup_Data(train_dir,val_dir,Batch_size,num_worker)

        model1,model2,fmEncoder, loss_function1,loss_function2, optimizer1,optimizer2, scaler, scheduler1,scheduler2=SetupModel(device)
        
        currepoch = 0
        best_loss1 = 1000000000
        if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth'):
            try:
                currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth',model1,optimizer1,scaler)
                print(f'Loading evi-Model\nRestarting from {currepoch}')
            except:
                currepoch = 0
                best_loss1 = 1000000000
                print('Error Loading evi-Checkpoint')

        best_loss2 = 1000000000
        if os.path.exists(f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth'):
            try:
                currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth',model2,optimizer2,scaler)
                print(f'Loading mfi-Model\nRestarting from {currepoch}')
            except:
                currepoch = 0
                best_loss2 = 1000000000
                print('Error Loading mfi-Checkpoint')

        print(f'Training Started')

        train_model(device,model1,model2,fmEncoder,best_loss1,best_loss2,currepoch,num_epochs, loss_function1,loss_function2, optimizer1,optimizer2, scaler,train_loader,val_loader,writer,Prefix,scheduler1,scheduler2)

        # RunInference()
    else:

        RunInference()