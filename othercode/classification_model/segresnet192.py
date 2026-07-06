Patch_size = (192,192,36)

writer_loc ='/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/logs'
save_model_loc = '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/models'

train_dir = '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train2'
val_dir = '/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/val2'

train_label ='/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/train_labels.csv'
val_label='/home/chaimeleon/persistent-home/classificationmodel_original_sagittal/total_work/normalized2/val_labels.csv'

csv_location = '/home/chaimeleon/persistent-home/Complete_set/Experiments/Results/Segresnet'

Batch_size = 8 # 16*8
num_epochs = 200
num_worker =8

Prefix = 'segresnet_sagittal_original5'

#train
___RUN_Inference___ = False  #For train model is false. for test model is true
___From_Scratch___ = True  # not pretrained model  #for train model is true. for test is false
___Export_results___ = False

#test
# ___RUN_Inference___ = True  #For train model is false. for test model is true
# ___From_Scratch___ = False  # not pretrained model  #for train model is true. for test is false
# ___Export_results___ = True

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
)
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
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
torch._logging.set_logs(recompiles=True)


def FromScratch():
    writer_path = (f"{writer_loc}/{Prefix}")
    if os.path.exists(writer_path):
        print('Deleting Previous Logs')
        shutil.rmtree(writer_path)
    models_path = (f'{save_model_loc}/{Prefix}checkpoint_best.pth')
    if os.path.exists(models_path):
        os.remove(models_path)
    if os.path.exists(models_path.replace("best","latest")):    
        os.remove(models_path.replace("best","latest"))

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

        labels = {'EVI':evi_label, 'MFI': mfi_label}
        
        return {'image': image,'label':labels}
    

class Classification1(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.avgpool = torch.nn.AdaptiveMaxPool3d((4, 4, 4))
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1)) #AdaptiveMaxPool3d

        self.classifier1 = torch.nn.Sequential(
            # torch.nn.Linear(20480, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Linear(4096, 1024),
            # torch.nn.ReLU(),
            torch.nn.Linear(512, 1)  #  use  320, or 512
        )

        # self.classifier2 = torch.nn.Sequential(
        #     # torch.nn.Linear(20480, 4096),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear(4096, 1024),
        #     # torch.nn.ReLU(),
        #     torch.nn.Linear(512, 1)  # use 320, or 512
        # )

    def forward(self, x):
        features = self.backbone(x)
        pooled_features = self.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)
		
        z1 = self.classifier1(flattened_features)
        # z2 = self.classifier2(flattened_features)
        # return z1, z2
        return z1

class Classification2(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.avgpool = torch.nn.AdaptiveMaxPool3d((4, 4, 4))
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1)) #AdaptiveMaxPool3d

        self.classifier1 = torch.nn.Sequential(
            # torch.nn.Linear(20480, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Linear(4096, 1024),
            # torch.nn.ReLU(),
            torch.nn.Linear(512, 1)  #  use  320, or 512
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled_features = self.avgpool(features)
        flattened_features = torch.flatten(pooled_features, 1)
		
        z1 = self.classifier1(flattened_features)
        return z1

def SetupModel(device):
    torch.backends.cudnn.benchmark = False
   # Use a resnet backbone.
    features_per_stage = (32,64,128,256,512)  #use 320, or 512
    num_stages = 5
    conv_op = torch.nn.Conv3d

    kernel_sizes = ( (3, 3, 1), (3, 3, 1), (3, 3, 3), (3, 3, 3), (3, 3, 3));
    strides = ( (2, 2, 1), (2, 2, 1), (2, 2, 2), (2, 2, 2), (2, 2, 1)); #or use (1, 1, 1)
    n_blocks_per_stage =(1,3,4,6,3) #(1,1,1,1,1,1,1) #
    conv_bias = False
    norm_op = torch.nn.BatchNorm3d; 
    norm_op_kwargs= {"eps": 1e-05, "affine": True}
    dropout_op = None
    dropout_op_kwargs = None

    nonlin = torch.nn.LeakyReLU
    nonlin_kwargs = {"inplace": True}
    squeeze_excitation = True

    backbone1 = ResidualEncoder(input_channels=1,n_stages=num_stages,features_per_stage=features_per_stage,conv_op=conv_op,kernel_sizes=kernel_sizes,
                            strides=strides,n_blocks_per_stage=n_blocks_per_stage,conv_bias=conv_bias,norm_op=norm_op,norm_op_kwargs=norm_op_kwargs,
                            dropout_op=dropout_op,dropout_op_kwargs=dropout_op_kwargs,nonlin=nonlin,nonlin_kwargs=nonlin_kwargs,
                            squeeze_excitation=squeeze_excitation)
    
    backbone2 = ResidualEncoder(input_channels=1,n_stages=num_stages,features_per_stage=features_per_stage,conv_op=conv_op,kernel_sizes=kernel_sizes,
                            strides=strides,n_blocks_per_stage=n_blocks_per_stage,conv_bias=conv_bias,norm_op=norm_op,norm_op_kwargs=norm_op_kwargs,
                            dropout_op=dropout_op,dropout_op_kwargs=dropout_op_kwargs,nonlin=nonlin,nonlin_kwargs=nonlin_kwargs,
                            squeeze_excitation=squeeze_excitation)
    
    model1 = Classification1(backbone1).to(device)
    model2 = Classification2(backbone2).to(device)
    
    
    loss_function1 = loss.focal_loss.FocalLoss(alpha=0.69, gamma=2.0, reduction='mean')#,weight=torch.Tensor([0.68789809,1.83050847]).to(device))
    loss_function2 = loss.focal_loss.FocalLoss(alpha=0.78, gamma=2.0, reduction='mean')

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001, weight_decay=1e-6)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, weight_decay=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)
    print('model Setup Complete')
    return model1, model2, loss_function1, loss_function2, optimizer1, optimizer2, scaler, scheduler1, scheduler2

def LoadCheckpoint(checkpointname,model,optimizer,scalar):
    checkpoint = torch.load(checkpointname)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scalar.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_score = checkpoint['best_val_score']

    return epoch,best_val_score
   

def Setup_Data(train_dir, val_dir, Batch_size,num_worker):   

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
        # NormalizeIntensityd(keys=(['image']),channel_wise=True),
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
        # NormalizeIntensityd(keys=(['image']),channel_wise=True),
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


def train_model(device,model1, model2, best_loss1, best_loss2, currepoch,num_epochs, loss_function1, loss_function2, optimizer1,optimizer2, scaler,train_loader,val_loader,writer,Prefix,scheduler1, scheduler2):
    print('Starting Train Loop')

    best_score1 = -1
    best_score2 = -1
    for epoch in range(currepoch,num_epochs):
 
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")

        model1.train()
        model2.train()
        RunningLoss1 = 0
        RunningLoss2 = 0
        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        for indx,batch in loop:
    
            img = batch['image'].to(device,non_blocking=True)
            evi_labels = batch['label']['EVI'].to(device,non_blocking=True)
            mfi_labels = batch['label']['MFI'].to(device,non_blocking=True)

            with torch.autocast(device_type='cuda'):
 
                out1 = model1(img)
                out2 = model2(img)
                # print("out1", out1.size()) #[batch, 1]
                # print("out2", out1.size()) #[batch, 1]

                loss1 = loss_function1(out1, evi_labels.unsqueeze(1))
                loss2 = loss_function2(out2, mfi_labels.unsqueeze(1))

                # loss = (loss1 + loss2) / 2
                    
            scaler.scale(loss1).backward()
            scaler.unscale_(optimizer1)
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 2)
            scaler.step(optimizer1)
            scaler.update()

            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer2)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 2)
            scaler.step(optimizer2)
            scaler.update()

            optimizer1.zero_grad(set_to_none=True)
            RunningLoss1 += loss1.item()
            optimizer2.zero_grad(set_to_none=True)
            RunningLoss2 += loss2.item()        

        writer.add_scalar("Loss/train1", RunningLoss1/len(train_loader), epoch)
        writer.add_scalar("Loss/train2", RunningLoss2/len(train_loader), epoch)

        if (epoch+1) % 10 == 0:
            print('Saving Latest')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_score': best_loss1,
            }, f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_score': best_loss2,
            }, f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth')
        
        val_loss1,val_loss2, metrics_evi, metrics_mfi  = validation(model1,model2,val_loader,device,loss_function1,loss_function2)
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
def validation(model1,model2,val_loader,device,loss_function1,loss_function2, csv_prefix=''):
    model1.eval()
    model2.eval()

    all_output_evi = []
    all_output_mfi = []
    all_label_evi = []
    all_label_mfi = []

    RunningLoss1 = 0
    RunningLoss2 = 0
    all_image_names = []
    
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(device, non_blocking=True)
            evi_labels = batch['label']['EVI'].to(device,non_blocking=True)
            mfi_labels = batch['label']['MFI'].to(device,non_blocking=True)

            if ___Export_results___:
                # Extract image names from MONAI's meta tensor
                image_names = [mt.meta['filename_or_obj'] for mt in img[:]]
                all_image_names.extend(image_names)

            with torch.autocast(device_type='cuda'):
                o1 = model1(img)
                o2 = model2(img)
                loss1 = loss_function1(o1, evi_labels.unsqueeze(1))
                loss2 = loss_function2(o2, mfi_labels.unsqueeze(1))
                # loss = (loss1 + loss2) / 2

            o1 = torch.sigmoid(o1)
            o2 = torch.sigmoid(o2)

            all_output_evi.append(o1.cpu())
            all_output_mfi.append(o2.cpu())
            all_label_evi.append(evi_labels.cpu())
            all_label_mfi.append(mfi_labels.cpu())

            RunningLoss1 += loss1.item()
            RunningLoss2 += loss2.item()

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

    return RunningLoss1 / len(val_loader), RunningLoss2 / len(val_loader), metrics_evi, metrics_mfi




def RunInference():
        global val_dir,val_label
        print('Running Inference')
        val_dir = val_dir.replace('/val2','/test2') #for testng
        val_label = val_label.replace('val_labels.csv','test_labels.csv') # for testing


        train_loader,val_loader = Setup_Data(train_dir,val_dir, Batch_size,num_worker)
        model1, model2, loss_function1, loss_function2, optimizer1, optimizer2, scaler, scheduler1, scheduler2=SetupModel(device)

        if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth'):
            try:
                currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth',model1,optimizer1,scaler)
                currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_best.pth',model2,optimizer2,scaler)
            except Exception as e: 
                print(e)
                print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_best.pth')
        print(f'Best LOSS model-evi : Value{best_loss1}, Epoch{currepoch}')
        print(f'Best LOSS model-mfi : Value{best_loss2}, Epoch{currepoch}')

        val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,val_loader,device,loss_function1,loss_function2,csv_prefix='best')
        print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')


        model1, model2, loss_function1, loss_function2, optimizer1, optimizer2, scaler, scheduler1, scheduler2=SetupModel(device)
        if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth'):
            try:
                currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth',model1,optimizer1,scaler)
                currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_bestScr.pth',model2,optimizer2,scaler)   
            except Exception as e: 
                print(e)
                print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_bestScr.pth')
        print(f'Best SCORE model-evi : Value{best_loss1}, Epoch{currepoch}')
        print(f'Best SCORE model-mfi: Value{best_loss2}, Epoch{currepoch}')

        val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,val_loader,device,loss_function1,loss_function2,csv_prefix='best_scr')
        print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')


        model1, model2, loss_function1, loss_function2, optimizer1, optimizer2, scaler, scheduler1, scheduler2=SetupModel(device)
        if os.path.exists(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth'):
            try:
                currepoch,best_loss1 = LoadCheckpoint(f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth',model1,optimizer1,scaler)
                currepoch,best_loss2 = LoadCheckpoint(f'{save_model_loc}/{Prefix}mfi-checkpoint_latest.pth',model2,optimizer2,scaler)
            except Exception as e: 
                print(e)
                print('Error Loading Checkpoint',f'{save_model_loc}/{Prefix}evi-checkpoint_latest.pth')
        print(f'Final : Value{best_loss2}, Epoch{currepoch}')
        val_loss1,val_loss2, metrics_evi, metrics_mfi = validation(model1,model2,val_loader,device,loss_function1,loss_function2,csv_prefix='latest')
        print(f'Val_evi: {val_loss1:.4f},Val_mfi: {val_loss2:.4f}, Best: {best_loss1:.4f}')
        print(f'fscore_evi: {metrics_evi["fScore"]:.4f}, auc_evi: {metrics_evi["auc"]:.4f}, balanced_acc: {metrics_evi["balanced_acc"]:.4f}, sensitivity_evi: {metrics_evi["sensitivity"]:.4f}, specificity_evi: {metrics_evi["specificity"]:.4f}, f1: {metrics_evi["f1"]:.4f}')
        print(f'fscore_mfi: {metrics_mfi["fScore"]:.4f}, auc_mfi: {metrics_mfi["auc"]:.4f}, balanced_acc: {metrics_mfi["balanced_acc"]:.4f}, sensitivity_mfi: {metrics_mfi["sensitivity"]:.4f}, specificity_mfi: {metrics_mfi["specificity"]:.4f}, f1: {metrics_mfi["f1"]:.4f}')

        print('Inference Complete')



if __name__ == '__main__':

    if not ___RUN_Inference___:

        if ___From_Scratch___:
            FromScratch()

        writer = SummaryWriter(f"{writer_loc}/{Prefix}")

        train_loader,val_loader = Setup_Data(train_dir, val_dir, Batch_size,num_worker)

        model1, model2, loss_function1, loss_function2, optimizer1, optimizer2, scaler, scheduler1, scheduler2=SetupModel(device)
                
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
        train_model(device,model1,model2,best_loss1,best_loss2,currepoch,num_epochs, loss_function1, loss_function2, optimizer1, optimizer2, scaler,train_loader,val_loader,writer,Prefix ,scheduler1,scheduler2)

        # RunInference()
    else:
        RunInference()