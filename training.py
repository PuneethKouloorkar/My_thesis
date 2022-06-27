import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
import os
# from torchvision.utils import save_image
# import robust_loss_pytorch.general
from path_file import intf_training
from training_utils import Constant_crop_dataset, Variable_crop_dataset, gen_hyperparams
from training_utils import plots, training_type
from training_utils import training_ccs, training_vcs, validation_ccs, validation_vcs
from training_utils import training_vcs_hyperparams, training_ccs_hyperparams
from DnCNN_plus import DnCNN_plus
import pickle
from torchvision import transforms

# Set the cuda device and training type--------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
t_type, d_type, data_type, multi_gpu_training = training_type()

# Global Variables-----------------------------------------------------------------------
fol, current_archi, out_fol = intf_training(d_type)
# fol = '90x90_with_bg_and_mask'
# current_archi = 'DnCNN_plus'
# data_type = 'only_gau_ph_gau'
# out_fol = '90x90_with_bg_and_mask/dncnn_plus/only_gau_ph_gau/patch'

# Hyperparameters------------------------------------------------------------------------
start_lr, l2_penalty, epochs, loss_fn, _, _, _, _ = gen_hyperparams()
_, _, _, _, lr_decay_mul, lr_decay_epoch_count, num_processes, checkpoint_divisor = gen_hyperparams()
if  t_type == 'cons_crop_size_training':
    batch = training_ccs_hyperparams()
else:
    batch, num, min_crop_size, max_crop_size, min_phantoms, max_phantoms, min_alpha, max_alpha = training_vcs_hyperparams()

# Initialize the model-------------------------------------------------------------------
model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)
if multi_gpu_training == True:
    model = nn.DataParallel(model)                                  

if  t_type == 'cons_crop_size_training':
    # Unpack the feather files---------------------------------------------------------------
    train_x_df = pd.read_feather(os.path.join(fol, f'train_{data_type}.ftr'))
    val_x_df = pd.read_feather(os.path.join(fol, f'val_{data_type}.ftr'))        

    train_x_sp = train_x_df["train_sp"].tolist()
    train_x_org = train_x_df["train_org"].tolist()
    val_x_sp = val_x_df["val_sp"].tolist()
    val_x_org = val_x_df["val_org"].tolist()

    print(len(train_x_sp), len(train_x_org))
    print(len(val_x_sp), len(val_x_org))

    # Build the loaders-----------------------------------------------------------------------
    training_set = Constant_crop_dataset(sp_image_list=train_x_sp, org_image_list=train_x_org)
    trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Trainloader size: {len(trainloader)}')

    val_set = Constant_crop_dataset(sp_image_list=val_x_sp, org_image_list=val_x_org)
    valloader = DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'valloader size: {len(valloader)}')
else:
    intf_list = pickle.load(open(f'{fol}/intf_imgs.pkl', "rb"))
    train_x = intf_list[:int(0.90*len(intf_list))]     
    val_x = intf_list[int(0.90*len(intf_list)):]    

    phantom_dict = {}
    for mask in range(1,5):
        for shift in range(5):
            phantom_data = pickle.load(open(f'{fol}/m{mask}{shift}.pkl', "rb"))
            phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data

    img_t = transforms.Compose([transforms.PILToTensor(),
                                transforms.Lambda(lambda x: x.type(torch.FloatTensor)),])
                                #transforms.Lambda(lambda x: x.to(device))])

    if data_type == 'only_gau':
        phantom = False
    else:
        phantom = True
    
    training_set = Variable_crop_dataset(train_x, img_t, phantom,
                                         phantom_dict, min_phantoms,
                                         max_phantoms, min_alpha, max_alpha)
    trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Trainloader size: {len(trainloader)}')

    val_set = Variable_crop_dataset(val_x, img_t, phantom,
                                    phantom_dict, min_phantoms,
                                    max_phantoms, min_alpha, max_alpha)
    valloader = DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Valloader size: {len(valloader)}')

# Construct the optimizer object-----------------------------------------------------------
optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)

# Lists to store training and validation losses, training and validation ratios------------
t, v = [], []
t_r, v_r = [], []

# Training---------------------------------------------------------------------------------
for epoch in range(1, epochs+1):
    if  t_type == 'cons_crop_size_training':
        training_loss, t_ratio = training_ccs(trainloader, model, device, loss_fn, optim)
        val_loss, val_r  = validation_ccs(model, valloader, loss_fn, device) 
    else:
        training_loss, t_ratio = training_vcs(trainloader, model, device, loss_fn, num, min_crop_size, max_crop_size, optim)
        val_loss, val_r  = validation_vcs(model, valloader, loss_fn, device, num, min_crop_size, max_crop_size) 
    
    v_r.append(val_r) 
    v.append(val_loss)
    t_r.append(t_ratio)   
    t.append(training_loss)
    
    print(epoch, optim.param_groups[0]["lr"], training_loss, val_loss, t_ratio, val_r)

    if epoch % lr_decay_epoch_count == 0:
        optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul
                
    if epoch % checkpoint_divisor == 0:
        torch.save(model.state_dict(), os.path.join(out_fol, current_archi + '_' + data_type + '_' + str(epoch) + '.pt'))    
        torch.save(torch.Tensor(t), os.path.join(out_fol, current_archi + '_training_loss_' + data_type + '_' + str(epoch) + '.pt'))
        torch.save(torch.Tensor(v), os.path.join(out_fol, current_archi + '_val_loss_' + data_type + '_' + str(epoch) + '.pt'))
        torch.save(torch.Tensor(t_r), os.path.join(out_fol, current_archi + '_training_ratio_' + data_type + '_' + str(epoch) + '.pt'))
        torch.save(torch.Tensor(v_r), os.path.join(out_fol, current_archi + '_val_ratio_' + data_type + '_' + str(epoch) + '.pt'))

# Plot training and validation loss, training and validation ratio-------------------------
plots(epochs, t, v, t_r, v_r, out_fol, current_archi, data_type)