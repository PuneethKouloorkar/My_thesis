import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import os
import pickle

from bs_vs_cs_utils import hyperparameters, Variable_crop_dataset, paths, root_path
from bs_vs_cs_utils import training_vcs, validation_vcs, plots, training_vcs_hyperparams


# Import all the architectures--------------------------------------------------
from DnCNN_plus import DnCNN_plus


# Load the GPU device-------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Load all the hyperparameters-----------------------------------------------------
epochs, start_lr, loss_fn, num_processes, lr_decay_mul, lr_decay_epoch_count, l2_penalty = hyperparameters()
min_phantoms, max_phantoms, min_alpha, max_alpha = training_vcs_hyperparams()


# Load all the required paths------------------------------------------------------
data_fol, data_type = paths()


# Load the dataset and dataloaders-------------------------------------------------
intf_list = pickle.load(open(os.path.join(data_fol, 'intf_imgs.pkl'), "rb"))
train_x = intf_list[:int(0.90*len(intf_list))]     
val_x = intf_list[int(0.90*len(intf_list)):]    

phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(os.path.join(data_fol, f'm{mask}{shift}.pkl'), "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data

img_t = transforms.Compose([transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor)),])

if data_type == 'only_gau':
    phantom = False
else:
    phantom = True


for bs_cs in [(32, 256), (48, 192), (64, 128), (128, 80)]:   
    os.makedirs(os.path.join(root_path, bs_cs))
    out_fol = os.path.join(root_path, bs_cs)
    batch, crop_size = bs_cs
    

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


    # Construct the model object and load it into the device--------------------------
    model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)


    # Construct the optimizer object---------------------------------------------------
    optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)


    t, v = [], []
    t_r, v_r = [], []


    # Training---------------------------------------------------------------------------------
    for epoch in range(1, epochs+1):
        training_loss, t_ratio = training_vcs(trainloader, model, device, loss_fn, crop_size, optim)
        val_loss, val_r  = validation_vcs(model, valloader, loss_fn, device, crop_size) 
        optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul

        t_r.append(t_ratio)    
        v_r.append(val_r) 
        t.append(training_loss)
        v.append(val_loss)
        print(epoch, optim.param_groups[0]["lr"], training_loss, val_loss) 


        # Save checkpoint and the rest------------------------------------------------------
        torch.save(model.state_dict(), os.path.join(out_fol, f'{bs_cs}.pt'))    
        torch.save(torch.Tensor(t), os.path.join(out_fol, f'{bs_cs}_training_loss.pt'))
        torch.save(torch.Tensor(v), os.path.join(out_fol, f'{bs_cs}_val_loss.pt'))
        torch.save(torch.Tensor(t_r), os.path.join(out_fol, f'{bs_cs}_training_ratio.pt'))
        torch.save(torch.Tensor(v_r), os.path.join(out_fol, f'{bs_cs}_val_ratio.pt'))


        # Plot training and validation loss, training and validation ratio-------------------
        plots(epochs, t, v, t_r, v_r, out_fol, bs_cs, data_type)