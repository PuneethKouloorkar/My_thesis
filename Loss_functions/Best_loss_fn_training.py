import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
import os
from loss_fns_utils import gen_hyperparams, training_hyperparams, paths, root_path
from loss_fns_utils import training, validation, plots, loss_fns, Superposition
import pickle
from torchvision import transforms

# Import all the architectures--------------------------------------------------
from DnCNN_plus import DnCNN_plus


#-----------------------Note------------------------------------------------------
# pip install pytorch-msssim
# pip install git+https://github.com/jonbarron/robust_loss_pytorch
#-----------------------Note------------------------------------------------------


# Load the GPU device-------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Load all the hyperparameters-----------------------------------------------------
batch, epochs, start_lr, loss_fn, l2_penalty, num_processes, lr_decay_mul, lr_decay_epoch_count = gen_hyperparams()
num, crop_size, min_phantoms, max_phantoms, min_alpha, max_alpha, phantom = training_hyperparams()


# Load all the required paths------------------------------------------------------
data_fol = paths()


# Load the phantom data------------------------------------------------------------
phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(os.path.join(data_fol, f'm{mask}{shift}.pkl'), "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data

img_t = transforms.Compose([transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor))])


# Training and validation datasets--------------------------------------------------
intf_list = pickle.load(open(os.path.join(data_fol, 'intf_imgs.pkl'), "rb"))
for cf in [1, 2, 3]:
    os.makedirs(os.path.join(root_path, f'cross_fold_{cf}'))

    if cf == 1:
        train_x = intf_list[:int(0.90*len(intf_list))]      # 6372  
        val_x = intf_list[int(0.90*len(intf_list)):]        # 708
        #print(len(train_x), len(val_x))
    elif cf == 2:
        train_x = intf_list[int(0.10*len(intf_list)):]      # 6372  
        val_x = intf_list[:int(0.10*len(intf_list))]        # 708
        #print(len(train_x), len(val_x))
    elif cf == 3:
        train_x = intf_list[:int((len(intf_list) - int(0.10*len(intf_list)))/2)]      # 3186
        train_x += intf_list[len(train_x) + int(0.10*len(intf_list)):]                # 3186
        val_x = intf_list[int((len(intf_list) - int(0.10*len(intf_list)))/2):int((len(intf_list) - int(0.10*len(intf_list)))/2)+int(0.10*len(intf_list))]        # 708
        #print(len(train_x), len(val_x))
    else:
        print('Cross-folds greater than 3!')


    training_set = Superposition(train_x, img_t, phantom, phantom_dict, min_phantoms, max_phantoms, min_alpha, max_alpha)
    trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Trainloader size: {len(trainloader)}')

    val_set = Superposition(val_x, img_t, phantom, phantom_dict, min_phantoms, max_phantoms, min_alpha, max_alpha)
    valloader = DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Valloader size: {len(valloader)}')


    # Training-------------------------------------------------------------------------
    for curr_loss in ['MAE', 'MSE', 'MSE_then_MAE', 'SSIM', 'MSSSIM', 'SmoothL1', 'cb_adaptive']:
        os.makedirs(os.path.join(root_path, f'cross_fold_{cf}', curr_loss))
        out_fol = os.path.join(root_path, f'cross_fold_{cf}', curr_loss)

        # Construct the model object and load it into the device--------------------------
        model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)
        
        # Construct the optimizer object---------------------------------------------------
        if curr_loss == 'cb_adaptive':
            loss_params = list(loss_fns[curr_loss].parameters())
            optim = torch.optim.Adam(list(model.parameters()) + loss_params, lr=start_lr, weight_decay=l2_penalty)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)


        t, v = [], []
        t_r, v_r = [], []


        # Training-------------------------------------------------------------------------
        if 'then' not in curr_loss:
            loss_fn = loss_fns[curr_loss]
            for epoch in range(1, epochs+1):
                training_loss, t_ratio = training(trainloader, model, device, loss_fn, num, crop_size, optim)
                val_loss, val_r  = validation(model, valloader, loss_fn, device, num, crop_size) 
                optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul
        else:
            loss_fn = (loss_fns[curr_loss.split("_")[0]], loss_fns[curr_loss.split("_")[-1]])
            for epoch in range(1, epochs+1):
                if epoch <= epochs/2:
                    training_loss, t_ratio = training(trainloader, model, device, loss_fn, num, crop_size, optim)
                    val_loss, val_r  = validation(model, valloader, loss_fn, device, num, crop_size) 
                else:
                    training_loss, t_ratio = training(trainloader, model, device, loss_fn, num, crop_size, optim)
                    val_loss, val_r  = validation(model, valloader, loss_fn, device, num, crop_size) 
                
                optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul

        t_r.append(t_ratio)    
        v_r.append(val_r) 
        t.append(training_loss)
        v.append(val_loss)
        print(epoch, optim.param_groups[0]["lr"], training_loss, val_loss, t_ratio, val_r)


        # Save checkpoint and the rest------------------------------------------------------
        torch.save(model.state_dict(), os.path.join(out_fol, f'{curr_loss}_{cf}.pt'))    
        torch.save(torch.Tensor(t), os.path.join(out_fol, f'{curr_loss}_{cf}_training_loss.pt'))
        torch.save(torch.Tensor(v), os.path.join(out_fol, f'{curr_loss}_{cf}_val_loss.pt'))
        torch.save(torch.Tensor(t_r), os.path.join(out_fol, f'{curr_loss}_{cf}_training_ratio.pt'))
        torch.save(torch.Tensor(v_r), os.path.join(out_fol, f'{curr_loss}_{cf}_val_ratio.pt'))


        # Plot training and validation loss, training and validation ratio-------------------
        plots(epochs, t, v, t_r, v_r, out_fol, curr_loss)