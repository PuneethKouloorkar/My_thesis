import torch
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
import os
from wd_utils import hyperparameters, Constant_crop_size_training, paths, root_path
from wd_utils import training_ccs, validation_ccs, plots


# Import all the architectures--------------------------------------------------
from DnCNN_plus import DnCNN_plus


#-----------------------Note------------------------------------------------------
# Generate train and validation .ftr files using 'data_gen_fin.py' in the Data folder beforehand.
#-----------------------Note------------------------------------------------------


# Load the GPU device-------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Load all the hyperparameters-----------------------------------------------------
batch, epochs, start_lr, loss_fn, num_processes, lr_decay_mul, lr_decay_epoch_count = hyperparameters()


# Load all the required paths------------------------------------------------------
data_fol, data_type = paths()


# Load the dataset and dataloaders-------------------------------------------------
train_x_df = pd.read_feather(os.path.join(data_fol, f'train_{data_type}.ftr'))
val_x_df = pd.read_feather(os.path.join(data_fol, f'val_{data_type}.ftr'))

train_x_sp = train_x_df["train_sp"].tolist()
train_x_org = train_x_df["train_org"].tolist()
val_x_sp = val_x_df["val_sp"].tolist()
val_x_org = val_x_df["val_org"].tolist()
print(len(train_x_sp), len(train_x_org))
print(len(val_x_sp), len(val_x_org))

training_set = Constant_crop_size_training(sp_image_list=train_x_sp, org_image_list=train_x_org)
trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
print(f'Trainloader size: {len(trainloader)}')

val_set = Constant_crop_size_training(sp_image_list=val_x_sp, org_image_list=val_x_org)
valloader = DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=num_processes)
print(f'valloader size: {len(valloader)}')


# Construct the model object and load it into the GPU------------------------------
model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)


# Training-------------------------------------------------------------------------
for w_d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    os.makedirs(os.path.join(root_path, str(w_d)))
    out_fol = os.path.join(root_path, str(w_d))


    # Construct the optimizer object---------------------------------------------------
    optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=w_d)


    t, v = [], []
    t_r, v_r = [], []


    # Training-------------------------------------------------------------------------
    for epoch in range(1, epochs+1):
        training_loss, t_ratio = training_ccs(trainloader, model, device, loss_fn, optim)
        val_loss, val_r  = validation_ccs(model, valloader, loss_fn, device) 
        
        t_r.append(t_ratio)    
        v_r.append(val_r) 
        t.append(training_loss)
        v.append(val_loss)
        print(epoch, optim.param_groups[0]["lr"], training_loss, val_loss)

        optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul


    # Save checkpoint and the rest------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_fol, f'{w_d}.pt'))    
    torch.save(torch.Tensor(t), os.path.join(out_fol, f'{w_d}_training_loss.pt'))
    torch.save(torch.Tensor(v), os.path.join(out_fol, f'{w_d}_val_loss.pt'))
    torch.save(torch.Tensor(t_r), os.path.join(out_fol, f'{w_d}_training_ratio.pt'))
    torch.save(torch.Tensor(v_r), os.path.join(out_fol, f'{w_d}_val_ratio.pt'))


    # Plot training and validation loss, training and validation ratio-------------------
    plots(epochs, t, v, t_r, v_r, out_fol, w_d, data_type)