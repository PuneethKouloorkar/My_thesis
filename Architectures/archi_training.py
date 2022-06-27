import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import pandas as pd
import os
from archi_utils import gen_hyperparams, Superposition, paths, root_path
from archi_utils import training, validation, plots, training_hyperparams
from torchvision import transforms
import pickle
import random

# Import all the architectures--------------------------------------------------
from DnCNN_plus import DnCNN_plus
from DnCNN_minus import DnCNN_minus
from Double_DnCNN_minus import Double_DnCNN_minus
from DnCNN_paper import DnCNN_paper
from ResNet import Block, ResNet
from UNet import UNet
from DenseNet_UNet import FCDenseNet57

# Load the GPU device-------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Load all the hyperparameters-----------------------------------------------------
batch, epochs, start_lr, loss_fn, l2_penalty, num_processes, lr_decay_mul, _ = gen_hyperparams()
num, crop_size, min_phantoms, max_phantoms, min_alpha, max_alpha, phantom, _ = training_hyperparams()


# Load all the required paths------------------------------------------------------
data_fol, data_type = paths()


for cf in range(1, 4):
    os.makedirs(os.path.join(root_path, f'cross_fold_{cf}'))
    
    # Load the dataset and dataloaders-------------------------------------------------
    intf_list = pickle.load(open(os.path.join(data_fol, 'intf_imgs.pkl'), "rb"))
    random.shuffle(intf_list)
    train_x = intf_list[:int(0.90*len(intf_list))]      # 6372  
    val_x = intf_list[int(0.90*len(intf_list)):]        # 708

    phantom_dict = {}
    for mask in range(1,5):
        for shift in range(5):
            phantom_data = pickle.load(open(os.path.join(data_fol, f'm{mask}{shift}.pkl'), "rb"))
            phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data

    img_t = transforms.Compose([transforms.PILToTensor(),
                                transforms.Lambda(lambda x: x.type(torch.FloatTensor))])

    training_set = Superposition(train_x, img_t, phantom, phantom_dict, min_phantoms, max_phantoms, min_alpha, max_alpha)
    trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Trainloader size: {len(trainloader)}')

    val_set = Superposition(val_x, img_t, phantom, phantom_dict, min_phantoms, max_phantoms, min_alpha, max_alpha)
    valloader = DataLoader(val_set, batch_size=batch, shuffle=True, num_workers=num_processes)
    print(f'Valloader size: {len(valloader)}')


    # Training-------------------------------------------------------------------------
    for archi in ['ResNet18', 'ResNet34', 'UNet', 'FCDenseNet', 'DnCNN_plus', 'DnCNN_minus', 'Double_DnCNN_minus', 'DnCNN_paper', 'DnCNN_paper_pT']:
        os.makedirs(os.path.join(root_path, f'cross_fold_{cf}', archi))
        out_fol = os.path.join(root_path, f'cross_fold_{cf}', archi)

        # DnCNN paper reconstructions noise instead of the clean image-----------------
        if archi.startswith('DnCNN_paper'):
            noise_recons = True
        else:
            noise_recons = False


        # Construct the model----------------------------------------------------------
        if archi == 'DnCNN_plus':
            model = DnCNN_plus(1, 1, [8,32,128,512])
        elif archi == 'DnCNN_minus':
            model = DnCNN_minus(1, 1, [8,32,128,512])
        elif archi == 'Double_DnCNN_minus':
            model = Double_DnCNN_minus(1, 1, [4,16,64,256], [8,32,128,512])
        elif archi == 'DnCNN_paper_pT':
            model = DnCNN_paper(1, 20)
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load('DnCNN_paper.pth')) 
        elif archi == 'DnCNN_paper':
            model = DnCNN_paper(1, 20)
        elif archi == 'ResNet18':
            model = ResNet(Block, [2,2,2,2], 1, 1)
        elif archi == 'ResNet34':
            model = ResNet(Block, [3,6,4,3], 1, 1)
        elif archi == 'UNet':
            model = UNet(1,1)
        elif archi == 'FCDenseNet':
            model = FCDenseNet57(1)

        
        # Load the model in GPU------------------------------------------------------------
        model = nn.DataParallel(model)
        model = model.to(device)


        # Construct the optimizer object---------------------------------------------------
        optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)

        t, v = [], []
        t_r, v_r = [], []


        # Training-------------------------------------------------------------------------
        for epoch in range(1, epochs+1):
            training_loss, t_ratio = training(trainloader, model, device, loss_fn, num, crop_size, optim)
            val_loss, val_r  = validation(model, valloader, loss_fn, device, num, crop_size) 
            
            t_r.append(t_ratio)    
            v_r.append(val_r) 
            t.append(training_loss)
            v.append(val_loss)
            print(epoch, optim.param_groups[0]["lr"], training_loss, val_loss, t_ratio, val_r)

            optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]*lr_decay_mul


        # Save checkpoint and the rest------------------------------------------------------
        torch.save(model.state_dict(), os.path.join(out_fol, f'{archi}.pt'))    
        torch.save(torch.Tensor(t), os.path.join(out_fol, f'{archi}_training_loss.pt'))
        torch.save(torch.Tensor(v), os.path.join(out_fol, f'{archi}_val_loss.pt'))
        torch.save(torch.Tensor(t_r), os.path.join(out_fol, f'{archi}_training_ratio.pt'))
        torch.save(torch.Tensor(v_r), os.path.join(out_fol, f'{archi}_val_ratio.pt'))


        # Plot training and validation loss, training and validation ratio-------------------
        plots(epochs, t, v, t_r, v_r, out_fol, archi, data_type)