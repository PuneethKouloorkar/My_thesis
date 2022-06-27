import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn


root_path = os.path.dirname(os.path.abspath(__file__))


def paths(noise_type, backend):
    os.makedirs(os.path.join(root_path, noise_type))

    out_fol = os.path.join(root_path, noise_type)
    name = f'ddp_{backend}'

    return out_fol, name


def gen_hyperparams():
    batch = 64
    start_lr = 1e-3
    l2_penalty = 1e-4
    num_epochs= 50
    loss_fn = nn.L1Loss()
    lr_decay_count = 5
    lr_decay_mul = 0.4
    checkpoint_count = 5
    return batch, start_lr, l2_penalty, num_epochs, loss_fn, lr_decay_count, lr_decay_mul, checkpoint_count


def load_ftr_datasets(file_name:str=None):
    train_df = pd.read_feather(os.path.join('Data', f'train_{file_name}.ftr'))
    val_df = pd.read_feather(os.path.join('Data', f'train_{file_name}.ftr'))

    train_sp = train_df["train_sp"].tolist()
    train_org = train_df["train_org"].tolist()
    val_sp = val_df["val_sp"].tolist()
    val_org = val_df["val_org"].tolist()
    
    print(len(train_sp), len(train_org))
    print(len(val_sp), len(val_org))
    
    return train_sp, train_org, val_sp, val_org


class For_training(Dataset):
    def __init__(self, sp_image_list, org_image_list):
        self.sp_image_list = sp_image_list
        self.org_image_list = org_image_list
        
    def __len__(self):
        return len(self.sp_image_list)     

    def __getitem__(self, i):
        sp = torch.from_numpy(self.sp_image_list[i])
        org = torch.from_numpy(self.org_image_list[i])
        
        sp, org = sp.reshape(1, 90, 90), org.reshape(1, 90, 90)     # As in crop_size variable in data_gen_fin.py
        return sp, org 


def ratio_computation(inp, label, op, patch):
    inp_noise = torch.sqrt(torch.mean(torch.square(inp[patch]-label[patch])))
    out_noise = torch.sqrt(torch.mean(torch.square(label[patch]-op[patch])))
    
    if float(inp_noise.item()) == 0: 
        inp_noise = torch.Tensor([1e-4])
    
    ratio_ = round(out_noise.item()/inp_noise.item(), 2)
    return ratio_


def validation(loader, trained_model, args, loss_fn):
    val_loss, val_ratio = [], []
    trained_model.eval()     
    
    with torch.no_grad():
        for _, data in enumerate(loader):
            sp, org = data
            sp, org = sp.to(torch.device(args.local_rank)), org.to(torch.device(args.local_rank))
            
            patch = (org != 0)                  
            outputs = trained_model(sp)
            
            ratio_ = ratio_computation(sp, org, outputs, patch)

            loss = loss_fn(outputs[patch], org[patch])

            val_loss.append(loss.item())  
            val_ratio.append(ratio_)  
    return torch.mean(torch.Tensor(val_loss)), torch.mean(torch.Tensor(val_ratio))     


def training(args, num_epochs, optimizer_, ddp_model, trainloader, valloader, 
             loss_fn, t, t_r, v, v_r, out_fol, name, lr_decay_count, lr_decay_mul, 
             checkpoint_count):   
    for epoch in range(1, num_epochs+1):
        
        if epoch % lr_decay_count == 0:
            optimizer_.param_groups[0]["lr"] = optimizer_.param_groups[0]["lr"]*lr_decay_mul
        
        train_loss, ratio = [], []
        
        for _, data in tqdm(enumerate(trainloader)):
            ddp_model.train()
            sp, org = data
            sp, org = sp.to(torch.device(args.local_rank)), org.to(torch.device(args.local_rank))
            
            patch = (org != 0)   

            optimizer_.zero_grad()
            outputs = ddp_model(sp)

            ratio_ = ratio_computation(sp, org, outputs, patch)
            
            loss = loss_fn(outputs[patch], org[patch]) 
            
            loss.backward()
            optimizer_.step()     
            
            train_loss.append(loss.item())   
            ratio.append(ratio_)   
        training_loss = torch.mean(torch.Tensor(train_loss))
        training_ratio = torch.mean(torch.Tensor(ratio))
        
        val_loss, val_ratio = validation(valloader, ddp_model, args, loss_fn)
        v.append(val_loss) 
        v_r.append(val_ratio)  
        t.append(training_loss)
        t_r.append(training_ratio)
        
        print(epoch, optimizer_.param_groups[0]["lr"], training_loss, val_loss, training_ratio, val_ratio)
            
        if epoch % checkpoint_count == 0:
            if args.local_rank == 0:
                torch.save(ddp_model.state_dict(), os.path.join(out_fol, name + '_' + str(epoch) + '.pt'))  
                torch.save(torch.Tensor(t), os.path.join(out_fol, name + '_training_loss' + '_' + str(epoch) + '.pt'))
                torch.save(torch.Tensor(v), os.path.join(out_fol, name + '_val_loss' + '_' + str(epoch) + '.pt'))
                torch.save(torch.Tensor(t_r), os.path.join(out_fol, name + '_training_ratio' + '_' + str(epoch) + '.pt'))
                torch.save(torch.Tensor(v_r), os.path.join(out_fol, name + '_val_ratio' + '_' + str(epoch) + '.pt'))
    
    return t, t_r, v, v_r


def plots(num_epochs, t, t_r, v, v_r, out_fol, name):
    plt.plot(range(num_epochs), t, label=f'training loss')
    plt.plot(range(num_epochs), v, label=f'val loss')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(out_fol, name + '.png'))
    plt.close()

    plt.plot(range(num_epochs), t_r, label=f'training ratio')
    plt.plot(range(num_epochs), v_r, label=f'val ratio')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('ratio')
    plt.savefig(os.path.join(out_fol, name + '_ratio' + '.png'))
    plt.close()


