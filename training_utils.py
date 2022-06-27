import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F


class Constant_crop_dataset():
    def __init__(self, sp_image_list, org_image_list):
        self.sp_image_list = sp_image_list
        self.org_image_list = org_image_list
        
    def __len__(self):
        return len(self.sp_image_list)     

    def __getitem__(self, i):
        sp = torch.from_numpy(self.sp_image_list[i])
        org = torch.from_numpy(self.org_image_list[i])
        
        sp, org = sp.reshape(1, 90, 90), org.reshape(1, 90, 90)
        return sp, org


class Variable_crop_dataset():
    def __init__(self, image_list, transforms, phantom, phantom_dict,
                 min_phantoms, max_phantoms, min_alpha, max_alpha):
        self.image_list = image_list
        self.transforms = transforms
        self.phantom = phantom
        self.phantom_dict = phantom_dict
        self.min_phantoms = min_phantoms
        self.max_phantoms = max_phantoms
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def __len__(self):
        return len(self.image_list)     

    def __getitem__(self, i):
        org = self.transforms(Image.open(self.image_list[i]))  
        mask = (org == 0)

        if i <= int(len(self.image_list)/4):
            sp = random.choice([0.1, 0.15, 0.2])*org.clone()   
        else:
            if int(len(self.image_list)/4) < i <= 2*int(len(self.image_list)/4):
                gau = torch.randn_like(org)*round(random.uniform(0, 1), 2)
            elif 2*int(len(self.image_list)/4) < i <= 3*int(len(self.image_list)/4):
                gau = torch.randn_like(org)*round(random.uniform(1, 15), 2)
            else:
                gau = torch.randn_like(org)*round(random.uniform(15, 30), 2)   

            if self.phantom == True:
                name = self.image_list[i].split("/")[-1]
                k = random.randint(self.min_phantoms, self.max_phantoms)
                temp = random.sample(self.phantom_dict[name], k=k)
                sp = 0
                for j in temp:
                    alpha = 1  #round(random.uniform(self.min_alpha, self.max_alpha), 2)
                    sp += alpha*self.transforms(Image.open(j)) 
                sp = random.choice([0.1, 0.15, 0.2])*org.clone() + (sp/k) + gau
            else:
                sp = random.choice([0.1, 0.15, 0.2])*org.clone() + gau

            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0 
        
        return sp/255, org/255


def training_type():
    t_type = 'cons_crop_size_training'           # Constant crop size per batch
    #t_type = 'var_crop_size_training'           # Variable crop size per batch
    
    data_type = 'only_gau'                       # For only Gaussian noise data
    #data_type = 'ph_gau'                        # For Gaussian + phantom noise data

    d_type = t_type.split("_")
    d_type = data_type + '_' + d_type[0][0] + d_type[1][0] + d_type[2][0]

    #multi_gpu_training = True                   # For training using multiple GPUs in parallel 
    multi_gpu_training = False                   # For training using only one GPU 
    
    return t_type, d_type, data_type, multi_gpu_training    


def gen_hyperparams():
    start_lr = 1e-3
    l2_penalty = 1e-4
    epochs = 50
    loss_fn = nn.L1Loss()
    lr_decay_mul = 0.4
    lr_decay_epoch_count = 6
    num_processes = 2
    checkpoint_divisor = 6
    return start_lr, l2_penalty, epochs, loss_fn, lr_decay_mul, lr_decay_epoch_count, num_processes, checkpoint_divisor


def training_vcs_hyperparams():
    batch = 64
    num = 157
    min_crop_size = 28
    max_crop_size = 128
    min_phantoms = 7   #1
    max_phantoms = 10  #5
    min_alpha = 0
    max_alpha = 0.5
    return batch, num, min_crop_size, max_crop_size, min_phantoms, max_phantoms, min_alpha, max_alpha


def training_ccs_hyperparams():
    batch = 128
    return batch


def ratio_computation(inp, label, op, patch):
    inp_noise = torch.sqrt(torch.mean(torch.square(inp[patch]-label[patch])))
    out_noise = torch.sqrt(torch.mean(torch.square(label[patch]-op[patch])))
    
    if float(inp_noise.item()) == 0: 
        inp_noise = torch.Tensor([1e-4])
    
    ratio_ = round(out_noise.item()/inp_noise.item(), 2)
    return ratio_


def training_ccs(data_loader, model, device, loss_fn, optimizer_):
    train_loss, ratio = [], []
    model.train()

    for idx, data in tqdm(enumerate(data_loader)):
        inp, label = data
        inp, label = inp.to(device), label.to(device)
        patch = (label != 0) 

        optimizer_.zero_grad()
        op = model(inp) 

        ratio_ = ratio_computation(inp, label, op, patch)
        
        op_patches_pixels = op[patch]
        label_patches_pixels = label[patch]
        loss = loss_fn(op_patches_pixels, label_patches_pixels)

        loss.backward()
        optimizer_.step()     
                
        train_loss.append(loss.item()) 
        ratio.append(ratio_)
    
    training_loss = torch.mean(torch.Tensor(train_loss))
    t_ratio = torch.mean(torch.Tensor(ratio))
    return training_loss, t_ratio


def validation_ccs(trained_model, data_loader, loss_fn, device):
    val_loss, val_ratio = [], []
    trained_model.eval()     
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            inp, label = data
            inp, label = inp.to(device), label.to(device)
            op = trained_model(inp)
            
            patch = (label != 0)
            ratio_ = ratio_computation(inp, label, op, patch)

            op_patches_pixels = op[patch]
            label_patches_pixels = label[patch]
            loss = loss_fn(op_patches_pixels, label_patches_pixels)

            val_loss.append(loss.item())
            val_ratio.append(ratio_)   
    return torch.mean(torch.Tensor(val_loss)), torch.mean(torch.Tensor(val_ratio))    


def training_vcs(data_loader, model, device, loss_fn, num, min_crop_size, max_crop_size, optim):
    train_loss, ratio = [], []
    model.train()

    for idx, data in tqdm(enumerate(data_loader)):
        inp_, label_ = data
        inp_, label_ = inp_.to(device), label_.to(device)

        for _ in range(num):
            count = 0.2
            cs = random.randint(min_crop_size, max_crop_size)
            while count <= 0.4:
                i, j, h, w = transforms.RandomCrop.get_params(inp_, output_size=(cs, cs))
                inp = F.crop(inp_, i, j, h, w)
                label = F.crop(label_, i, j, h, w)
                count = label.count_nonzero()/label.numel()        
            
            patch = (label != 0) 

            optim.zero_grad()
            op = model(inp) 

            ratio_ = ratio_computation(inp, label, op, patch)
            
            op_patch_pixels = op[patch]
            label_patch_pixels = label[patch]
            loss = loss_fn(op_patch_pixels, label_patch_pixels)

            loss.backward()
            optim.step()     
                    
            train_loss.append(loss.item()) 
            ratio.append(ratio_)
    training_loss = torch.mean(torch.Tensor(train_loss))
    t_ratio = torch.mean(torch.Tensor(ratio))
    return training_loss, t_ratio


def validation_vcs(trained_model, data_loader, loss_fn, device, num, min_crop_size, max_crop_size):
    val_loss, val_ratio = [], []
    trained_model.eval()     
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            inp_, label_ = data
            inp_, label_ = inp_.to(device), label_.to(device)

            for _ in range(num):
                count = 0.2
                cs = random.randint(min_crop_size, max_crop_size)
                while count <= 0.4:
                    i, j, h, w = transforms.RandomCrop.get_params(inp_, output_size=(cs, cs))
                    inp = F.crop(inp_, i, j, h, w)
                    label = F.crop(label_, i, j, h, w)
                    count = label.count_nonzero()/label.numel()   
                
                patch = (label != 0)
                op = trained_model(inp)
                
                ratio_ = ratio_computation(inp, label, op, patch)

                op_patch_pixels = op[patch]
                label_patch_pixels = label[patch]
                loss = loss_fn(op_patch_pixels, label_patch_pixels)

                val_loss.append(loss.item())
                val_ratio.append(ratio_)   
    return torch.mean(torch.Tensor(val_loss)), torch.mean(torch.Tensor(val_ratio))    


def plots(num_epochs, training_loss, validation_loss, training_ratio,
          validation_ratio, out_fol, current_archi, data_type):
    
    plt.plot(range(num_epochs), training_loss, label=f'training loss')
    plt.plot(range(num_epochs), validation_loss, label=f'val loss')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(out_fol, current_archi + '_' + data_type + '.png'))
    plt.close()

    plt.plot(range(num_epochs), training_ratio, label=f'training ratio')
    plt.plot(range(num_epochs), validation_ratio, label=f'val ratio')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('ratio')
    plt.savefig(os.path.join(out_fol, current_archi + '_ratio_' + data_type + '.png'))
    plt.close()
