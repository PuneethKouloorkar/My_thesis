import torch.nn as nn
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as F


root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    data_fol = 'Data'
    data_type = 'only_gau'
    #data_type = 'ph_gau'
    return data_fol, data_type


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
    

def hyperparameters():
    batch = 32
    epochs = 1
    start_lr = 1e-7 
    loss_fn = nn.L1Loss()
    crop_size = 256
    num_processes = 2
    lr_decay_mul = 0.4
    lr_decay_epoch_count = 1
    l2_penalty = 1e-4
    return batch, epochs, start_lr, crop_size, loss_fn, num_processes, lr_decay_mul, lr_decay_epoch_count, l2_penalty


def training_vcs_hyperparams():
    min_phantoms = 7    #1
    max_phantoms = 10   #5
    min_alpha = 0
    max_alpha = 0.5
    return min_phantoms, max_phantoms, min_alpha, max_alpha


def ratio_computation(inp, label, op, patch):
    inp_noise = torch.sqrt(torch.mean(torch.square(inp[patch]-label[patch])))
    out_noise = torch.sqrt(torch.mean(torch.square(label[patch]-op[patch])))
    
    if float(inp_noise.item()) == 0: 
        inp_noise = torch.Tensor([1e-4])
    
    ratio_ = round(out_noise.item()/inp_noise.item(), 2)
    return ratio_


def training_vcs(data_loader, model, device, loss_fn, crop_size, optim, lr):
    train_loss, ratio = [], []
    model.train()

    for idx, data in tqdm(enumerate(data_loader)):
        if idx == 75:
            break

        optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"]/0.8

        inp_, label_ = data
        inp_, label_ = inp_.to(device), label_.to(device)

        count = 0.2
        while count <= 0.4:
            i, j, h, w = transforms.RandomCrop.get_params(inp_, output_size=(crop_size, crop_size))
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
        lr.append(optim.param_groups[0]["lr"])
    return train_loss, ratio, lr


def plots(lr, batch_size, crop_size, train_loss):
    plt.plot(range(len(lr)), lr)
    plt.yscale("log")
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    plt.savefig(os.path.join(root_path, str(batch_size), str(str(crop_size)) + '_It_vs_LR.png'))
    plt.close()

    plt.plot(lr, train_loss)
    plt.xscale("log")
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(root_path, str(batch_size), str(str(crop_size)) + '_LR_vs_loss.png'))
    plt.close()
    

def rate_of_change(lr, train_loss, batch_size, crop_size):
    t_l = [train_loss[0]] + train_loss[:-1]
    roc = (torch.Tensor(train_loss) - torch.Tensor(t_l))/torch.Tensor(t_l)     # Rate of change
    plt.plot(lr, roc)
    plt.xscale("log")
    plt.xlabel('Learning rate')
    plt.ylabel('ROC of Loss')
    plt.savefig(os.path.join(root_path, str(batch_size), str(str(crop_size)) + '_LR_vs_lossROC.png'))
    plt.close()

    ma_roc = [roc[0].data]*4
    for i in range(0, len(roc)-5+1):
        ma_roc.append(torch.mean(roc[i:i+5]))     # Moving average of rate of change
    plt.plot(lr, ma_roc)
    plt.xscale("log")
    plt.xlabel('Learning rate')
    plt.ylabel('Moving avg of ROC of Loss')
    plt.savefig(os.path.join(root_path, str(batch_size), str(str(crop_size)) + '_LR_vs_loss_maROC.png'))
    plt.close()



# He initialization-------------------------------------------------------------------
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
