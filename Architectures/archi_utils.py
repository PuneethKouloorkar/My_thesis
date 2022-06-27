import torch.nn as nn
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    data_fol = 'Data'
    #data_type = 'only_gau'
    data_type = 'ph_gau'
    return data_fol, data_type


def gen_hyperparams():
    batch = 48
    epochs = 5
    start_lr = 1e-3 
    loss_fn = nn.L1Loss()
    l2_penalty = 1e-4
    num_processes = 2
    lr_decay_mul = 0.8
    lr_decay_epoch_count = 1
    return batch, epochs, start_lr, loss_fn, l2_penalty, num_processes, lr_decay_mul, lr_decay_epoch_count


def training_hyperparams():
    num = 16
    crop_size = 120
    min_phantoms = 1
    max_phantoms = 5
    min_alpha = 0
    max_alpha = 0.5
    phantom = True
    count = 1
    return num, crop_size, min_phantoms, max_phantoms, min_alpha, max_alpha, phantom, count


class Superposition():
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
            sp = org.clone()   #random.choice([0.1, 0.15, 0.2])*org.clone()   
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
                    alpha = round(random.uniform(self.min_alpha, self.max_alpha), 2)   #1
                    sp += alpha*self.transforms(Image.open(j)) 
                sp = org.clone() + (sp/k) + gau   #random.choice([0.1, 0.15, 0.2])*org.clone() + (sp/k) + gau
            else:
                sp = org.clone() + gau   #random.choice([0.1, 0.15, 0.2])*org.clone() + gau

            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0 
        
        return sp/255, org/255


def ratio_computation(inp, label, op, patch):
    inp_noise = torch.sqrt(torch.mean(torch.square(inp[patch]-label[patch])))
    out_noise = torch.sqrt(torch.mean(torch.square(label[patch]-op[patch])))
    
    if float(inp_noise.item()) == 0: 
        inp_noise = torch.Tensor([1e-4])
    
    ratio_ = round(out_noise.item()/inp_noise.item(), 2)
    return ratio_


def training(data_loader, model, device, loss_fn, num, crop_size, optim):
    train_loss, ratio = [], []
    model.train()

    for idx, data in tqdm(enumerate(data_loader)):
        inp_, label_ = data
        inp_, label_ = inp_.to(device), label_.to(device)

        for _ in range(num):
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
    training_loss = torch.mean(torch.Tensor(train_loss))
    t_ratio = torch.mean(torch.Tensor(ratio))
    return training_loss, t_ratio


def validation(trained_model, data_loader, loss_fn, device, num, crop_size):
    val_loss, val_ratio = [], []
    trained_model.eval()     
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader)):
            inp_, label_ = data
            inp_, label_ = inp_.to(device), label_.to(device)

            for _ in range(num):
                count = 0.2
                while count <= 0.4:
                    i, j, h, w = transforms.RandomCrop.get_params(inp_, output_size=(crop_size, crop_size))
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
    
    plt.plot(range(1, num_epochs+1), training_loss, label=f'training loss')
    plt.plot(range(1, num_epochs+1), validation_loss, label=f'val loss')
    plt.yscale("log")
    plt.legend()
    plt.title(f"{current_archi}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(out_fol, current_archi + '_' + data_type + '.png'))
    plt.close()

    plt.plot(range(1, num_epochs+1), training_ratio, label=f'training ratio')
    plt.plot(range(1, num_epochs+1), validation_ratio, label=f'val ratio')
    plt.yscale("log")
    plt.legend()
    plt.title(f"{current_archi}")
    plt.xlabel('Epoch')
    plt.ylabel('ratio')
    plt.savefig(os.path.join(out_fol, current_archi + '_ratio_' + data_type + '.png'))
    plt.close()


# He initialization-------------------------------------------------------------------
# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)