import torch.nn as nn
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    data_fol = 'Data'
    data_type = 'only_gau'
    #data_type = 'ph_gau'
    return data_fol, data_type


class Constant_crop_size_training():
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
    

def hyperparameters():
    batch = 32
    epochs = 6
    start_lr = 1e-3 
    loss_fn = nn.L1Loss()
    num_processes = 2
    lr_decay_mul = 0.4
    lr_decay_epoch_count = 1
    return batch, epochs, start_lr, loss_fn, num_processes, lr_decay_mul, lr_decay_epoch_count


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


# He initialization-------------------------------------------------------------------
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
