from torchvision.utils import save_image
from PIL import Image
import pickle
import os
import random
import torch
from tqdm import tqdm
import cv2
import numpy as np


# Load all the test images-----------------------------------------------------------
test_list = pickle.load(open(os.path.join('Data', 'intf_imgs_test_1.pkl'), 'rb'))
test_list.sort()


def params():
    min_alpha = 0
    max_alpha = 0.5
    num_phantoms = 3
    count = 2
    crop_size = 1024  
    
    # ('', int) = (test case name, beta)----------------------------------------------
    test_cases = [('only_gau=0.5', 0.5), ('only_gau=5', 5), ('only_gau=10', 10),
                  ('only_phantom', 0),('ph+gau=0.5', 0.5), ('ph+gau=5', 5), ('ph+gau=10', 10)
                 ]
    
    return min_alpha, max_alpha, num_phantoms, count, crop_size, test_cases


class Superposition_test():
    def __init__(self, image_list, transforms, phantom_dict, beta, fol_name,
                min_alpha, max_alpha, num_phantoms):
        self.image_list = image_list
        self.transforms = transforms
        self.phantom_dict = phantom_dict
        self.beta = beta
        self.fol_name = fol_name
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.num_phantoms = num_phantoms

    def __len__(self):
        return len(self.image_list)     

    def __getitem__(self, i):
        org = self.transforms(Image.open(self.image_list[i]))  # 0 to 255
        mask = (org == 0)

        if self.fol_name.startswith('ph+gau'):
            name = self.image_list[i].split("/")[-1]
            temp = random.sample(self.phantom_dict[name], k=self.num_phantoms)
            sp = 0
            for phs in temp:
                alpha = round(random.uniform(self.min_alpha, self.max_alpha), 2)
                sp += alpha*self.transforms(Image.open(phs))
            sp = org + (sp/self.num_phantoms) + torch.randn_like(org)*self.beta
            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0
        
        elif self.fol_name.startswith('only_gau'):
            sp = org + torch.randn_like(org)*self.beta
            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0
        
        elif self.fol_name.startswith('only_ph'):
            name = self.image_list[i].split("/")[-1]
            temp = random.sample(self.phantom_dict[name], k=self.num_phantoms)
            sp = 0
            for phs in temp:
                alpha = round(random.uniform(0, 0.5), 2)
                sp += alpha*self.transforms(Image.open(phs))
            sp = org + (sp/self.num_phantoms)
            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0

        return sp/255, org/255 


def sup_ph_gau(data, transforms, phantom_dict, num_phantoms, min_alpha, max_alpha, beta):
    org = transforms(Image.open(data))  
    mask = (org == 0)
    name = data.split("/")[-1]
    temp = random.sample(phantom_dict[name], k=num_phantoms)
    sp = 0
    for phs in temp:
        alpha = round(random.uniform(min_alpha, max_alpha), 2)
        sp += alpha*transforms(Image.open(phs))
    sp = org + (sp/num_phantoms) + torch.randn_like(org)*beta
    sp = torch.clamp(sp, 0, 255)
    sp[mask] = 0
    return sp/255, org/255 


def sup_only_gau(data, beta, transforms):
    org = transforms(Image.open(data))
    mask = (org == 0)
    sp = org + torch.randn_like(org)*beta
    sp = torch.clamp(sp, 0, 255)
    sp[mask] = 0
    return sp/255, org/255 


def sup_only_ph(data, transforms, phantom_dict, num_phantoms, min_alpha, max_alpha):
    org = transforms(Image.open(data))
    mask = (org == 0)
    name = data.split("/")[-1]
    temp = random.sample(phantom_dict[name], k=num_phantoms)
    sp = 0
    for phs in temp:
        alpha = round(random.uniform(min_alpha, max_alpha), 2)
        sp += alpha*transforms(Image.open(phs))
    sp = org + (sp/num_phantoms)
    sp = torch.clamp(sp, 0, 255)
    sp[mask] = 0
    return sp/255, org/255 


def recons_save_image(inp, final_dir, fol_name, set_name, intf_name):
    temp = inp.reshape(inp.shape[2], inp.shape[3], 1).cpu().numpy()
    temp = temp*255.
    temp = temp.astype(np.uint8)
    cv2.imwrite(os.path.join(final_dir, fol_name, set_name, f'{intf_name}.bmp'), temp)


def testing_proc(test_list, testing_params, device, folders, TestIntfNorm, TestSpFinalNorm, 
                 TestOpNorm, final_dir, count, c_size, model):
    
    if folders.startswith('8_bit'):
        os.makedirs(f'{final_dir}/clean_intf/{set_name}')
        os.makedirs(f'{final_dir}/noisy_intf/{set_name}')
        os.makedirs(f'{final_dir}/denoised_intf/{set_name}')    
    
    beta, transforms, phantom_dict, num_phantoms, min_alpha, max_alpha = testing_params

    #for i, data in tqdm(enumerate(dataloader)):
    for i, data in tqdm(enumerate(test_list)):
        if folders.startswith('only_gau'):
            inp, label = sup_only_gau(data, beta, transforms)
        elif folders.startswith('only_ph'):
            inp, label = sup_only_ph(data, transforms, phantom_dict, num_phantoms, min_alpha, max_alpha)
        elif folders.startswith('ph+gau'):
            inp, label = sup_ph_gau(data, transforms, phantom_dict, num_phantoms, min_alpha, max_alpha, beta)
        
        #inp, label = data      
        inp, label = inp.to(device), label.to(device)

        outputs = torch.zeros_like(inp)

        if not folders.startswith('8_bit'):
            # Save the input and label image--------------------------------------------------------
            TestIntfNorm.append(label.cpu())
            TestSpFinalNorm.append(inp.cpu())
            #save_image(label, os.path.join(final_dir, 'label_norm', f'label_norm_{i}.png'))
            #save_image(inp, os.path.join(final_dir, 'inp_norm', f'inp_norm_{i}.png'))
        else:
            set_name = test_list[i].split("/")[-2]
            intf_name = test_list[i].split("/")[-1]
            intf_name = intf_name.split(".")[0] 
            
            recons_save_image(label, final_dir, 'clean_intf', set_name, intf_name)
            recons_save_image(inp, final_dir, 'noisy_intf', set_name, intf_name)

        # Generate the denoised image---------------------------------------------------------------
        for m in range(count):
            for n in range(count):
                patch = (label[:, :, m*c_size:(m+1)*c_size, n*c_size:(n+1)*c_size] == 0)
                outputs[:, :, m*c_size:(m+1)*c_size, n*c_size:(n+1)*c_size] = model(inp[:, :, m*c_size:(m+1)*c_size, n*c_size:(n+1)*c_size])
                outputs[:, :, m*c_size:(m+1)*c_size, n*c_size:(n+1)*c_size][patch] = 0
        
        if not folders.startswith('8_bit'):
            # Save the output image-----------------------------------------------------------------
            TestOpNorm.append(outputs.cpu())
            #save_image(outputs, os.path.join(final_dir, 'out_norm', f'out_norm_{i}.png'))
        else:
            recons_save_image(outputs, final_dir, 'denoised_intf', set_name, intf_name)
    
    return TestSpFinalNorm, TestIntfNorm, TestOpNorm


def remove_modules(model_path):
    current_state_dict = torch.load(model_path)
    
    from collections import OrderedDict
    
    new_state_dict = OrderedDict()
    for k_, v_ in current_state_dict.items():
        key_name = k_[7:]       # remove "module."
        new_state_dict[key_name] = v_

    return new_state_dict
    
    #model.load_state_dict(new_state_dict)
