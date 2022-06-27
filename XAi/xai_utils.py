import torch.nn as nn
import os
import torch
from glob import glob
from PIL import Image
import random
import numpy as np
import cv2


root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    test_intfs = []
    for i in range(1, 7):
        data = glob(f'/home/kouloo01/SharedData/OPTIK/MachineLearningData/FixedInterferograms/testset_{i}/pos1/*.bmp')
        data.sort
        test_intfs.sort()

    data_fol = 'Data'
    model_path = ''
    return test_intfs, data_fol, model_path


def hyperparams():
    num_phantoms = 8  #3
    min_alpha = 0
    max_alpha = 0.5
    batch = 1
    return num_phantoms, min_alpha, max_alpha, batch


class XAi_dataset():
    def __init__(self, image_list, transforms, phantom, phantom_dict,
                 num_phantoms, min_alpha, max_alpha):
        self.image_list = image_list
        self.transforms = transforms
        self.phantom = phantom
        self.phantom_dict = phantom_dict
        self.num_phantoms = num_phantoms
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def __len__(self):
        return len(self.image_list)     

    def __getitem__(self, i):
        org = self.transforms(Image.open(self.image_list[i]))  
        mask = (org == 0)

        gau = torch.randn_like(org)*10

        if self.phantom == True:
            name = self.image_list[i].split("/")[-1]
            temp = random.sample(self.phantom_dict[name], k=self.num_phantoms)
            sp = 0
            for j in temp:
                alpha = 1  #round(random.uniform(self.min_alpha, self.max_alpha), 2)
                sp += alpha*self.transforms(Image.open(j)) 
            sp = random.choice([0.1, 0.15, 0.2])*org.clone() + (sp/self.num_phantoms) + gau
        else:
            sp = random.choice([0.1, 0.15, 0.2])*org.clone() + gau

            sp = torch.clamp(sp, 0, 255)
            sp[mask] = 0 
        
        return sp/255, org/255


def create_dirs():
    os.makedirs(os.path.join(root_path, 'gbp_greyscale_heatmap'))
    os.makedirs(os.path.join(root_path, 'gbp_colored_heatmap'))

    os.makedirs(os.path.join(root_path, 'sa_greyscale_heatmap'))
    os.makedirs(os.path.join(root_path, 'sa_colored_heatmap'))

    os.makedirs(os.path.join(root_path, 'model_input'))
    os.makedirs(os.path.join(root_path, 'model_output_masked'))
    os.makedirs(os.path.join(root_path, 'model_output_unmasked'))
    os.makedirs(os.path.join(root_path, 'label'))


class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0]

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            grad[grad > 0] = 1 
            
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad
            return (new_grad_in,)

        # Get module, here is only for alexnet, if it is other, you need to modify
        modules = self.model.named_children()

        # for module_name, module in modules:
        #   print(module_name)

        for module_name, module in modules:
            #if module_name != 'conv7':
            for m in module:
                if isinstance(m, nn.ReLU):
                    m.register_forward_hook(forward_hook_fn)
                    m.register_full_backward_hook(backward_hook_fn)            

        # Register hook for the first convolutional layer
        mod = self.model.b1.named_children()
        for _, m in mod:
            if isinstance(m, nn.Conv2d):
                m.register_full_backward_hook(first_layer_hook_fn)
                break

    def visualize(self, input_image, wd, i, mask):
        outputs = self.model(input_image)         
        out_s = outputs.clone()   
        
        self.model.zero_grad()
        outputs.sum().backward()
        out_s[mask] = 0
        
        result = self.image_reconstruction
        
        return result, outputs, out_s

def normalize(I):
    # Normalize gradient map, first normalize to mean=0 std=1
    norm = (I-I.mean())/I.std()
    # Set the gradient values ​​other than 0 and 1 to 0 and 1 respectively
    norm = norm.clip(0, 1)
    
    return norm


def sensitivity_analysis(inp):
    # Grey heatmap---------------------------------------------------------
    sa_grey_hm = inp.grad.data.abs()
    sa_grey_hm = normalize(sa_grey_hm)

    # Colored heatmap------------------------------------------------------
    temp = sa_grey_hm.cpu().numpy().squeeze(0)
    temp = np.moveaxis(temp, 0, -1)
    temp = np.uint8(255 * temp)     # convert to RGB 
    sa_col_hm = cv2.applyColorMap(temp, cv2.COLORMAP_JET)  
    
    return sa_grey_hm, sa_col_hm

def gbp_heatmaps(I):
    # Grey heatmap---------------------------------------------------------
    gbp_grey_hm = normalize(I)

    # Colored heatmap------------------------------------------------------
    temp = gbp_grey_hm.cpu().numpy().squeeze(0)
    temp = np.moveaxis(temp, 0, -1)
    temp = np.uint8(255 * temp)     # convert to RGB 
    gbp_col_hm = cv2.applyColorMap(temp, cv2.COLORMAP_JET)  
    
    return gbp_grey_hm, gbp_col_hm 
