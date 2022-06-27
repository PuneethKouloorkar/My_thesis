from glob import glob
from PIL import Image
from torchvision.utils import save_image
import os
from collections import OrderedDict


root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    test_list = glob('/home/kouloo01/SharedData/OPTIK/MachineLearningData/Real images/a30-26hpx_ptb_24500_real3_real6/*.bmp')
    test_list.sort()

    test_list_masks = glob('/home/kouloo01/SharedData/OPTIK/MachineLearningData/Real images/a30-26hpx_ptb_24500_real3_real6/*.png')
    test_list_masks.sort()

    model_path = ''
    return test_list, test_list_masks, model_path


def dirs():
    os.makedirs(os.path.join(root_path, 'Noisy_input'))
    os.makedirs(os.path.join(root_path, 'Denosied_input'))



class Masking():
    def __init__(self, image_list, transforms, mask_list):
        self.image_list = image_list
        self.transforms = transforms
        self.mask_list = mask_list

    def __len__(self):
        return len(self.image_list)     

    def __getitem__(self, i):
        sp = self.transforms(Image.open(self.image_list[i])) 

        mask_index = self.image_list[i].split("/")[-1][1]
        intf_mask = self.transforms(Image.open(self.mask_list[int(mask_index)-1]))
        
        zero_mask = (intf_mask == 0)
        sp[zero_mask] = 0
        
        save_image(sp, os.path.join(root_path, 'Noisy_input', f'{i}.png'))
        
        return sp/255, zero_mask


def remove_module(model, state_dict):
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:]                               # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()  
    
    return model
