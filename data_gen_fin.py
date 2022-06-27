import random
import pickle
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm
# import pandas as pd
import os
# from torchvision.utils import save_image
from path_file import d_gen
from training_utils import training_vcs_hyperparams


# from path_file import df_train, df_val, df_test, fol as data_fol, name as file_name, out_fol1 as out_fol
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

# Global Variables-----------------------------------------------------
# df_train = pd.DataFrame(columns=['train_sp', 'train_org'])
# df_val = pd.DataFrame(columns=['val_sp', 'val_org'])
# df_test = pd.DataFrame(columns=['test_sp', 'test_org'])
# data_fol = ''
# file_name = ''
# out_fol = ''
df_train, df_val, df_test, data_fol, file_name, out_fol, phantom, masking = d_gen()
# phantom = False
# masking = True
_, num, _, _, min_phantoms, max_phantoms, min_alpha, max_alpha = training_vcs_hyperparams()
crop_size = 90
# Global Variables-----------------------------------------------------


phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(f'{data_fol}/m{mask}{shift}.pkl', "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data


intf_list = pickle.load(open(f'{data_fol}/intf_imgs.pkl', "rb"))
train_x = intf_list[:int(0.90*len(intf_list))]     
val_x = intf_list[int(0.90*len(intf_list)):] 
print(len(train_x), len(val_x))


if phantom == True:
    print('Generating phantom + Gaussian examples...')
else:
    print('Generating Gaussian examples...')


img_t = transforms.Compose([transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor))])


# Remove background pixels----------------------------------------------
img_ct = transforms.CenterCrop((1260, 1260))   


def save_data(img_list, pd_df, file_name):
    count = 0
    for idx, data in tqdm(enumerate(img_list)):
        org = img_t(Image.open(data))    # 0 to 255, FloatTensor
        
        mask = (org == 0)

        if idx <= int(len(img_list)/4):
            sp = random.choice([0.1, 0.15, 0.2])*org.clone()
        else:
            if int(len(img_list)/4) < idx <= 2*int(len(img_list)/4):
                gau = torch.randn_like(org)*round(random.uniform(0, 1), 2)
            elif 2*int(len(img_list)/4) < idx <= 3*int(len(img_list)/4):
                gau = torch.randn_like(org)*round(random.uniform(1, 15), 2)
            else:
                gau = torch.randn_like(org)*round(random.uniform(15, 30), 2)            
            
            if phantom == True:
                name = data.split("/")[-1]
                k = random.randint(min_phantoms, max_phantoms)
                temp = random.sample(phantom_dict[name], k=k)
                sp = 0
                for i in temp:
                    alpha = 1   #round(random.uniform(min_alpha, max_alpha), 2)
                    sp += alpha*img_t(Image.open(i))
                sp = random.choice([0.1, 0.15, 0.2])*org.clone() + (sp/k) + gau
            else:
                sp = random.choice([0.1, 0.15, 0.2])*org.clone() + gau
            sp = torch.clamp(sp, 0, 255)
            
            if masking == True:
                sp[mask] = 0
        
        sp, org = sp/255., org/255.
        sp, org = img_ct(sp), img_ct(org)
        
        for i in range(num):
            count_nz = 0.2
            while count_nz <= 0.4:
                i, j, h, w = transforms.RandomCrop.get_params(sp, output_size=(crop_size, crop_size))
                sp_ = F.crop(sp, i, j, h, w)
                org_ = F.crop(org, i, j, h, w)
                count_nz = org_.count_nonzero()/org_.numel()
            
            # mask = (org_ == 0)
            # sp_, org_, mask = sp_.reshape(crop_size*crop_size), org_.reshape(crop_size*crop_size), mask.reshape(crop_size*crop_size)
            # pd_df.loc[count] = [sp_.numpy(), org_.numpy(), mask.numpy()]
            sp_, org_ = sp_.reshape(crop_size*crop_size), org_.reshape(crop_size*crop_size)
            pd_df.loc[count] = [sp_.numpy(), org_.numpy()]

            count += 1
       
    pd_df = pd_df.reset_index(drop=True)        
    pd_df.to_feather(os.path.join(out_fol, file_name))  


save_data(train_x, df_train, f'train_{file_name}.ftr')
save_data(val_x, df_val, f'val_{file_name}.ftr')
