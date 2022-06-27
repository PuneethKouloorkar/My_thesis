import random
import torch
import pickle
from PIL import Image
import math
from torchvision.utils import save_image
from tqdm import tqdm
from glob import glob
import os
from path_file import m_gen

# # Global Variable---------------------------------------------------
out_fol = m_gen()
# out_fol = ''
# # Global Variable---------------------------------------------------

# Seperating phantoms based on mask and shift--------------------------------------------
def mask_shift(path):
    ph_list_1 = pickle.load(open(os.path.join(path, 'Phantom_imgs_1.pkl'), "rb"))
    ph_list_2 = pickle.load(open(os.path.join(path, 'Phantom_imgs_2.pkl'), "rb"))
    ph_list_3 = pickle.load(open(os.path.join(path, 'Phantom_imgs_3.pkl'), "rb"))
    ph_list_4 = pickle.load(open(os.path.join(path, 'Phantom_imgs_4.pkl'), "rb"))

    m1img0, m1img1, m1img2, m1img3, m1img4 = [], [], [], [], []
    m2img0, m2img1, m2img2, m2img3, m2img4 = [], [], [], [], []
    m3img0, m3img1, m3img2, m3img3, m3img4 = [], [], [], [], []
    m4img0, m4img1, m4img2, m4img3, m4img4 = [], [], [], [], []

    # Brute force--------------------------------------------------------------
    for i in ph_list_1:
        if i.split("\\")[-1] == 'm1_i0_img0.bmp':
            m1img0.append(i)
        elif i.split("\\")[-1] == 'm1_i0_img1.bmp':
            m1img1.append(i)
        elif i.split("\\")[-1] == 'm1_i0_img2.bmp':
            m1img2.append(i)
        elif i.split("\\")[-1] == 'm1_i0_img3.bmp':
            m1img3.append(i)
        else:
            m1img4.append(i)  

    for i in ph_list_2:
        if i.split("\\")[-1] == 'm2_i0_img0.bmp':
            m2img0.append(i)
        elif i.split("\\")[-1] == 'm2_i0_img1.bmp':
            m2img1.append(i)
        elif i.split("\\")[-1] == 'm2_i0_img2.bmp':
            m2img2.append(i)
        elif i.split("\\")[-1] == 'm2_i0_img3.bmp':
            m2img3.append(i)
        else:
            m2img4.append(i)

    for i in ph_list_3:
        if i.split("\\")[-1] == 'm3_i0_img0.bmp':
            m3img0.append(i)
        elif i.split("\\")[-1] == 'm3_i0_img1.bmp':
            m3img1.append(i)
        elif i.split("\\")[-1] == 'm3_i0_img2.bmp':
            m3img2.append(i)
        elif i.split("\\")[-1] == 'm3_i0_img3.bmp':
            m3img3.append(i)
        else:
            m3img4.append(i)

    for i in ph_list_4:
        if i.split("\\")[-1] == 'm4_i0_img0.bmp':
            m4img0.append(i)
        elif i.split("\\")[-1] == 'm4_i0_img1.bmp':
            m4img1.append(i)
        elif i.split("\\")[-1] == 'm4_i0_img2.bmp':
            m4img2.append(i)
        elif i.split("\\")[-1] == 'm4_i0_img3.bmp':
            m4img3.append(i)
        else:
            m4img4.append(i)
    # Brute force--------------------------------------------------------------

    print(len(m1img0), len(m1img1), len(m1img2), len(m1img3), len(m1img4))
    print(len(m2img0), len(m2img1), len(m2img2), len(m2img3), len(m2img4))
    print(len(m3img0), len(m3img1), len(m3img2), len(m3img3), len(m3img4))
    print(len(m4img0), len(m4img1), len(m4img2), len(m4img3), len(m4img4))

    pickle.dump(m1img0, open(os.path.join(path, 'm10.pkl'), "wb"))
    pickle.dump(m1img1, open(os.path.join(path, 'm11.pkl'), "wb"))
    pickle.dump(m1img2, open(os.path.join(path, 'm12.pkl'), "wb"))
    pickle.dump(m1img3, open(os.path.join(path, 'm13.pkl'), "wb"))
    pickle.dump(m1img4, open(os.path.join(path, 'm14.pkl'), "wb"))
    pickle.dump(m2img0, open(os.path.join(path, 'm20.pkl'), "wb"))
    pickle.dump(m2img1, open(os.path.join(path, 'm21.pkl'), "wb"))
    pickle.dump(m2img2, open(os.path.join(path, 'm22.pkl'), "wb"))
    pickle.dump(m2img3, open(os.path.join(path, 'm23.pkl'), "wb"))
    pickle.dump(m2img4, open(os.path.join(path, 'm24.pkl'), "wb"))
    pickle.dump(m3img0, open(os.path.join(path, 'm30.pkl'), "wb"))
    pickle.dump(m3img1, open(os.path.join(path, 'm31.pkl'), "wb"))
    pickle.dump(m3img2, open(os.path.join(path, 'm32.pkl'), "wb"))
    pickle.dump(m3img3, open(os.path.join(path, 'm33.pkl'), "wb"))
    pickle.dump(m3img4, open(os.path.join(path, 'm34.pkl'), "wb"))
    pickle.dump(m4img0, open(os.path.join(path, 'm40.pkl'), "wb"))
    pickle.dump(m4img1, open(os.path.join(path, 'm41.pkl'), "wb"))
    pickle.dump(m4img2, open(os.path.join(path, 'm42.pkl'), "wb"))
    pickle.dump(m4img3, open(os.path.join(path, 'm43.pkl'), "wb"))
    pickle.dump(m4img4, open(os.path.join(path, 'm44.pkl'), "wb"))
# Seperating phantoms based on mask and shift--------------------------------------------

mask_shift(f'{out_fol}')