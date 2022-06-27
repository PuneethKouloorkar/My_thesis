import bm3d
import cv2
import numpy as np
import os
from torchvision.utils import save_image
from torchvision import transforms
from bm3d_utils import paths, dirs_files_bm3d, dirs_files_dl
from bm3d_utils import compute_psnr, psnr_mean, noise_level


#------------------------------------------Note--------------------------------------------------------
# The Clean interferograms are expected to be 8-bit images (data range: [0-255])
# Provide paths for the clean interferograms and DL denoised interferograms in the 'paths' method of bm3d_utils.py
#------------------------------------------Note--------------------------------------------------------


# Load the paths, text files, directories, lists and noise level---------------------------------------
clean_intfs_path, out_fol, DL_denoised_intfs_path = paths()
sigma = noise_level()
f1, f2, f3, _bm3d_c_n, _bm3d_n_d, _bm3d_c_d = dirs_files_bm3d(sigma)
f4, f5, f6, _dl_c_n, _dl_n_d, _dl_c_d = dirs_files_dl(sigma)


# BM3D and DL PSNR computation-------------------------------------------------------------------------
for i in range(len(clean_intfs_path)):
    clean = cv2.imread(clean_intfs_path[i])
    clean = clean[:,:,0]
    
    noisy = clean + np.random.normal(loc=0, scale=sigma, size=clean.shape)
    noisy = np.clip(noisy, 0, 255)

    mask = (clean != 0)

    # BM3D---------------------------------------------------------------------------------------------
    denoised = bm3d.bm3d(noisy, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES) 
    denoised = denoised.astype(np.uint8)
    
    compute_psnr(clean, noisy, mask, f1, i, _bm3d_c_n)
    compute_psnr(noisy, denoised, mask, f2, i, _bm3d_n_d)
    compute_psnr(clean, denoised, mask, f3, i, _bm3d_c_d)
    
    cv2.imwrite(os.path.join(out_fol, 'BM3D_denoised', f'{i}.bmp', denoised))
    d_tensor = transforms.ToTensor()(denoised)
    save_image(d_tensor, os.path.join(out_fol, 'BM3D_denoised_norm', f'{i}.png'))


    # DL-------------------------------------------------------------------------------------------------
    denoised = cv2.imread(os.path.join(out_fol, DL_denoised_intfs_path, f'{i}.bmp'))
    denoised = denoised[:,:,0]
    
    compute_psnr(clean, noisy, mask, f4, i, _dl_c_n)
    compute_psnr(noisy, denoised, mask, f5, i, _dl_n_d)
    compute_psnr(clean, denoised, mask, f6, i, _dl_c_d) 
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()


# Check if all the examples are denoised-----------------------------------------------------------------
print(len(_bm3d_c_n), len(_bm3d_n_d), len(_bm3d_c_d), len(_dl_c_n), len(_dl_n_d), len(_dl_c_d))


# Compute mean PSNR of all the examples------------------------------------------------------------------
psnr_mean(sigma, _bm3d_c_n, _bm3d_n_d, _bm3d_c_d, _dl_c_n, _dl_n_d, _dl_c_d)
