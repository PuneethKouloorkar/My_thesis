import os
import numpy as np

root_path = os.path.dirname(os.path.abspath(__file__))


def paths():
    clean_intfs_path = ''
    DL_denoised_intfs_path = ''
    return clean_intfs_path, root_path, DL_denoised_intfs_path


def noise_level():
    sigma = 0.5        # Standard deviation of the Gaussian
    os.makedirs(os.path.join(root_path, f'sigma_{sigma}'))
    return sigma


def dirs_files_bm3d(sig):
    os.makedirs(os.path.join(root_path, f'sigma_{sig}', 'BM3D_denoised'))
    os.makedirs(os.path.join(root_path, f'sigma_{sig}', 'BM3D_denoised_norm'))

    # BM3D----------------------------------------------------------------------------------------------
    f1 = open(os.path.join(root_path, f'sigma_{sig}', 'BM3D_PSNR_c_n.txt'), 'w')         # PSNR(clean, noisy)
    f2 = open(os.path.join(root_path, f'sigma_{sig}', 'BM3D_PSNR_n_d.txt'), 'w')         # PSNR(noisy, denoised)
    f3 = open(os.path.join(root_path, f'sigma_{sig}', 'BM3D_PSNR_c_d.txt'), 'w')         # PSNR(clean, denoised)
    _bm3d_c_n, _bm3d_n_d, _bm3d_c_d = [], [], []                                         # To store the PSNR values

    return f1, f2, f3, _bm3d_c_n, _bm3d_n_d, _bm3d_c_d, 


def dirs_files_dl(sig):
    # DL------------------------------------------------------------------------------------------------
    f4 = open(os.path.join(root_path, f'sigma_{sig}', 'DL_PSNR_c_n.txt'), 'w')           # PSNR(clean, noisy)
    f5 = open(os.path.join(root_path, f'sigma_{sig}', 'DL_PSNR_n_d.txt'), 'w')           # PSNR(noisy, denoised)
    f6 = open(os.path.join(root_path, f'sigma_{sig}', 'DL_PSNR_c_d.txt'), 'w')           # PSNR(clean, denoised)
    _dl_c_n, _dl_n_d, _dl_c_d = [], [], []                                               # To store the PSNR values

    return f4, f5, f6, _dl_c_n, _dl_n_d, _dl_c_d


def PSNR(i1, i2):
    D = i1 - i2
    MSE = np.mean(D**2)
    psnr = 20*np.log10(float(255.**2)/MSE)
    return psnr


def compute_psnr(I1, I2, mask, txt_file, i, storing_list):
    psnr_c_n = PSNR(I1[mask], I2[mask])      
    txt_file.write(f'{i}: {psnr_c_n}')
    txt_file.write(f'\n')
    storing_list.append(psnr_c_n)


def psnr_mean(sig, _bm3d_c_n, _bm3d_n_d, _bm3d_c_d, _dl_c_n, _dl_n_d, _dl_c_d):
    f = open(os.path.join(root_path, f'sigma_{sig}', f'Mean_PSNR_results.txt', 'w'))
    f.write(f'BM3D avg PSNR between clean and noisy: {np.mean(np.array(_bm3d_c_n))}')
    f.write(f'\n')
    f.write(f'BM3D avg PSNR between noisy and denoised: {np.mean(np.array(_bm3d_n_d))}')
    f.write(f'\n')
    f.write(f'BM3D avg PSNR between clean and denoised: {np.mean(np.array(_bm3d_c_d))}')
    f.write(f'\n')
    f.write(f'DL avg PSNR between clean and noisy: {np.mean(np.array(_dl_c_n))}')
    f.write(f'\n')
    f.write(f'DL avg PSNR between noisy and denoised: {np.mean(np.array(_dl_n_d))}')
    f.write(f'\n')
    f.write(f'DL avg PSNR between clean and denoised: {np.mean(np.array(_dl_c_d))}')
    f.write(f'\n')
    f.close()
