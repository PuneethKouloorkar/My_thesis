import numpy as np
import matplotlib.image as img
import math
import os
import torch
import pickle
import xlwt
from pytorch_msssim import ssim
from testing_utils import test_list
import pandas as pd

# Phase image computation-------------------------------------------------------------------------------
ComPhaseP = {}
ComPhaseTildeP = {}
ComPhaseXP = {}

def com_phase_p(wd, folders, intfs):
    intfs_float = []

    for file in intfs:
        load_image = img.imread(file)
        intfs_float.append(load_image.astype('float64'))

    for idx in range(0, len(intfs_float), 5):
        upper = 2 * (intfs_float[idx+1] - intfs_float[idx+3])
        lower = 2 * intfs_float[idx+2] - intfs_float[idx+4] - intfs_float[idx]
        
        phi = np.arctan2(upper.astype('float64'), lower.astype('float64')) + np.pi   # [0 to 2pi]
        phi[upper==0] = 0
        phi[lower==0] = 0
        phi = phi/(2*np.pi)    # [0 to 1]
        
        phi_ten = torch.Tensor(phi).unsqueeze(0)
        ComPhaseP[f'{wd}/{folders}/p_[0-1]/p_[0-1]_{idx//5}'] = phi_ten
        
def com_phase_tilde_p(wd, folders, inp, out):
    fx_path = out
    xp_path = inp

    for idx in range(0, len(fx_path), 5):
        I1_fx = fx_path[idx]      # [0 to 1]
        I2_fx = fx_path[idx+1]    # [0 to 1]
        I3_fx = fx_path[idx+2]    # [0 to 1]
        I4_fx = fx_path[idx+3]    # [0 to 1]
        I5_fx = fx_path[idx+4]    # [0 to 1]

        upper_fx = 2 * (I2_fx - I4_fx)
        lower_fx = 2 * I3_fx - I5_fx - I1_fx

        I1_xp = xp_path[idx]      # [0 to 1]
        I2_xp = xp_path[idx+1]    # [0 to 1]
        I3_xp = xp_path[idx+2]    # [0 to 1]
        I4_xp = xp_path[idx+3]    # [0 to 1]
        I5_xp = xp_path[idx+4]    # [0 to 1]

        upper_xp = 2 * (I2_xp - I4_xp)
        lower_xp = 2 * I3_xp - I5_xp - I1_xp

        phi_fx = torch.atan2(upper_fx, lower_fx) + math.pi
        phi_fx[upper_fx==0] = 0
        phi_fx[lower_fx==0] = 0
        phi_fx = phi_fx/(2*math.pi) 

        phi_xp = torch.atan2(upper_xp, lower_xp) + math.pi
        phi_xp[upper_xp==0] = 0
        phi_xp[lower_xp==0] = 0
        phi_xp = phi_xp/(2*math.pi)

        ComPhaseTildeP[f'{wd}/{folders}/tilde_p_[0-1]/tilde_p_[0-1]_{idx//5}'] = phi_fx
        ComPhaseXP[f'{wd}/{folders}/x_p_[0-1]/x_p_[0-1]_{idx//5}'] = phi_xp


# SSIM metric------------------------------------------------------------------------------------------
def SSIM(parent_folder, child_folder):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'

    snc = open(f'{wd}/ssim_noisy_clean.txt', 'w')
    scd = open(f'{wd}/ssim_clean_denoised.txt', 'w')

    s_nc, s_cd = torch.zeros(len(test_list),), torch.zeros(len(test_list),)
    
    clean = pickle.load(open(f'{wd}/TestIntfNorm.pkl', 'rb'))
    noisy = pickle.load(open(f'{wd}/TestSpFinalNorm.pkl', 'rb'))
    denoised = pickle.load(open(f'{wd}/TestOpNorm.pkl', 'rb'))

    for idx in range(len(clean)):
        ssim_n_c = ssim(noisy[idx], clean[idx], data_range=1, size_average=True)
        ssim_c_d = ssim(clean[idx], denoised[idx], data_range=1, size_average=True)

        s_nc[idx] = ssim_n_c
        s_cd[idx] = ssim_c_d

        snc.write(f'{idx}: {ssim_n_c}')
        snc.write(f'\n')
        scd.write(f'{idx}: {ssim_c_d}')
        scd.write(f'\n')
    snc.close()
    scd.close()

    snc = open(f'{wd}/ssim_noisy_clean.txt', 'a')
    scd = open(f'{wd}/ssim_clean_denoised.txt', 'a')
    
    snc.write(f'Mean: {torch.mean(s_nc)}')
    scd.write(f'Mean: {torch.mean(s_cd)}')
    snc.close()
    scd.close()

def SSIM_phase(parent_folder, child_folder):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'

    spnc = open(f'{wd}/ssim_phase_noisy_clean.txt', 'w')
    spcd = open(f'{wd}/ssim_phase_clean_denoised.txt', 'w')

    s_p_nc, s_p_cd = torch.zeros(len(test_list),), torch.zeros(len(test_list),)

    for k,v in ComPhaseP.items():
        idx = (k.split("/")[-1]).split("_")[-1]
        idx = int(idx)
        
        clean = ComPhaseP[f'{wd}/p_[0-1]/p_[0-1]_{idx}']  
        denoised = ComPhaseTildeP[f'{wd}/tilde_p_[0-1]/tilde_p_[0-1]_{idx}']   
        noisy = ComPhaseXP[f'{wd}/x_p_[0-1]/x_p_[0-1]_{idx}'] 

        noisy = noisy.reshape(1, 1, noisy.shape[-2], noisy.shape[-1])
        clean = clean.reshape(1, 1, clean.shape[-2], clean.shape[-1])
        denoised = denoised.reshape(1, 1, denoised.shape[-2], denoised.shape[-1])

        ssim_p_n_c = ssim(noisy, clean, data_range=1, size_average=True)
        ssim_p_c_d = ssim(clean, denoised, data_range=1, size_average=True)

        s_p_nc[idx] = ssim_p_n_c
        s_p_cd[idx] = ssim_p_c_d

        spnc.write(f'{idx}: {s_p_nc}')
        spnc.write(f'\n')
        spcd.write(f'{idx}: {s_p_cd}')
        spcd.write(f'\n')
    spnc.close()
    spcd.close()   

    spnc = open(f'{wd}/ssim_phase_noisy_clean.txt', 'a')
    spcd = open(f'{wd}/ssim_phase_clean_denoised.txt', 'a')
    
    spnc.write(f'Mean: {torch.mean(s_p_nc)}')
    spcd.write(f'Mean: {torch.mean(s_p_cd)}')
    spnc.close()
    spcd.close()     


# Difference image computation--------------------------------------------------------------------------
DiffOrgOp = {}
DiffSpOrg = {}
DiffXpP = {}
DiffPtP = {}

def diff(parent_folder, child_folder, out, inp, lab):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'

    clean = lab
    noisy = inp
    denoised = out

    for idx in range(len(clean)):
        org = clean[idx]
        out = denoised[idx]
        sp = noisy[idx]

        diff_org_op = torch.abs(org - out)   # Reconstruction error (noise in output)
        diff_sp_org = torch.abs(sp - org)    # Noise

        DiffOrgOp[f'{wd}/diff_org_op/diff_org_op_{idx}'] = diff_org_op
        DiffSpOrg[f'{wd}/diff_sp_org/diff_sp_org_{idx}'] = diff_sp_org

def diff_phase(parent_folder, child_folder):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'

    for k,v in ComPhaseP.items():
        idx = (k.split("/")[-1]).split("_")[-1]
        idx = int(idx)
        
        p = ComPhaseP[f'{wd}/p_[0-1]/p_[0-1]_{idx}']  
        t_p = ComPhaseTildeP[f'{wd}/tilde_p_[0-1]/tilde_p_[0-1]_{idx}']   
        x_p = ComPhaseXP[f'{wd}/x_p_[0-1]/x_p_[0-1]_{idx}']   

        d_in = torch.abs(x_p - p)    # Input noise
        d_out = torch.abs(p - t_p)   # Reconstruction error (noise in output)
        
        DiffXpP[f'{wd}/diff_xp_p/diff_xp_p_{idx}'] = d_in
        DiffPtP[f'{wd}/diff_p_tp/diff_p_tp_{idx}'] = d_out


# RMSD metric------------------------------------------------------------------------------------------
def RMSD(parent_folder, child_folder):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'
  
    foo = open(f'{wd}/rmsd_org_op.txt', 'w')
    fso = open(f'{wd}/rmsd_sp_org.txt', 'w')

    foo_psnr = open(f'{wd}/psnr_org_op.txt', 'w')
    fso_psnr = open(f'{wd}/psnr_sp_org.txt', 'w')


    r_oo, r_so = torch.zeros(len(test_list),), torch.zeros(len(test_list),)
    p_oo, p_so = torch.zeros(len(test_list),), torch.zeros(len(test_list),)
      
    for k,v in DiffOrgOp.items():
        idx = (k.split("/")[-1]).split("_")[-1]
        idx = int(idx)

        oo =  DiffOrgOp[f'{wd}/diff_org_op/diff_org_op_{idx}']     
        so =  DiffSpOrg[f'{wd}/diff_sp_org/diff_sp_org_{idx}']           
        
        patch = (so != 0)

        roo_mse = torch.mean(torch.square(oo[patch]))
        rso_mse = torch.mean(torch.square(so[patch]))
        
        # RMSD-----------------------------------------------
        roo = torch.sqrt(roo_mse)
        rso = torch.sqrt(rso_mse)
        poo = 20 * torch.log10(1.0/roo)
        pso = 20 * torch.log10(1.0/rso)
        r_oo[idx] = roo
        r_so[idx] = rso
        p_oo[idx] = poo
        p_so[idx] = pso

        foo.write(f'{idx}: {roo.item()}')
        foo.write(f'\n')
        fso.write(f'{idx}: {rso.item()}')
        fso.write(f'\n')
        foo_psnr.write(f'{idx}: {poo.item()}')
        foo_psnr.write(f'\n')
        fso_psnr.write(f'{idx}: {pso.item()}')
        fso_psnr.write(f'\n')
        # RMSD-----------------------------------------------

    foo.close()
    fso.close()
    foo_psnr.close()
    fso_psnr.close()

    # RMSD-----------------------------------------------
    foo = open(f'{wd}/rmsd_org_op.txt', 'a')
    fso = open(f'{wd}/rmsd_sp_org.txt', 'a')
    foo_psnr = open(f'{wd}/psnr_org_op.txt', 'a')
    fso_psnr = open(f'{wd}/psnr_sp_org.txt', 'a')
    
    foo.write(f'Mean: {torch.mean(r_oo)}')
    fso.write(f'Mean: {torch.mean(r_so)}')
    foo.close()
    fso.close()
    foo_psnr.write(f'Mean: {torch.mean(p_oo)}')
    fso_psnr.write(f'Mean: {torch.mean(p_so)}')
    foo_psnr.close()
    fso_psnr.close()
    # RMSD-----------------------------------------------
    
def RMSD_phase(parent_folder, child_folder):
    #print(f'{parent_folder}: {child_folder}')
    wd = f'{parent_folder}/{child_folder}'
  
    foo = open(f'{wd}/rmsd_p_tp.txt', 'w')
    fso = open(f'{wd}/rmsd_xp_p.txt', 'w') 
    foo_psnr = open(f'{wd}/psnr_p_tp.txt', 'w')
    fso_psnr = open(f'{wd}/psnr_xp_p.txt', 'w')

    r_oo, r_so = torch.zeros(len(test_list),), torch.zeros(len(test_list),)
    p_oo, p_so = torch.zeros(len(test_list),), torch.zeros(len(test_list),)
      
    for k,v in DiffPtP.items():
        idx = (k.split("/")[-1]).split("_")[-1]
        idx = int(idx)

        oo = DiffPtP[f'{wd}/diff_p_tp/diff_p_tp_{idx}']        
        so = DiffXpP[f'{wd}/diff_xp_p/diff_xp_p_{idx}']         
        
        patch = (so != 0)

        roo_mse = torch.mean(torch.square(oo[patch]))
        rso_mse = torch.mean(torch.square(so[patch]))
        
        # RMSD-----------------------------------------------
        roo = torch.sqrt(roo_mse)
        rso = torch.sqrt(rso_mse)
        poo = 20 * torch.log10(1.0/roo)
        pso = 20 * torch.log10(1.0/rso)
        r_oo[idx] = roo
        r_so[idx] = rso
        p_oo[idx] = poo
        p_so[idx] = pso

        foo.write(f'{idx}: {roo.item()}')
        foo.write(f'\n')
        fso.write(f'{idx}: {rso.item()}')
        fso.write(f'\n')
        foo_psnr.write(f'{idx}: {poo.item()}')
        foo_psnr.write(f'\n')
        fso_psnr.write(f'{idx}: {pso.item()}')
        fso_psnr.write(f'\n')
        # RMSD-----------------------------------------------

    foo.close()
    fso.close()
    foo_psnr.close()
    fso_psnr.close()

    # RMSD-----------------------------------------------
    foo = open(f'{wd}/rmsd_p_tp.txt', 'a')
    fso = open(f'{wd}/rmsd_xp_p.txt', 'a')
    foo_psnr = open(f'{wd}/psnr_p_tp.txt', 'a')
    fso_psnr = open(f'{wd}/psnr_xp_p.txt', 'a')
    
    foo.write(f'Mean: {torch.mean(r_oo)}')
    fso.write(f'Mean: {torch.mean(r_so)}')
    foo.close()
    fso.close()
    foo_psnr.write(f'Mean: {torch.mean(p_oo)}')
    fso_psnr.write(f'Mean: {torch.mean(p_so)}')
    foo_psnr.close()
    fso_psnr.close()
    # RMSD-----------------------------------------------


# RMSD ratio computation-----------------------------------------------------------------------
def RMSD_ratio(fol, name):    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    
    cols = []
    for i in os.listdir(fol):
        if i.startswith('ph_gau') or i.startswith('only_ph') or i.startswith('only_gau'):
            cols.append(i)
    cols.sort()

    for idx, col in enumerate(cols):
        # RMSD-----------------------------------------------------------------------------
        a_file = open(os.path.join(fol, col, 'rmsd_org_op.txt'), "r")
        lines = a_file.readlines()
        last_lines = lines[-1].split(":")[-1]

        b_file = open(os.path.join(fol, col, 'rmsd_sp_org.txt'), "r")
        lines_ = b_file.readlines()
        last_lines_ = lines_[-1].split(":")[-1]

        ratio = round(float(last_lines),3)/round(float(last_lines_),3)
        #print(f"{i}: io={round(float(last_lines),3)} | si={round(float(last_lines_),3)} | r={round(ratio,3)}")

        # PSNR-----------------------------------------------------------------------------
        a_file_psnr = open(os.path.join(fol, col, 'psnr_org_op.txt'), "r")
        lines_psnr = a_file_psnr.readlines()
        last_lines_psnr = lines_psnr[-1].split(":")[-1]

        b_file_psnr = open(os.path.join(fol, col, 'psnr_sp_org.txt'), "r")
        lines_psnr_ = b_file_psnr.readlines()
        last_lines_psnr_ = lines_psnr_[-1].split(":")[-1]

        ratio_psnr = round(float(last_lines_psnr),3)/round(float(last_lines_psnr_),3)
        #print(f"{i}: io={round(float(last_lines_psnr),3)} | si={round(float(last_lines_psnr_),3)} | r={round(ratio_psnr,3)}")
        #print()

        # Write to the spreadsheet----------------------------------------------------------
        sheet1.write(0, 3*idx, i)
        sheet1.write(1, 3*idx, "intf_rmse(y,tilde_y)")
        sheet1.write(1, (3*idx)+1, "intf_rmse(x,y)")
        sheet1.write(1, (3*idx)+2, "ratio")
        sheet1.write(4, 3*idx, "intf_psnr(y,tilde_y)")
        sheet1.write(4, (3*idx)+1, "intf_psnr(x,y)")
        sheet1.write(4, (3*idx)+2, "ratio")

        sheet1.write(2, 3*idx, round(float(last_lines),3))
        sheet1.write(2, (3*idx)+1, round(float(last_lines_),3))
        sheet1.write(2, (3*idx)+2, round(ratio,3))
        sheet1.write(5, 3*idx, round(float(last_lines_psnr),3))
        sheet1.write(5, (3*idx)+1, round(float(last_lines_psnr_),3))
        sheet1.write(5, (3*idx)+2, round(ratio_psnr,3))
    book.save(f"{fol}/{name}.xls")

def RMSD_ratio_phase(fol, name):    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    
    cols = []
    for i in os.listdir(fol):
        if i.startswith('ph_gau') or i.startswith('only_ph') or i.startswith('only_gau'):
            cols.append(i)
    cols.sort()

    for idx, col in enumerate(cols):
        # RMSD-----------------------------------------------------------------------------
        a_file = open(os.path.join(fol, col, 'rmsd_p_tp.txt'), "r")
        lines = a_file.readlines()
        last_lines = lines[-1].split(":")[-1]

        b_file = open(os.path.join(fol, col, 'rmsd_xp_p.txt'), "r")
        lines_ = b_file.readlines()
        last_lines_ = lines_[-1].split(":")[-1]

        ratio = round(float(last_lines),3)/round(float(last_lines_),3)
        #print(f"{i}: io={round(float(last_lines),3)} | si={round(float(last_lines_),3)} | r={round(ratio,3)}")

        # PSNR-----------------------------------------------------------------------------
        a_file_psnr = open(os.path.join(fol, col, 'psnr_p_tp.txt'), "r")
        lines_psnr = a_file_psnr.readlines()
        last_lines_psnr = lines_psnr[-1].split(":")[-1]

        b_file_psnr = open(os.path.join(fol, col, 'psnr_xp_p.txt'), "r")
        lines_psnr_ = b_file_psnr.readlines()
        last_lines_psnr_ = lines_psnr_[-1].split(":")[-1]

        ratio_psnr = round(float(last_lines_psnr),3)/round(float(last_lines_psnr_),3)
        #print(f"{i}: io={round(float(last_lines_psnr),3)} | si={round(float(last_lines_psnr_),3)} | r={round(ratio_psnr,3)}")
        #print()

        # Write to the spreadsheet----------------------------------------------------------
        sheet1.write(0, 3*idx, i)
        sheet1.write(1, 3*idx, "phase_rmse(y,tilde_y)")
        sheet1.write(1, (3*idx)+1, "phase_rmse(x,y)")
        sheet1.write(1, (3*idx)+2, "ratio")
        sheet1.write(4, 3*idx, "phase_psnr(y,tilde_y)")
        sheet1.write(4, (3*idx)+1, "phase_psnr(x,y)")
        sheet1.write(4, (3*idx)+2, "ratio")

        sheet1.write(2, 3*idx, round(float(last_lines),3))
        sheet1.write(2, (3*idx)+1, round(float(last_lines_),3))
        sheet1.write(2, (3*idx)+2, round(ratio,3))
        sheet1.write(5, 3*idx, round(float(last_lines_psnr),3))
        sheet1.write(5, (3*idx)+1, round(float(last_lines_psnr_),3))
        sheet1.write(5, (3*idx)+2, round(ratio_psnr,3))
    book.save(f"{fol}/{name}.xls")


# SSIM ratio computation-----------------------------------------------------------------------
def SSIM_ratio(fol, name):    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")

    sheet1.write(0, 0, "only_gau")
    sheet1.write(1, 0, "s(y,tilde_y)")
    sheet1.write(1, 1, "s(x,y)")
    sheet1.write(1, 2, "ratio")

    sheet1.write(0, 3, "only_ph")
    sheet1.write(1, 3, "s(y,tilde_y)")
    sheet1.write(1, 4, "s(x,y)")
    sheet1.write(1, 5, "ratio")

    sheet1.write(0, 6, "ph+gau*0.25")
    sheet1.write(1, 6, "s(y,tilde_y)")
    sheet1.write(1, 7, "s(x,y)")
    sheet1.write(1, 8, "ratio")

    sheet1.write(0, 9, "ph+gau*0.5")
    sheet1.write(1, 9, "s(y,tilde_y)")
    sheet1.write(1, 10, "s(x,y)")
    sheet1.write(1, 11, "ratio")

    sheet1.write(0, 12, "ph+gau*0.75")
    sheet1.write(1, 12, "s(y,tilde_y)")
    sheet1.write(1, 13, "s(x,y)")
    sheet1.write(1, 14, "ratio")

    sheet1.write(0, 15, "ph+gau*5")
    sheet1.write(1, 15, "s(y,tilde_y)")
    sheet1.write(1, 16, "s(x,y)")
    sheet1.write(1, 17, "ratio")

    sheet1.write(0, 18, "ph+gau*10")
    sheet1.write(1, 18, "s(y,tilde_y)")
    sheet1.write(1, 19, "s(x,y)")
    sheet1.write(1, 20, "ratio")

    sheet1.write(0, 21, "BM3D_only_ph")
    sheet1.write(1, 21, "s(y,tilde_y)")
    sheet1.write(1, 22, "s(x,y)")
    sheet1.write(1, 23, "ratio")

    sheet1.write(0, 24, "BM3D_ph+beta=0.5")
    sheet1.write(1, 24, "s(y,tilde_y)")
    sheet1.write(1, 25, "s(x,y)")
    sheet1.write(1, 26, "ratio")

    #print(' ==> Computing RMSD ratio')
    for i in os.listdir(fol):
        if i.startswith('ph') or i.startswith('on'):# or i.startswith('8_bit'):
            a_file = open(f"{fol}/{i}/ssim_clean_denoised.txt", "r")
            lines = a_file.readlines()
            last_lines = lines[-1].split(":")[-1]

            b_file = open(f"{fol}/{i}/ssim_noisy_clean.txt", "r")
            lines_ = b_file.readlines()
            last_lines_ = lines_[-1].split(":")[-1]

            ratio = round(float(last_lines),3)/round(float(last_lines_),3)
            #print(f"{i}: io={round(float(last_lines),3)} | si={round(float(last_lines_),3)} | r={round(ratio,3)}")

            if i=='only_gau':
                sheet1.write(2, 0, round(float(last_lines),3))
                sheet1.write(2, 1, round(float(last_lines_),3))
                sheet1.write(2, 2, round(ratio,3))
            if i=='only_ph':
                sheet1.write(2, 3, round(float(last_lines),3))
                sheet1.write(2, 4, round(float(last_lines_),3))
                sheet1.write(2, 5, round(ratio,3))
            if i=='ph+beta=0.25':
                sheet1.write(2, 6, round(float(last_lines),3))
                sheet1.write(2, 7, round(float(last_lines_),3))
                sheet1.write(2, 8, round(ratio,3))
            if i=='ph+beta=0.5':
                sheet1.write(2, 9, round(float(last_lines),3))
                sheet1.write(2, 10, round(float(last_lines_),3))
                sheet1.write(2, 11, round(ratio,3))
            if i=='ph+beta=0.75':
                sheet1.write(2, 12, round(float(last_lines),3))
                sheet1.write(2, 13, round(float(last_lines_),3))
                sheet1.write(2, 14, round(ratio,3))
            if i=='ph+beta=5':
                sheet1.write(2, 15, round(float(last_lines),3))
                sheet1.write(2, 16, round(float(last_lines_),3))
                sheet1.write(2, 17, round(ratio,3))
            if i=='ph+beta=10':
                sheet1.write(2, 18, round(float(last_lines),3))
                sheet1.write(2, 19, round(float(last_lines_),3))
                sheet1.write(2, 20, round(ratio,3))  
            if i=='8_bit_only_ph':
                sheet1.write(2, 21, round(float(last_lines),3))
                sheet1.write(2, 22, round(float(last_lines_),3))
                sheet1.write(2, 23, round(ratio,3))  
            if i=='8_bit_ph+beta=0.5':
                sheet1.write(2, 24, round(float(last_lines),3))
                sheet1.write(2, 25, round(float(last_lines_),3))
                sheet1.write(2, 26, round(ratio,3))            

    book.save(f"{fol}/{name}.xls")

def SSIM_ratio_phase(fol, name):    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")

    sheet1.write(0, 0, "only_gau")
    sheet1.write(1, 0, "phi_s(p,tilde_p)")
    sheet1.write(1, 1, "phi_s(x_p,p)")
    sheet1.write(1, 2, "ratio")

    sheet1.write(0, 3, "only_ph")
    sheet1.write(1, 3, "phi_s(p,tilde_p)")
    sheet1.write(1, 4, "phi_s(x_p,p)")
    sheet1.write(1, 5, "ratio")

    sheet1.write(0, 6, "ph+gau*0.25")
    sheet1.write(1, 6, "phi_s(p,tilde_p)")
    sheet1.write(1, 7, "phi_s(x_p,p)")
    sheet1.write(1, 8, "ratio")

    sheet1.write(0, 9, "ph+gau*0.5")
    sheet1.write(1, 9, "phi_s(p,tilde_p)")
    sheet1.write(1, 10, "phi_s(x_p,p)")
    sheet1.write(1, 11, "ratio")

    sheet1.write(0, 12, "ph+gau*0.75")
    sheet1.write(1, 12, "phi_s(p,tilde_p)")
    sheet1.write(1, 13, "phi_s(x_p,p)")
    sheet1.write(1, 14, "ratio")

    sheet1.write(0, 15, "ph+gau*5")
    sheet1.write(1, 15, "phi_s(y,tilde_y)")
    sheet1.write(1, 16, "phi_s(x,y)")
    sheet1.write(1, 17, "ratio")

    sheet1.write(0, 18, "ph+gau*10")
    sheet1.write(1, 18, "phi_s(y,tilde_y)")
    sheet1.write(1, 19, "phi_s(x,y)")
    sheet1.write(1, 20, "ratio")

    sheet1.write(0, 21, "BM3D_only_ph")
    sheet1.write(1, 21, "phi_s(y,tilde_y)")
    sheet1.write(1, 22, "phi_s(x,y)")
    sheet1.write(1, 23, "ratio")

    sheet1.write(0, 24, "BM3D_ph+beta=0.5")
    sheet1.write(1, 24, "phi_s(y,tilde_y)")
    sheet1.write(1, 25, "phi_s(x,y)")
    sheet1.write(1, 26, "ratio")

    #print(' ==> Computing RMSD ratio phase')
    for i in os.listdir(fol):
        if i.startswith('ph') or i.startswith('on'):# or i.startswith('8_bit'):
            a_file = open(f"{fol}/{i}/ssim_phase_clean_denoised.txt", "r")
            lines = a_file.readlines()
            last_lines = lines[-1].split(":")[-1]

            b_file = open(f"{fol}/{i}/ssim_phase_noisy_clean.txt", "r")
            lines_ = b_file.readlines()
            last_lines_ = lines_[-1].split(":")[-1]

            ratio = round(float(last_lines),3)/round(float(last_lines_),3)
            #print(f"{i}: io={round(float(last_lines),3)} | si={round(float(last_lines_),3)} | r={round(ratio,3)}")

            if i=='only_gau':
                sheet1.write(2, 0, round(float(last_lines),3))
                sheet1.write(2, 1, round(float(last_lines_),3))
                sheet1.write(2, 2, round(ratio,3))
            if i=='only_ph':
                sheet1.write(2, 3, round(float(last_lines),3))
                sheet1.write(2, 4, round(float(last_lines_),3))
                sheet1.write(2, 5, round(ratio,3))
            if i=='ph+beta=0.25':
                sheet1.write(2, 6, round(float(last_lines),3))
                sheet1.write(2, 7, round(float(last_lines_),3))
                sheet1.write(2, 8, round(ratio,3))
            if i=='ph+beta=0.5':
                sheet1.write(2, 9, round(float(last_lines),3))
                sheet1.write(2, 10, round(float(last_lines_),3))
                sheet1.write(2, 11, round(ratio,3))
            if i=='ph+beta=0.75':
                sheet1.write(2, 12, round(float(last_lines),3))
                sheet1.write(2, 13, round(float(last_lines_),3))
                sheet1.write(2, 14, round(ratio,3))
            if i=='ph+beta=5':
                sheet1.write(2, 15, round(float(last_lines),3))
                sheet1.write(2, 16, round(float(last_lines_),3))
                sheet1.write(2, 17, round(ratio,3))
            if i=='ph+beta=10':
                sheet1.write(2, 18, round(float(last_lines),3))
                sheet1.write(2, 19, round(float(last_lines_),3))
                sheet1.write(2, 20, round(ratio,3))  
            if i=='8_bit_only_ph':
                sheet1.write(2, 21, round(float(last_lines),3))
                sheet1.write(2, 22, round(float(last_lines_),3))
                sheet1.write(2, 23, round(ratio,3))  
            if i=='8_bit_ph+beta=0.5':
                sheet1.write(2, 24, round(float(last_lines),3))
                sheet1.write(2, 25, round(float(last_lines_),3))
                sheet1.write(2, 26, round(ratio,3)) 
    
    book.save(f"{fol}/{name}.xls")
