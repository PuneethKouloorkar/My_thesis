import os
import pandas as pd

root_path = os.path.dirname(os.path.abspath(__file__))

# num_images and mask_gen------------------------------------------------------------------------------
def num_img():
    intfs_path = 'E:\PTB\Data'    # Set the path to interferograms
    #phantom_path = '/home/kouloo01/san/Interferogram_data/Images'
    
    #if not os.path.exists(os.path.join(root_path, 'Data')):
    os.makedirs(os.path.join(root_path, 'Data'))
    out_fol = os.path.join(root_path, 'Data')                             # .pkl files directory
    infs_path_1 = os.path.join(intfs_path, 'InterferogramML4')            # InterferogramML4 directory
    infs_path_2 = os.path.join(intfs_path, 'InterferogramML5')            # InterferogramML5 directory
    emp_infs_path = os.path.join(intfs_path, '20201127_zero_images')    # 20201127_zero_images directory
    return out_fol, infs_path_1, infs_path_2, emp_infs_path

def m_gen():
    out_fol = os.path.join(root_path, 'Data')
    return out_fol
# num_images and mask_gen------------------------------------------------------------------------------

# data gen---------------------------------------------------------------------------------------------
def d_gen():
    df_train = pd.DataFrame(columns=['train_sp', 'train_org'])
    df_val = pd.DataFrame(columns=['val_sp', 'val_org'])
    df_test = pd.DataFrame(columns=['test_sp', 'test_org'])
    fol = os.path.join(root_path, 'Data')
    phantom = False                                                       # False: Only Gaussian noise, True: Gaussian + phantom noise
    
    if phantom == False:
        name = 'only_gau'                                                 # Set a name for the dataset depending on the type of noise
    else:
        name = 'ph_gau'                                                   # Set a name for the dataset depending on the type of noise
    
    out_fol1 = os.path.join(root_path, 'Data')                                                    
    masking = True                                                        # True: Background pixels are set to 0 after superposition,
                                                                          # False: Background pixels are not set to 0 after superposition
    return df_train, df_val, df_test, fol, name, out_fol1, phantom, masking
# data gen---------------------------------------------------------------------------------------------

# training-------------------------------------------------------------------------------------
def intf_training(data_type):
    fol1 = os.path.join(root_path, 'Data')
    current_archi = 'DnCNN_plus'
    
    #if not os.path.exists(os.path.join(root_path, 'only_gau')):
    os.makedirs(os.path.join(root_path, data_type))
    out_fol2 = os.path.join(root_path, data_type)
    return fol1, current_archi, out_fol2
# training-------------------------------------------------------------------------------------
