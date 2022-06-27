import pandas as pd
import os


#-------------------------Note------------------------------------------------
# file_name = as given in d_gen method of path_file.py ('only_gau' or 'ph_gau')
#-------------------------Note------------------------------------------------


for file_name in ['only_gau', 'ph_gau']:
    train_df = pd.read_feather(os.path.join('Data', f'train_{file_name}.ftr'))
    train_df_0 = train_df[:int(len(train_df)/2)]
    train_df_1 = train_df[int(len(train_df)/2):]
    train_df_0 = train_df_0.reset_index(drop=True)     
    train_df_1 = train_df_1.reset_index(drop=True)    
    train_df_0.to_feather(os.path.join('Data', f'train_{file_name}_0.ftr'))
    train_df_1.to_feather(os.path.join('Data', f'train_{file_name}_1.ftr'))


    val_df = pd.read_feather(os.path.join('Data', f'val_{file_name}.ftr'))
    val_df_0 = val_df[:int(len(val_df)/2)]
    val_df_1 = val_df[int(len(val_df)/2):]
    val_df_0 = val_df_0.reset_index(drop=True)     
    val_df_1 = val_df_1.reset_index(drop=True)    
    val_df_0.to_feather(os.path.join('Data', f'val_{file_name}_0.ftr'))
    val_df_1.to_feather(os.path.join('Data', f'val_{file_name}_1.ftr'))
