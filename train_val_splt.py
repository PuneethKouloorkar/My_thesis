import pickle
import os
from sklearn.model_selection import KFold


data_fol = 'Data'
intf_list = pickle.load(open(os.path.join(data_fol, 'intf_imgs.pkl'), "rb"))

for cf in range(1, 4):
    if cf == 1:
        train_x = intf_list[:int(0.90*len(intf_list))]      # 6372  
        val_x = intf_list[int(0.90*len(intf_list)):]        # 708
        print(len(train_x), len(val_x))
    elif cf == 2:
        train_x = intf_list[int(0.10*len(intf_list)):]      # 6372  
        val_x = intf_list[:int(0.10*len(intf_list))]        # 708
        print(len(train_x), len(val_x))
    elif cf == 3:
        train_x = intf_list[:int((len(intf_list) - int(0.10*len(intf_list)))/2)]      # 3186
        train_x += intf_list[len(train_x) + int(0.10*len(intf_list)):]                # 3186
        val_x = intf_list[int((len(intf_list) - int(0.10*len(intf_list)))/2):int((len(intf_list) - int(0.10*len(intf_list)))/2)+int(0.10*len(intf_list))]        # 708
        print(len(train_x), len(val_x))
    else:
        print('Cross-folds greater than 3!')
