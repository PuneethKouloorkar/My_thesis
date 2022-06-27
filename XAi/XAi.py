import cv2
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
from tqdm import tqdm
import os

from DnCNN_plus import DnCNN_plus
from xai_utils import paths, XAi_dataset, hyperparams, create_dirs, root_path
from xai_utils import sensitivity_analysis, Guided_backprop, gbp_heatmaps


#-----------------------------------Note------------------------------------------#
# Provide the saved model path to 'model_path' variable in 'paths()' method in xai_utils.py
#-----------------------------------Note------------------------------------------#


# Use CPU to input the entire image into the model----------------------------------
device = torch.device("cpu")


# Load the interferograms------------------------------------------------------------
test_intfs, data_fol, model_path = paths()


# Load all the hyperparameters-------------------------------------------------------
num_phantoms, min_alpha, max_alpha, batch = hyperparams()


# Construct the model object and load it into the device-----------------------------
model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)


# Load the trained model-------------------------------------------------------------
model.load_state_dict(torch.load(model_path, map_location=device))


img_t = transforms.Compose([transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor)),])
                            

phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(os.path.join(data_fol, f'm{mask}{shift}.pkl'), "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data


# Construct the dataloader object------------------------------------------------------
test_set = XAi_dataset(test_intfs, img_t, True, phantom_dict, num_phantoms, min_alpha, max_alpha)                        
testloader = DataLoader(test_set, batch_size=batch, shuffle=False)   


# Create all the required directories in the root path---------------------------------
create_dirs()


# Create heatmaps for all the examples------------------------------------------------- 
for i, data in tqdm(enumerate(testloader)):
    inp, label  = data
    inp, label = inp.to(device), label.to(device)
    inp.requires_grad = True

    mask = (label == 0)     
    
    save_image(inp, os.path.join(root_path, 'model_input', f'inp_{i}.png'))
    save_image(label, os.path.join(root_path, 'label', f'lab_{i}.png'))
    
    op = model(inp)
    op.sum().backward()
    
    # Sensitivity Analysis---------------------------------------------------------------
    sa_grey_hm, sa_col_hm = sensitivity_analysis(inp)
    save_image(sa_grey_hm, os.path.join(root_path, 'sa_greyscale_heatmap', f'sa_bw_{i}.png'))
    cv2.imwrite(os.path.join(root_path, 'sa_colored_heatmap', f'sa_c_{i}.png'))
    

    # Guided Backpropagation-------------------------------------------------------------
    guided_bp = Guided_backprop(model)
    result, op_unmasked, op_masked = guided_bp.visualize(inp, root_path, i, mask)
    save_image(op_unmasked, os.path.join(root_path, 'model_output_unmasked', f'op_unmasked_{i}.png'))
    save_image(op_masked, os.path.join(root_path, 'model_output_masked', f'op_masked_{i}.png'))

    gbp_grey_hm, gbp_col_hm = gbp_heatmaps(result)
    save_image(gbp_grey_hm, os.path.join(root_path, 'gbp_greyscale_heatmap', f'gbp_bw_{i}.png'))
    save_image(gbp_col_hm, os.path.join(root_path, 'gbp_colored_heatmap', f'gbp_c_{i}.png'))

    

