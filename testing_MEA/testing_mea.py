import torch
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from DnCNN_plus import DnCNN_plus
from testing_mea_utils import paths, dirs, Masking, root_path, remove_module


# Set the device-----------------------------------------------------------------------------
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the measurement data and masks--------------------------------------------------------
test_list, test_list_masks, saved_model = paths()


# Create the required directories------------------------------------------------------------
dirs()


img_t = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor))
                            ])


# Build the Dataloader------------------------------------------------------------------------
test_set = Masking(test_list, img_t, test_list_masks)
testloader = DataLoader(test_set, batch_size=1, shuffle=False)     


# Create the model object and load the saved state dict---------------------------------------
model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)
state_dict = torch.load(saved_model)


# Remove 'module.' from the parameters name if training is done on multiple GPUs--------------
model = remove_module(model, state_dict)  


# Testing-------------------------------------------------------------------------------------
with torch.no_grad():
    for i, data in tqdm(enumerate(testloader)):
        inp, mask = data
        inp, mask = inp.to(device), mask.to(device)
        
        outputs = model(inp)
        outputs[mask] = 0

        save_image(outputs, os.path.join(root_path, 'Denosied_input', f'out_{i}.png'))
