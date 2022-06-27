import torch
from torch.utils.data import DataLoader
import os
import pickle
from torchvision import transforms


# Import the architecture, all the methods from utils.py-------------------------
from DnCNN_plus import DnCNN_plus
from lr_finder_utils import hyperparameters, training_vcs_hyperparams, paths, Variable_crop_dataset
from lr_finder_utils import training_vcs, root_path, plots, rate_of_change


# Load the GPU device-------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


# Load all the hyperparameters-----------------------------------------------------
batch, epochs, start_lr, crop_size, loss_fn, _,_,_,_ = hyperparameters()
_,_,_,_,_, num_processes, lr_decay_mul, lr_decay_epoch_count, l2_penalty = hyperparameters()
min_phantoms, max_phantoms, min_alpha, max_alpha = training_vcs_hyperparams()


# Load all the required paths------------------------------------------------------
data_fol, data_type = paths()


# Load the dataset and dataloaders-------------------------------------------------
intf_list = pickle.load(open(os.path.join(data_fol, 'intf_imgs.pkl'), "rb"))
train_x = intf_list[:int(0.90*len(intf_list))]     
val_x = intf_list[int(0.90*len(intf_list)):]    


phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(os.path.join(data_fol, f'm{mask}{shift}.pkl'), "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data


img_t = transforms.Compose([transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.type(torch.FloatTensor)),])


if data_type == 'only_gau':
    phantom = False
else:
    phantom = True

# Create only the train dataloader-----------------------------------------------
training_set = Variable_crop_dataset(train_x, img_t, phantom,
                                    phantom_dict, min_phantoms,
                                    max_phantoms, min_alpha, max_alpha)
trainloader = DataLoader(training_set, batch_size=batch, shuffle=True, num_workers=num_processes)
print(f'Trainloader size: {len(trainloader)}')


# Construct the model object and load it into the device--------------------------
model = DnCNN_plus(1, 1, [8,32,128,512]).to(device)


# Construct the optimizer object---------------------------------------------------
optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)


t, v = [], []
t_r, v_r = [], []
lr = []


# Training--------------------------------------------------------------------------
for epoch in range(1, epochs+1):
    train_loss, t_ratio, lr = training_vcs(trainloader, model, device, loss_fn, crop_size, optim, lr)


# Save the tensor files-------------------------------------------------------------
torch.save(torch.Tensor(train_loss), os.path.join(root_path, '_loss.pt'))
torch.save(torch.Tensor(lr), os.path.join(root_path, '_lr.pt'))


# Plot learning rate vs iterations, training loss vs learning rate------------------ 
plots(lr, batch, crop_size, train_loss)


# Plot ROC of training loss vs learning rate, simple moving average of ROC of training
# loss vs learning rate---------------------------------------------------------------
rate_of_change(lr, train_loss, batch, crop_size)

