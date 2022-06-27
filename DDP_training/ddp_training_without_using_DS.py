import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from DnCNN_plus import DnCNN_plus
from ddp_training_without_using_DS_utils import loaders, paths
from ddp_training_without_using_DS_utils import setup, run_demo, gen_hyperparams
from ddp_training_without_using_DS_utils import training, plots


#----------------------------Note----------------------------------------------------
# Run split_ftr.py first once
#----------------------------Note----------------------------------------------------


# Set the type of training data------------------------------------------------------
file_name = 'only_gau'
#file_name = 'ph_gau'


# Set the environment----------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"


# Global Variables-------------------------------------------------------------------
out_fol, name = paths(file_name, 'gloo')
batch, start_lr, l2_penalty, num_epochs, loss_fn, lr_decay_count, _,_ = gen_hyperparams()
_,_,_,_,_,_, lr_decay_mul, checkpoint_count = gen_hyperparams()


# DDP training-----------------------------------------------------------------------
def main(rank, worldsize):
    setup(rank, worldsize)

    # Create the model object and wrap it up in DDP method---------------------------
    model = DnCNN_plus(1, 1, [8,32,128,512]).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    #model.load_state_dict(torch.load('', map_location=torch.device(rank)))

    # Manually load the data into GPU 0 and GPU 1------------------------------------
    if rank == 0:
        trainloader, valloader = loaders(rank, file_name, batch)
    elif rank == 1:
        trainloader, valloader = loaders(rank, file_name, batch)
    else:
        sys.exit(99)

    # Build the optimizer object-----------------------------------------------------
    optim = torch.optim.Adam(ddp_model.parameters(), lr=start_lr, weight_decay=l2_penalty)
 
    # Training and validation--------------------------------------------------------
    t, t_r, v, v_r = training(num_epochs, optim, lr_decay_count, lr_decay_mul, checkpoint_count,
                              trainloader, valloader, ddp_model, rank, loss_fn, [], [], [], [], 
                              out_fol, name)

    # Plotting training and validation loss and ratio curves-------------------------
    plots(num_epochs, t, t_r, v, v_r, out_fol, name)


# Run the file-----------------------------------------------------------------------
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(main, world_size)