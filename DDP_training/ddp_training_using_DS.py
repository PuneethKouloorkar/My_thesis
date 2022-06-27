import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from DnCNN_plus import DnCNN_plus
from ddp_training_using_DS_utils import load_ftr_datasets, paths, gen_hyperparams
from ddp_training_using_DS_utils import For_training, training, plots


#----------------------------Note----------------------------------------------------
# Run job.sh to start training
#----------------------------Note----------------------------------------------------


# Set the type of training data------------------------------------------------------
file_name = 'only_gau'
#file_name = 'ph_gau'


# Load al the paths and hyperparameters----------------------------------------------
out_fol, name = paths(file_name, 'gloo')
batch, start_lr, l2_penalty, num_epochs, _,_,_,_ = gen_hyperparams()
_,_,_,_, loss_fn, lr_decay_count, lr_decay_mul, checkpoint_count = gen_hyperparams()


# DDP training-----------------------------------------------------------------------
def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  
    args = parser.parse_args()

    # Keep track of whether the current process is the `master` process (totally optional, but I find it useful 
    # for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    # Set the device
    args.device = torch.cuda.device(args.local_rank)

    # Initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying 
    # `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script 
    # on different machines)
    dist.init_process_group(backend='gloo', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # Load the dataset, create DataSampler (DS) object and pass it to the Dataloader----
    train_sp, train_org, val_sp, val_org = load_ftr_datasets()

    training_set = For_training(sp_image_list=train_sp, org_image_list=train_org)
    t_sampler = DistributedSampler(training_set, shuffle=False)
    trainloader = DataLoader(training_set, batch_size=batch, sampler=t_sampler, num_workers=2)
    print(f'Trainloader size: {len(trainloader)}')

    val_set = For_training(sp_image_list=val_sp, org_image_list=val_org)
    v_sampler = DistributedSampler(val_set, shuffle=False)
    valloader = DataLoader(val_set, batch_size=batch, sampler=v_sampler, num_workers=2)
    print(f'Valloader size: {len(valloader)}')
    
    # Create the model object and wrap it up in DDP method------------------------------
    model = DnCNN_plus(1, 1, [8,32,128,512]).to(torch.device(args.local_rank))
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Create the optimizer object-------------------------------------------------------
    optim = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=l2_penalty)

    # Training and validation-----------------------------------------------------------
    t, t_r, v, v_r = training(args, num_epochs, optim, ddp_model, trainloader, valloader, 
                              loss_fn, [], [], [], [], out_fol, name, lr_decay_count, 
                              lr_decay_mul, checkpoint_count)

    # Plotting training and validation loss and ratio curves----------------------------
    plots(num_epochs, t, t_r, v, v_r, out_fol, name)


# Run the file--------------------------------------------------------------------------
if __name__ == '__main__':
    main()