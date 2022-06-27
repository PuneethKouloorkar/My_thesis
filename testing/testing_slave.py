import torch
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from testing_utils import test_list, Superposition_test, params, testing_proc, remove_modules

# Import all the architectures---------------------------------------------------------------
from DnCNN_plus import DnCNN_plus
from DnCNN_minus import DnCNN_minus
from Double_DnCNN_minus import Double_DnCNN_minus
from DnCNN_paper import DnCNN_paper
from ResNet import Block, ResNet
from UNet import UNet
from DenseNet_UNet import FCDenseNet57

# Import all the required metrics------------------------------------------------------------ 
from testing_metrics import com_phase_p, com_phase_tilde_p, diff, diff_phase 
from testing_metrics import RMSD, RMSD_phase, RMSD_ratio, RMSD_ratio_phase
from testing_metrics import ComPhaseP, ComPhaseTildeP, ComPhaseXP, DiffOrgOp
from testing_metrics import DiffSpOrg, DiffXpP, DiffPtP


from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument('-a', '--archi', type=str, default='ResNet18')
final_args = args.parse_args()


# Set the GPU device-------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load all the required paramters------------------------------------------------------------
min_alpha, max_alpha, num_phantoms, count, crop_size, test_cases = params()


# Load all th phantom paths------------------------------------------------------------------
phantom_dict = {}
for mask in range(1,5):
    for shift in range(5):
        phantom_data = pickle.load(open(os.path.join('Data', f'm{mask}{shift}.pkl'), "rb"))
        phantom_dict[f'm{mask}_i0_img{shift}.bmp'] = phantom_data


def archi_testing(archi):
    if archi == 'DnCNN_plus':
        model = DnCNN_plus(1, 1, [8,32,128,512])
    elif archi == 'DnCNN_minus':
        model = DnCNN_minus(1, 1, [8,32,128,512])
    elif archi == 'Double_DnCNN_minus':
        model = Double_DnCNN_minus(1, 1, [4,16,64,256], [8,32,128,512])
    elif archi == 'DnCNN_paper_pT':
        model = DnCNN_paper(1, 20) 
    elif archi == 'DnCNN_paper':
        model = DnCNN_paper(1, 20)
    elif archi == 'ResNet18':
        model = ResNet(Block, [2,2,2,2], 1, 1)
    elif archi == 'ResNet34':
        model = ResNet(Block, [3,6,4,3], 1, 1)
    elif archi == 'UNet':
        model = UNet(1,1)
    elif archi == 'FCDenseNet':
        model = FCDenseNet57(1)
    
    for cf in [1, 2, 3]:
        # Set the output folder path------------------------------------------------------------------
        fol = os.path.join(f'cross_fold_{cf}', archi)

        # Load the saved checkpoint-------------------------------------------------------------------
        model_path = os.path.join(fol, f'{archi}.pt')
        new_state_dict = remove_modules(model_path)
        model.load_state_dict(new_state_dict)
        model.eval()

        # Testing-------------------------------------------------------------------------------------
        for f in test_cases:
            TestIntfNorm, TestSpFinalNorm, TestOpNorm = [], [], []
            
            # Load the test case name, and beta----------------------------------------------------
            folders, beta = f
            
            # Create the output folder for respective test cases-----------------------------------
            os.makedirs(os.path.join(fol, folders))
            final_dir = os.path.join(fol, folders)
            
            if folders.startswith('8_bit'):
                os.makedirs(os.path.join(final_dir, 'clean_intf'))
                os.makedirs(os.path.join(final_dir, 'denoised_intf'))
                os.makedirs(os.path.join(final_dir, 'noisy_intf'))  
            else:
                os.makedirs(os.path.join(final_dir, 'org_norm'))
                os.makedirs(os.path.join(final_dir, 'out_norm'))
                os.makedirs(os.path.join(final_dir, 'sp_norm'))                        
            
            img_t = transforms.Compose([transforms.PILToTensor(),
                                        transforms.Lambda(lambda x: x.type(torch.FloatTensor))])

            
            # Create the dataloader object----------------------------------------------------------
            # test_set = Superposition_test(test_list, img_t, phantom_dict, beta, folders,
            #                             min_alpha, max_alpha, num_phantoms)
                        
            # testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)          

            print(f'archi:{archi} | op folder: {folders} | beta: {beta}')
            testing_params = (beta, img_t, phantom_dict, num_phantoms, min_alpha, max_alpha)

            with torch.no_grad():
                TestSpFinalNorm, TestIntfNorm, TestOpNorm = testing_proc(test_list, testing_params, device, folders, 
                                                                        TestIntfNorm, TestSpFinalNorm, 
                                                                        TestOpNorm, final_dir, count, 
                                                                        crop_size, model)

            # Run all the metrics--------------------------------------------------------------------
            com_phase_p(final_dir, folders, test_list)
            com_phase_tilde_p(final_dir, folders, TestSpFinalNorm, TestOpNorm)
            diff(final_dir, folders, TestOpNorm, TestSpFinalNorm, TestIntfNorm)
            diff_phase(final_dir, folders)
            RMSD(final_dir, folders)
            RMSD_phase(final_dir, folders)
            
            #sheet_name = final_dir.split("/")[-1] 
            #RMSD_ratio(final_dir, sheet_name.lower())
            #RMSD_ratio_phase(final_dir, sheet_name.lower() + '_phase')
    print("0")


archi_testing(final_args.archi)

