from testing_metrics import RMSD_ratio, RMSD_ratio_phase
import os
#import pandas as pd

for cf in [1, 2, 3]:
    for archi in ['ResNet18', 'ResNet34', 'UNet', 'FCDenseNet', 'DnCNN_plus', 'DnCNN_minus', 'Double_DnCNN_minus', 'DnCNN_paper', 'DnCNN_paper_pT']:
        RMSD_ratio(os.path.join(f'cross_fold_{cf}', archi), archi.lower() + f'_{cf}')
        RMSD_ratio_phase(os.path.join(f'cross_fold_{cf}', archi), archi.lower() + f'_phase_{cf}')
