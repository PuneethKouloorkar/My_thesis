import os
import pickle
import random
from path_file import num_img

# # Global variable-------------------------------------------------------
out_fol, infs_path_1, infs_path_2, emp_infs_path = num_img()
# out_fol = ''
# infs_path_1 = ''                             #InterferogramML4
# infs_path_2 = ''                             #InterferogramML5
# emp_infs_path = ''                           #20201127_zero_images
# # Global variable-------------------------------------------------------

tot_intfs = [i for i in range(1, 241)]
test_intfs = []
for i in range(6):
    a = random.choice(tot_intfs)
    test_intfs.append(a)
    tot_intfs.remove(a)

#print(test_intfs)

f = open(os.path.join(out_fol, 'test_samples.txt'), 'w')
f.write(f'{test_intfs}')
f.close()

if __name__ == '__main__':
    # Interferograms
    intfs = []

    for i in os.listdir(infs_path_2):
        if i.startswith('set'):
            set_no = int(i[4:])            # New lines
            if set_no not in test_intfs:   # New lines
                for j in os.listdir(os.path.join(infs_path_2, i)):
                    if j.endswith('.bmp'):
                        intfs.append(os.path.join(infs_path_2,i,j))
    
    for i in os.listdir(infs_path_1):
        if i.startswith('set'):
            for j in os.listdir(os.path.join(infs_path_1, i)):
                if j.endswith('.bmp'):
                    intfs.append(os.path.join(infs_path_1,i,j))

    t_1 = []
    for i in os.listdir(infs_path_2):
        if i.startswith('set'):
            set_no = int(i[4:])        # New lines
            if set_no in test_intfs:   # New lines
                for j in os.listdir(os.path.join(infs_path_2, i)):
                    if j.endswith('.bmp'):
                        t_1.append(os.path.join(infs_path_2,i,j))

    print(f'Intferogram count: {len(intfs)}')
    
    open_file = open(os.path.join(out_fol, 'intf_imgs.pkl'), "wb")
    pickle.dump(intfs, open_file)
    open_file.close()

    #for i,j in enumerate([t_1, t_2, t_3, t_4, t_5]):
    for i,j in enumerate([t_1]):
        open_file = open(os.path.join(out_fol, f'intf_imgs_test_{i+1}.pkl'), "wb")
        pickle.dump(j, open_file)
        open_file.close()

    # Phantom images
    m1_nosample = []
    m2_nosample = []
    m3_nosample = []
    m4_nosample = []

    for k in os.listdir(emp_infs_path):
        for l in os.listdir(os.path.join(emp_infs_path, k)):
            if l.startswith('Images'):
                for m in os.listdir(os.path.join(emp_infs_path, k, l,'pos1')):
                    if m.startswith('m1') and m.endswith('.bmp'):
                        m1_nosample.append(os.path.join(emp_infs_path,k,l,'pos1',m))
                    elif m.startswith('m2') and m.endswith('.bmp'):
                        m2_nosample.append(os.path.join(emp_infs_path,k,l,'pos1',m))
                    elif m.startswith('m3') and m.endswith('.bmp'):
                        m3_nosample.append(os.path.join(emp_infs_path,k,l,'pos1',m))
                    elif m.startswith('m4') and m.endswith('.bmp'):
                        m4_nosample.append(os.path.join(emp_infs_path,k,l,'pos1',m))
            elif l.startswith('m1') and l.endswith('.bmp'):
                m1_nosample.append(os.path.join(emp_infs_path,k,l))
            elif l.startswith('m2') and l.endswith('.bmp'):
                m2_nosample.append(os.path.join(emp_infs_path,k,l))
            elif l.startswith('m3') and l.endswith('.bmp'):
                m3_nosample.append(os.path.join(emp_infs_path,k,l))
            elif l.startswith('m4') and l.endswith('.bmp'):
                m4_nosample.append(os.path.join(emp_infs_path,k,l))
            else:
                continue

    print(f'Phantom count: {len(m1_nosample), len(m2_nosample), len(m3_nosample), len(m4_nosample)}')

    for idx, m in enumerate([m1_nosample, m2_nosample, m3_nosample, m4_nosample]):
        open_file = open(os.path.join(out_fol, f'Phantom_imgs_{idx+1}.pkl'), "wb")
        pickle.dump(m, open_file)
    open_file.close()