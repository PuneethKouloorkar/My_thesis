import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (15,10)


archi = ['DnCNN+', 'DnCNN-', 'Double_DnCNN-', 'DnCNN_paper', 'DnCNN_paper_pT',
         'ResNet18', 'ResNet34', 'UNet', 'FCDenseNet']
ratio = [0.2, 0.3, 0.12, 0.5, 0.8, 2.0, 3.0, 0.8, 0.9]
y_pos = np.arange(len(archi))

plt.bar(y_pos, ratio)
plt.ylabel('Ratio')
plt.xticks(y_pos, archi)
plt.title('Interferogram ratio')

for index, data in enumerate(ratio):
    plt.text(index-0.1, data+0.05, str(data))

plt.savefig('bar_plot.png')
#plt.show()
