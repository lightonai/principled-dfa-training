import matplotlib.pyplot as plt
import seaborn as sns
import os

bottlenecks = [1, 3, 5, 8, 11, 18, 28, 46, 73, 118, 191, 308, 496, 800]
PAI = [{'acc': [36.4, 36.1], 'angles': [[0.0, 0.0], [0.2792, 0.0473], [0.9656, 0.0107]]}, {'acc': [38.5, 38.1], 'angles': [[0.0, 0.0], [0.4292, 0.052], [0.9738, 0.0074]]}, {'acc': [39.8, 39.4], 'angles': [[0.0, 0.0], [0.5164, 0.0473], [0.9777, 0.007]]}, {'acc': [40.7, 40.2], 'angles': [[0.004, 0.0493], [0.5976, 0.0463], [0.981, 0.0064]]}, {'acc': [41.8, 41.1], 'angles': [[0.012, 0.0559], [0.6516, 0.0416], [0.9839, 0.0061]]}, {'acc': [43.3, 42.4], 'angles': [[0.0184, 0.0546], [0.7268, 0.0403], [0.9856, 0.0057]]}, {'acc': [44.9, 43.0], 'angles': [[0.0329, 0.0643], [0.7858, 0.0446], [0.9867, 0.0054]]}, {'acc': [47.7, 44.7], 'angles': [[0.0591, 0.0718], [0.8364, 0.0426], [0.9873, 0.0052]]}, {'acc': [51.4, 46.2], 'angles': [[0.0932, 0.0644], [0.8831, 0.0374], [0.9883, 0.0052]]}, {'acc': [57.3, 47.2], 'angles': [[0.1533, 0.0666], [0.9214, 0.0249], [0.9894, 0.0047]]}, {'acc': [66.3, 48.2], 'angles': [[0.2431, 0.0574], [0.9508, 0.0129], [0.9904, 0.0032]]}, {'acc': [80.2, 47.2], 'angles': [[0.383, 0.0453], [0.9695, 0.0136], [0.9853, 0.0607]]}, {'acc': [93.9, 46.4], 'angles': [[0.6112, 0.0372], [0.9547, 0.121], [0.956, 0.1601]]}, {'acc': [99.4, 46.5], 'angles': [[0.976, 0.0901], [0.9608, 0.1207], [0.9645, 0.1402]]}]
STD = [{'acc': [41.2, 40.5], 'angles': [[0.0, 0.0], [0.0689, 0.0396], [0.4657, 0.0679]]}, {'acc': [41.9, 41.1], 'angles': [[0.0019, 0.0212], [0.1677, 0.0513], [0.4642, 0.0684]]}, {'acc': [42.5, 41.5], 'angles': [[0.0005, 0.0039], [0.2342, 0.0613], [0.4629, 0.0687]]}, {'acc': [43.3, 41.9], 'angles': [[0.0026, 0.0186], [0.3154, 0.0677], [0.4608, 0.0686]]}, {'acc': [44.0, 42.8], 'angles': [[0.0076, 0.0482], [0.3744, 0.0726], [0.4582, 0.0699]]}, {'acc': [45.3, 43.4], 'angles': [[0.0238, 0.0615], [0.4655, 0.0736], [0.4575, 0.0688]]}, {'acc': [46.8, 44.6], 'angles': [[0.033, 0.0589], [0.5406, 0.0767], [0.4629, 0.0691]]}, {'acc': [49.4, 45.6], 'angles': [[0.0535, 0.0615], [0.6217, 0.0725], [0.476, 0.0662]]}, {'acc': [53.0, 46.8], 'angles': [[0.0817, 0.0498], [0.6917, 0.0598], [0.4985, 0.0638]]}, {'acc': [58.5, 47.9], 'angles': [[0.1375, 0.0499], [0.7597, 0.0361], [0.5355, 0.0524]]}, {'acc': [67.7, 47.6], 'angles': [[0.2264, 0.0503], [0.826, 0.0237], [0.5836, 0.0403]]}, {'acc': [81.4, 46.8], 'angles': [[0.3726, 0.0434], [0.8705, 0.0191], [0.6405, 0.0283]]}, {'acc': [94.4, 46.4], 'angles': [[0.6081, 0.0439], [0.8953, 0.0756], [0.6796, 0.0607]]}, {'acc': [99.5, 47.1], 'angles': [[0.9686, 0.1004], [0.8924, 0.1338], [0.677, 0.1051]]}]

b_list = []
pai_train_acc, pai_test_acc = [], []
std_train_acc, std_test_acc = [], []

pai_b_angles, pai_b_std = [], []
std_b_angles, std_b_std = [], []

for i, b in enumerate(bottlenecks):
    b_list.append(b)
    pai_train_acc.append(PAI[i]['acc'][0])
    pai_test_acc.append(PAI[i]['acc'][1])
    std_train_acc.append(STD[i]['acc'][0])
    std_test_acc.append(STD[i]['acc'][1])
    pai_b_angles.append(PAI[i]['angles'][1][0])
    pai_b_std.append(PAI[i]['angles'][1][1])
    std_b_angles.append(STD[i]['angles'][1][0])
    std_b_std.append(STD[i]['angles'][1][1])

sns.set()

#plt.errorbar(b_list, pai_b_angles, yerr=pai_b_std, fmt='.')
plt.errorbar(b_list, std_b_angles, yerr=std_b_std, fmt='.')
plt.xlabel('Bottleneck size [neurons]')
plt.ylabel('Alignment angle on bottleneck layer')
plt.ylim([0, 1])
plt.xscale('log')
# TODO: Select where to save file.
figure_save_file = os.path.join('', "A_bot_LB.png")
plt.savefig(figure_save_file)

plt.clf()

plt.scatter(b_list, std_test_acc)
plt.xlabel('Bottleneck size [neurons]')
plt.ylabel('Accuracy [%]')
plt.xscale('log')
#plt.yscale('log')
# TODO: Select where to save file.
figure_save_file = os.path.join('', "A_bot_acc.png")
plt.savefig(figure_save_file)
