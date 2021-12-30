import os
import matplotlib.pyplot as plt


File = os.listdir('/home/austin/nctu_hw/DL/DL_hw3/loss_record/EEG')
File.sort()

LIST = []
mode = []
for name in File:
    List = []
    b = name.split('_')[3]
    model = name.split('_')[2]
    act = name.split('_')[6].split('.')[0]
    name = '/home/austin/nctu_hw/DL/DL_hw3/loss_record/EEG/' + name
    file = open(name, "rt")
    line = file.readlines()
    for s in line:
        s = float(s.strip('\n'))
        List.append(s)

    line1, = plt.plot(List, label=model+'_'+act)
    LIST.append(line1)
    mode.append(model+'_'+act+'_'+b)

plt.legend(LIST, mode, loc='upper right')
plt.title('Loss curve')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
# plt.savefig('/home/austin/nctu_hw/DL/DL_hw3/Figure/Deep_loss_compare.png',dpi=300)
plt.show()

File = os.listdir('/home/austin/nctu_hw/DL/DL_hw3/acc_record/EEG')
File.sort()

LIST = []
mode = []
for name in File:
    List = []
    b = name.split('_')[3]
    Set = name.split('_')[0]
    model = name.split('_')[2]
    act = name.split('_')[6].split('.')[0]
    name = '/home/austin/nctu_hw/DL/DL_hw3/acc_record/EEG/' + name
    file = open(name, "rt")
    line = file.readlines()
    for s in line:
        s = float(s.strip('\n'))
        List.append(s)

    line1, = plt.plot(List, label=model+'_'+act+'_'+Set+'_'+b)
    LIST.append(line1)
    mode.append(model+'_'+act+'_'+Set+'_'+b)

plt.legend(LIST, mode, loc='upper left')
plt.title('Acc curve')
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.grid(True)
# plt.savefig('/home/austin/nctu_hw/DL/DL_hw3/Figure/Deep_acc_compare.png',dpi=300)
plt.show()