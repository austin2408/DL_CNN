import numpy as np
import pandas as pd
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class dataloader(Dataset):
    def __init__(self, data_path, imagepath, mode='train'):
        self.mode = mode
        self.path = [data_path,imagepath]
        self.img_list = np.array(pd.read_csv(self.path[0]+'/'+self.mode+'_img.csv',header=None))
        self.label_list = np.array(pd.read_csv(self.path[0]+'/'+self.mode+'_label.csv',header=None))
        self.img_name = []
        self.label = []

        for i in range(self.img_list.shape[0]):
            self.img_name.append(self.path[1]+'/'+str(self.img_list[i][0])+'.jpeg')
            self.label.append(int(self.label_list[i][0]))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_name[idx])
        label = self.label[idx]
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5769, 0.3852, 0.2649],std=[0.1061, 0.0809, 0.0555])])

        image_tensor = transform(image).float()
        sample = {"Image": image_tensor, "label": label}

        return sample