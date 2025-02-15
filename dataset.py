from torch.utils.data import Dataset
import glob
import torch
import csiread
import numpy as np
import torch.nn.functional as F
import scipy.signal as signal
from utils import truncate_or_pad_csi,standard,butterworth,linear_insert
class CSI_Dataset_Exist(Dataset):
    """CSI dataset."""
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir + '/*.dat')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_dir = self.data_list[idx]
        y = int(sample_dir.split('_')[-1].split('.')[0])

        wifi = csiread.Intel(sample_dir,nrxnum=3,ntxnum=3,if_report=False)
        wifi.read()
        csi = wifi.get_scaled_csi()

        csi = csi[:,:,:,:2] #使用发送天线数
        x = np.abs(csi)
        x = np.apply_along_axis(signal.medfilt, 0, x, 3)  # 中值滤波,窗口必须为奇数，此处窗口为3
        x = np.apply_along_axis(butterworth,0,x,4,5,"low",100) #4阶滤波器，截止频率5，采样率100

        x = np.apply_along_axis(linear_insert,0,x, 500)#线性插值为500
        x = np.apply_along_axis(standard,0,x)#归一化
        x = torch.from_numpy(x).float()
        return x,y

if __name__ == '__main__':
    root = 'D:\desk\dataset\\11'
    dataset = CSI_Dataset_Exist(root)
    data = dataset[40]
    print('label:',data[1])
    print(data[0].shape)