
import numpy as np, torch
from torch.utils import data
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
from . import OPENOCC_DATAWRAPPER


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)
    
    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, metas = self.point_cloud_dataset[index]
        #### adapt to vae input
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        return input, target, metas
        

def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple
