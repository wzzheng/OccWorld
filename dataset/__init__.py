
from mmengine.registry import Registry
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')

from .dataset import nuScenesSceneDatasetLidar, nuScenesSceneDatasetLidarTraverse
from .dataset_wrapper import tpvformer_dataset_nuscenes, custom_collate_fn_temporal
from .sampler import CustomDistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
import yaml

def get_dataloader(
    train_dataset_config, 
    val_dataset_config, 
    train_wrapper_config,
    val_wrapper_config,
    train_loader, 
    val_loader, 
    nusc=dict(
        version='v1.0-trainval',
        dataroot='data/nuscenes'),
    dist=False,
    iter_resume=False,
    train_sampler_config=dict(
        shuffle=True,
        drop_last=True),
    val_sampler_config=dict(
        shuffle=False,
        drop_last=False),
):
    # if nusc is not None:
    #     from nuscenes import NuScenes
    #     nusc = NuScenes(**nusc)
    
    train_dataset = OPENOCC_DATASET.build(
        train_dataset_config,
        default_args={'nusc': nusc})
    val_dataset = OPENOCC_DATASET.build(
        val_dataset_config,
        default_args={'nusc': nusc})
    
    train_wrapper = OPENOCC_DATAWRAPPER.build(
        train_wrapper_config,
        default_args={'in_dataset': train_dataset})
    val_wrapper = OPENOCC_DATAWRAPPER.build(
        val_wrapper_config,
        default_args={'in_dataset': val_dataset})
    
    train_sampler = val_sampler = None
    if dist:
        if iter_resume:
            train_sampler = CustomDistributedSampler(train_wrapper, **train_sampler_config)
        else:
            train_sampler = DistributedSampler(train_wrapper, **train_sampler_config)
        val_sampler = DistributedSampler(val_wrapper, **val_sampler_config)

    train_dataset_loader = DataLoader(
        dataset=train_wrapper,
        batch_size=train_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False if dist else train_loader["shuffle"],
        sampler=train_sampler,
        num_workers=train_loader["num_workers"],
        pin_memory=True)
    val_dataset_loader = DataLoader(
        dataset=val_wrapper,
        batch_size=val_loader["batch_size"],
        collate_fn=custom_collate_fn_temporal,
        shuffle=False,
        sampler=val_sampler,
        num_workers=val_loader["num_workers"],
        pin_memory=True)

    return train_dataset_loader, val_dataset_loader


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]
    return nuScenes_label_name


