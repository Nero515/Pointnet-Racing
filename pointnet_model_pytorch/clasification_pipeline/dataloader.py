from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import numpy as np

class DataLoader(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        #seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #resample
        if point_set.ndim == 2:
            point_set = point_set[choice, :]
        else:
            point_set = np.expand_dims(point_set, axis=0)
            point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, cls

    def __len__(self):
        return len(self.datapath)