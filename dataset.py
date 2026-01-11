import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from glob import glob
from scipy.io import loadmat
from tqdm import tqdm
import random


def loader(dataset, device):
    """
        Data generator
    """
    for data, label in dataset:
        data = data.to(device)
        label = label.to(device)
        yield data, label


class ColoredNumber(data.dataset.Dataset):
    def __init__(self, dt=0.05, T=10, dataset_type="3-color", transform=None):
        self.dt = dt
        self.T = T
        self.transform = transform

        self.pattern = {
            0: [[1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1]],
            1: [[0, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 1, 0]],
            4: [[1, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0]],
            7: [[1, 1, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]]
        }

        # color to spiking period (1/Hz) mapping
        self.color_mapping = {'red': 1/1.6, 'green': 1/1.2, 'blue': 1/1.0}
        self.vals = []  # [n_sample, n_unit]
        self.cs = []
        if dataset_type == "3-color":
            self.vals = np.array([np.array(p).flatten() * f for p in self.pattern.values() for f in self.color_mapping.values()])  # [n_sample, n_unit]
            self.cs = np.array([c for _ in self.pattern.values() for c in range(len(self.color_mapping.values()))])
        else:
            self.color_mapping = {
                'red': [1 / 1.6, 0., 0.],
                'green': [0., 1 / 1.2, 0.],
                'blue': [0., 0., 1 / 1.0],
                'yellow': [1 / 1.6, 1 / 1.2, 0.],
                'magenta': [1 / 1.6, 0., 1 / 1.0],
                'cyan': [0., 1 / 1.2, 1 / 1.0],
                'white': [1 / 1.6, 1 / 1.2, 1 / 1.0]
            }
            for p in self.pattern.values():
                for c in self.color_mapping.values():
                    temp = []
                    for f in c:
                        temp.append(np.array(p).flatten() * f)
                    self.vals.append(np.concatenate(temp))
            self.vals = np.array(self.vals)
            for _ in self.pattern.values():
                for c in range(len(self.color_mapping.values())):
                    self.cs.append(c)
            self.cs = np.array(self.cs)

        self.vals = np.where(self.vals == 0, 99999., self.vals)
        self.vals = torch.Tensor(self.vals)
        self.cs = torch.Tensor(self.cs).to(dtype=torch.int64)

        self.vals_encoded = torch.zeros([self.vals.shape[0], int(T / dt), self.vals.shape[1]])
        for i, s in enumerate(self.vals):
            self.vals_encoded[i] = self.encode(s)

    def __getitem__(self, index):
        sample = (self.vals_encoded[index], self.cs[index])
        if self.transform is not None:
            self.transform(sample)
        return sample

    def __len__(self):
        return len(self.cs)

    def encode(self, sample):
        timestep = int(self.T / self.dt)
        raster = torch.zeros([timestep, len(sample)])
        # compute cumulative spike times for all channels
        spike_times = torch.cumsum(sample.unsqueeze(0).repeat(timestep, 1), dim=0)
        spike_times = spike_times + spike_times * torch.randn_like(spike_times) * 0.01  # add noise to spike times
        # convert continuous spike times to discrete timesteps
        spike_times = torch.round(spike_times / self.dt).to(dtype=torch.int)
        # create a mask for valid indices within bounds
        valid_mask = spike_times < timestep
        # populate the spike raster using advanced indexing
        print(spike_times[valid_mask])
        raster[spike_times[valid_mask], torch.arange(len(sample)).unsqueeze(0).repeat(timestep, 1)[valid_mask]] = 1
        return raster

    def get_classes(self, numeric=True):
        return range(len(self.color_mapping.values())) if numeric else self.color_mapping.keys()


class Karlsruhe(data.dataset.Dataset):
    def __init__(
            self, data_path='karlsruhe\\objects_2011_a\\labeldata_344x100',
            target_path='karlsruhe\\objects_2011_a\\labeldata',
            dt=0.05, T=10, mixup_alpha=0.2, jitter=0., invert=False, transform=None, device='cuda'
    ):
        assert os.path.exists(data_path), f'Data path {data_path} does not exist. Please prepare the dataset first by running generate_karlsruhe.py.'

        self.dt = dt
        self.T = T
        self.mixup_alpha = mixup_alpha
        self.jitter = jitter
        self.transform = transform
        self.device = device
        self.scale_to_min = 1.0
        self.scale_to_max = 1.8  # scale to [1, 1.8]
        self.max_spike_count = int(
            T / self._get_period(self.scale_to_max)) + 5  # add some extra spikes to avoid overflow
        self.use_transform = False

        fs_img_pos = glob(f'{data_path}/pos/*.png')
        fs_img_neg = glob(f'{data_path}/neg/*.png')
        fs_tar_pos = glob(f'{target_path}/pos/*.mat')
        fs_img_pos = sorted(fs_img_pos, key=lambda x: os.path.basename(x))[:]
        fs_img_neg = sorted(fs_img_neg, key=lambda x: os.path.basename(x))[:]
        fs_tar_pos = sorted(fs_tar_pos, key=lambda x: os.path.basename(x))[:]

        assert [f.split('/')[-1].split('.')[0][1:] for f in fs_img_pos] == [f.split('/')[-1].split('.')[0][1:] for f in fs_tar_pos]

        self.tars = []
        self.vals = []
        self.fnames = []

        for f_img, f_tar in tqdm(zip(fs_img_pos, fs_tar_pos), total=len(fs_tar_pos)):
            img = mpimg.imread(f_img)
            img = np.array(img, dtype=float)
            if invert:
                img = img * (-1.) + 1.
            self.vals.append(torch.Tensor(img))

            data = loadmat(f_tar)['L']
            data = pd.DataFrame(data, columns=['U', 'V', 'Width', 'Height', 'Class', 'Heading', 'Easy'])
            data = data[data['Easy'] == 1.]
            data = np.unique(data['Class'].to_numpy()) - 1
            self.tars.append(data.astype(int))
            
            self.fnames.append(f_img.split('/')[-1].split('.')[0])

        for f_img in tqdm(fs_img_neg, total=len(fs_img_neg)):
            img = mpimg.imread(f_img)
            img = np.array(img, dtype=float)
            if invert:
                img = img * (-1.) + 1.
            self.vals.append(torch.Tensor(img))

            self.tars.append(None)
            
            self.fnames.append(f_img.split('/')[-1].split('.')[0])

        self.class_names = ['Car_truck', 'Person']
        self.class_indices = np.arange(len(self.class_names))

        # one-hot encode each array
        self.cs = []
        for tar in self.tars:
            encoded = np.zeros(len(self.class_names), dtype=int)
            if tar is not None:
                encoded[tar] = 1
            self.cs.append(encoded)
        self.cs = np.array(self.cs)

        self.vals = torch.stack(self.vals, dim=0)
        self.cs = torch.Tensor(self.cs).to(dtype=torch.int64)

        self.train_indices = []

    def _get_first_spike_time(self, x):
        a, b = 1.14677454592408, 0.968298667327956
        return a * np.log(x / (x - b))

    def _get_period(self, x):
        a, b, c = 2.5, 1.02, 0.239
        return 1 / (a * (x - b) ** 2 + c)
    
    def get_item_with_fname(self, index):
        x, y = self.__getitem__(index)
        return (x, y, self.fnames[index])

    def __getitem__(self, index):

        img = self.vals[index]
        label = self.cs[index]

        if self.transform is not None and self.use_transform:
            lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)  # sample mixing coefficient
            lambda_ = max(lambda_, 1 - lambda_)  # symmetric mixing (ensure >= 0.5)

            index_mix = random.choice(self.train_indices)
            img_mix = self.vals[index_mix]
            label_mix = self.cs[index_mix]

            img = lambda_ * img + (1 - lambda_) * img_mix  # mix images
            label = lambda_ * label + (1 - lambda_) * label_mix  # mix labels

            img = self.transform(img).squeeze()

        img = img * (self.scale_to_max - self.scale_to_min) + self.scale_to_min
        ttfs = self._get_first_spike_time(img)
        period = self._get_period(img)
        sample = (self.encode(ttfs, period), label)
        return sample

    def __len__(self):
        return len(self.cs)

    def encode(self, sample_ttfs, sample_period):
        """
            Encode an image into spike trains.

            Args:
                sample_ttfs (torch.Tensor): Time to first spike, shape (width, height).
                sample_period (torch.Tensor): Spiking period, shape (width, height).

            Returns:
                torch.Tensor: Spike train tensor, shape (int(T/dt), 1, width, height).
            """
        sample_ttfs = sample_ttfs.to(self.device)
        sample_period = sample_period.to(self.device)
        width, height = sample_ttfs.shape
        timestep = int(self.T / self.dt)
        raster = torch.zeros((timestep, width, height), dtype=torch.float32, device=self.device)

        # compute cumulative spike times
        first_spike_times = sample_ttfs.unsqueeze(0).repeat(self.max_spike_count, 1, 1)  # repeat for max_spike_count
        periods = sample_period.unsqueeze(0).repeat(self.max_spike_count, 1, 1)  # repeat for max_spike_count
        periods = torch.cumsum(periods, dim=0) - periods  # get times of spikes by cumulating, subtract one period to add ttfs
        spike_times = first_spike_times + periods
        spike_times += torch.randn_like(spike_times) * self.jitter
        # convert spike times to discrete timesteps
        spike_indices = torch.round(spike_times / self.dt).to(dtype=torch.int)
        # create a mask for valid indices within bounds
        valid_mask = spike_indices < timestep
        # populate the spike raster using advanced indexing
        raster[
            spike_indices[valid_mask],
            torch.arange(width, device=self.device)[None, ..., None].repeat(self.max_spike_count, 1, height)[
                valid_mask],
            torch.arange(height, device=self.device)[None, None, ...].repeat(self.max_spike_count, width, 1)[valid_mask]
        ] = 1
        return raster.unsqueeze(1)  # add channel dimension

    def get_classes(self, numeric=True):
        return np.arange(10) if numeric else self.class_names
