import os
import random
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

logger = getLogger()


class ADNI_DX_scale(Dataset):
    def __init__(
        self, 
        split='',
        processed_dir='',
        use_normalization=False,
        downsample=False,
        sampling_rate=3,
        num_frames=160,
    ):
        self.use_normalization = use_normalization
        
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        
        self.n_rois = 450
        self.seq_length = 490
        self.root_dir = ''
        # os.makedirs(processed_dir, exist_ok=True)
        
        self.npz_data = np.load(processed_dir)
        
        self.input_xs = torch.tensor(self.npz_data["X"], dtype=torch.float32).transpose(1, 2)
        self.label_ys = torch.tensor(self.npz_data["Y"], dtype=torch.float32)

        # ckp = torch.load(processed_dir)
        # self.input_xs = ckp[split]['X']
        # self.label_ys = 1 - ckp[split]['Y']
            
    def __len__(self):
        return len(self.input_xs)

    def __getitem__(self, idx):
        input_x, label_y = self.input_xs[idx], self.label_ys[idx]
        input_x = input_x.float()

        if self.use_normalization:
            mean = input_x.mean()
            std = input_x.std()
            input_x = (input_x - mean) / std
        
        # if self.downsample:
        #     clip_size = self.sampling_rate * self.num_frames
        #     start_idx, end_idx = self._get_start_end_idx(self.seq_length, clip_size)
        #     ts_array = self._temporal_sampling(
        #                 input_x, start_idx, end_idx, self.num_frames
        #             )
        #     input_x = torch.unsqueeze(ts_array, 0).to(torch.float32)
        # else:
        #     input_x = torch.unsqueeze(input_x, 0).to(torch.float32)  
        input_x = torch.unsqueeze(input_x, 0).to(torch.float32)  
        
        return input_x.to(torch.float32), int(label_y)
    
    def _get_start_end_idx(self, fmri_size, clip_size):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx
    
    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        new_frames = torch.index_select(frames, 1, index)
        return new_frames
    
    
def make_adni_dx(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    drop_last=True,
    processed_dir='',
    use_normalization=False,
    label_normalization=False,
    downsample=False,
):    
    dataset = ADNI_DX_scale(
        processed_dir=processed_dir,
        use_normalization=use_normalization,
        downsample=downsample
    )

    indices = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in indices])  # 获取所有的标签

    # 先按 8:2 划分 train 和 (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, valid_test_idx = next(sss1.split(indices, labels))

    # 再按 2:2 划分 val 和 test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss2.split(valid_test_idx, labels[valid_test_idx]))

    # 创建子数据集
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    train_data_loader = DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    valid_data_loader = DataLoader(
        valid_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    test_data_loader = DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    logger.info('adni_dx dataset created')

    return train_data_loader, valid_data_loader, test_data_loader, train_dataset, valid_dataset, test_dataset

