# Standard libraries
import sys
import os
import json
import datetime
import gc
import pickle
import copy
from pathlib import Path
from hashlib import sha256
from multiprocessing import Pool

# Third-party libraries
import numpy as np
import torch
from tqdm import tqdm
from dipy.io.streamline import load_tractogram
from torch.utils.data import Dataset

# Project-specific imports
from .build import DATASETS
from utils.logger import *
from utils.config import cfg_from_yaml_file
from utils.tracklet_ops import get_tracklet_from_streamline
from utils.streamline_ops import interpolate_streamline


def load_stream(trk_path):
    """ Load streamlines from a SINGLE .trk file. """
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    return list(bundle_track.streamlines)

@DATASETS.register_module()
class Tracklet(Dataset):
    def __init__(self, config):
        self.config = config

        self.data_dir = config.data_dir
        self.split = config.split
        self.dataset = config.dataset

        self.sample_per_subject = config.sample_per_subject
        self.samples_per_class = config.samples_per_class

    def hash_streamline(self, streamline):
        return sha256(streamline.tobytes()).hexdigest()
    
    def load_subject(self,subject_data):
        """ Helper function to load streamlines of all classes for a particular subject. """
        subject, data_path, dataset, sample_per_subject, samples_per_class = subject_data
        subject_id = subject
        streamlines = [] # List of streamlines
        streamline_to_class = {} # Dictionary mapping streamline hash to class name

        if dataset == 'TracktoInferno':
            dir_content = Path(os.path.join(data_path, subject, 'tractography')).iterdir()
        elif dataset == '105HCP':
            dir_content = Path(os.path.join(data_path, subject, 'tracts')).iterdir()
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")
        dir_content = sorted([p for p in dir_content if p.is_file() and p.suffix == '.trk'])

        for clss in dir_content:
            if dataset == 'TracktoInferno':
                class_name = clss.stem.split('__')[-1]
            elif dataset == '105HCP':
                class_name = clss.stem
            
            # if class_name not in self.label_int_mapping.keys():
            #     continue
            
            if dataset == 'TracktoInferno':
                cls_path = os.path.join(data_path, subject, 'tractography', clss)
            elif dataset == '105HCP':
                cls_path = os.path.join(data_path, subject, 'tracts', clss)
            loaded_streamlines = load_stream(cls_path)

            # Optionally downsample if needed
            if len(loaded_streamlines) > samples_per_class:
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({self.hash_streamline(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            # streamlines = select_random_set_of_streamlines(streamlines, sample_per_subject)
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        labels = [streamline_to_class[self.hash_streamline(streamline)] for streamline in streamlines]
        return subject_id, streamlines, labels

    def load_streamlines_mp(self):
        """ Load streamlines from the dataset. 
            This function is parallelized using multiprocessing. 
            This function will only load .trk files. """

        # Handling edge cases
        if isinstance(self.data_dir, list):
            if not isinstance(self.split, list) or not isinstance(self.dataset, list):
                raise ValueError("If data_dir is a list, split and dataset must also be a list.")
            if len(self.data_dir) != len(self.split) != len(self.dataset):
                raise ValueError("data_dir, split and dataset must have the same length if list.")
            data_paths = [os.path.join(data_dir, split) for data_dir, split in zip(self.data_dir, self.split)]
            datasets = self.dataset
        else:
            if isinstance(self.split, list) or isinstance(self.dataset, list):
                raise ValueError("If data_dir is a string, split and dataset must also be a string.")
            data_paths = [os.path.join(self.data_dir, self.split)]
            datasets = [self.dataset]

        streamlines = []
        subject_ids = []
        labels = []

        subjects = []
        subject_data = []
        for i, data_path in enumerate(data_paths):
            subjects_ = [subject for subject in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, subject))]
            if self.split[i] == 'test' or self.split[i] == 'testset':
                subjects_ = [subject for subject in subjects_ if subject in self.test_subjects]
            subjects.extend(subjects_)

            subject_data_ = [(subject, data_path, datasets[i], self.sample_per_subject, self.samples_per_class)
                             for subject in subjects_]
            subject_data.extend(subject_data_)
        

        # Using multiprocessing to parallelize the loading of streamlines
        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.load_subject, subject_data), total=len(subject_data), desc='Loading Streamlines'))

        for subject_id, subject_streamlines, subject_labels in results:
            streamlines.extend(subject_streamlines)
            subject_ids.extend([subject_id] * len(subject_streamlines))
            labels.extend(subject_labels)

        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
        # self.labels = np.array(labels)
        self.labels = labels

    def __getitem__(self, index):
        """ returns tracklet of a single streamline with necessary padding
            and corresponding label """
        streamline = self.streamlines[index]
        label = self.labels[index]
        streamline = interpolate_streamline(streamline, 64)
        tracklet = get_tracklet_from_streamline(streamline)
        tracklet = np.array([interpolate_streamline(tracklet[i], 10) for i in range(len(tracklet))])
        return tracklet, label
    def __len__(self):
        return len(self.streamlines)


yaml_file = "cfgs/dataset_configs/tracklet.yaml"
config = cfg_from_yaml_file(yaml_file)
dataset = Tracklet(config)
