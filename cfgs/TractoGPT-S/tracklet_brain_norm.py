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
import plotly.graph_objs as go
import plotly.express as px

# Third-party libraries
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from dipy.io.streamline import load_tractogram
from torch.utils.data import Dataset

# Project-specific imports
from .build import DATASETS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import *
from utils.config import cfg_from_yaml_file
from utils.tracklet_ops import get_tracklet_from_streamline
from utils.streamline_ops import interpolate_streamline
from datasets import data_transforms_tracklet

data_transforms = transforms.Compose([
    data_transforms_tracklet.StreamlineFlip(),
    data_transforms_tracklet.StreamlineRotate(),
])
def plot_point_cloud(points, red_points=None, filename='plot', title="3D Point Cloud"):
    """
    Plots a 3D point cloud using Plotly.

    Parameters:
        points (numpy.ndarray): A NumPy array of shape (X, 3) containing 3D point coordinates.
        title (str): Title of the plot (default: "3D Point Cloud").
    """
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        raise ValueError("Input must be a NumPy array of shape (X, 3)")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=1, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    ))
    if red_points is not None:
        fig.add_trace(go.Scatter3d(
            x=red_points[:, 0],
            y=red_points[:, 1],
            z=red_points[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', opacity=0.8)
        ))

    fig.update_layout(title=title, scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
    ))
    fig.write_html(f"{filename}.html")
    fig.show()

def load_stream(trk_path):
    """ Load streamlines from a SINGLE .trk file. """
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    return list(bundle_track.streamlines)

@DATASETS.register_module()
class Tracklet_Brain_Norm(Dataset):
    def __init__(self, config):
        logger = "Tracklet_Brain_Norm_Dataset"
        self.config = config

        self.data_dir = config.data_dir
        self.split = config.subset
        self.dataset = config.dataset
        self.num_workers = config.num_workers

        self.sample_per_subject = config.sample_per_subject
        self.samples_per_class = config.samples_per_class
        self.data_transformation = config.data_transformation
        self.test_subjects = config.get('test_subjects', None)

        self.max_num_tokens = config.max_num_tokens
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        self.patch_size = 12
        print_log(f"Patch size: {self.patch_size}", logger=logger)

        self.label_to_int_mapping = self.load_json(self.label_to_int_mapping_path)
        print_log(f"Starting to load streamlines...", logger=logger)
        self.load_streamlines_mp()


    def load_json(self, path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        return json_data
    
    def hash_streamline(self, streamline):
        return sha256(streamline.tobytes()).hexdigest()
    
    def load_subject(self,subject_data):
        """ Helper function to load streamlines of all classes for a particular subject. """
        subject, data_path, dataset, sample_per_subject, samples_per_class = subject_data
        subject_id = subject
        streamlines = [] # List of streamlines
        streamline_to_class = {} # Dictionary mapping streamline hash to class name

        if dataset == 'TractoInferno':
            dir_content = Path(os.path.join(data_path, subject, 'tractography')).iterdir()
        elif dataset == '105HCP':
            dir_content = Path(os.path.join(data_path, subject, 'tracts')).iterdir()
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")
        dir_content = sorted([p for p in dir_content if p.is_file() and p.suffix == '.trk'])

        for clss in dir_content:
            if dataset == 'TractoInferno':
                class_name = clss.stem.split('__')[-1]
            elif dataset == '105HCP':
                class_name = clss.stem
            
            if class_name not in self.label_to_int_mapping.keys():
                continue
            
            if dataset == 'TractoInferno':
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
        
        # normalize complete brain using norm_pc 
        streamline_lengths = [len(streamline) for streamline in streamlines]
        streamlines = np.concatenate(streamlines, axis=0)
        streamlines = self.pc_norm(streamlines)
        # subject_center = np.mean(streamlines, axis=0) # return a single point in 3d coordinates
        streamlines = data_transforms(streamlines)
        streamlines = np.split(streamlines, np.cumsum(streamline_lengths)[:-1])

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
            self.split = [self.split]

        streamlines = []
        subject_ids = []
        labels = []

        subjects = []
        subject_data = []
        for i, data_path in enumerate(data_paths):
            subjects_ = [subject for subject in os.listdir(data_path)
                         if os.path.isdir(os.path.join(data_path, subject))]
            if self.split[i] == 'test' or self.split[i] == 'testset':
                print(self.test_subjects)
                subjects_ = [subject for subject in subjects_ if self.test_subjects is None or subject in self.test_subjects]
                print(subjects_)
            subjects.extend(subjects_)

            subject_data_ = [(subject, data_path, datasets[i], self.sample_per_subject, self.samples_per_class)
                             for subject in subjects_]
            subject_data.extend(subject_data_)
        
        

        # Using multiprocessing to parallelize the loading of streamlines
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.load_subject, subject_data), total=len(subject_data), desc='Loading Streamlines'))

        for subject_id, subject_streamlines, subject_labels in results:
            streamlines.extend(subject_streamlines)
            subject_ids.extend([subject_id] * len(subject_streamlines))
            labels.extend(subject_labels)

        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
        # self.labels = np.array(labels)
        self.labels = labels

    def zero_padding(self, tracklet):
        """ Zero pads the tracklet to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            tracklet = np.concatenate((tracklet, np.zeros((padding, self.patch_size, 3))), axis=0)
        return tracklet
    
    def pad_with_noise(self, tracklet):
        """ Zero pads the tracklet with noise to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            noise = np.random.normal(0, 1e-4, (padding, self.patch_size, 3))
            tracklet = np.concatenate((tracklet, noise), axis=0)
        return tracklet
    
    # padding the remaining tracklet by repeating the last element
    def repeat_last_padding(self, tracklet):
        """ Repeats the last element of the tracklet to pad it to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            tracklet = np.concatenate((tracklet, np.tile(tracklet[-1], (padding, 1, 1))), axis=0)
        return tracklet
    
    # repeat all the tracklets to the maximum number of tokens
    def repeat_all_padding(self, tracklet):
        """ Repeats the complete tracklet to pad it to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            num_tracklets = len(tracklet)
            complete_reps = padding // num_tracklets
            partial_rep = padding % num_tracklets
            tracklet = np.concatenate((tracklet, np.tile(tracklet, (complete_reps, 1, 1))), axis=0)
            tracklet = np.concatenate((tracklet, tracklet[:partial_rep]), axis=0)
        return tracklet
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __getitem__(self, index):
        """ returns tracklet of a single streamline with necessary padding
            and corresponding label """
        streamline = self.streamlines[index]
        label = self.labels[index]
        label = self.label_to_int_mapping[label]

        # getting tracklet from streamline and randomly transforming it
        streamline = interpolate_streamline((streamline, 64))
        # if self.data_transformation and self.split in ['train', 'trainset']: streamline = data_transforms(streamline)
        # streamline = self.pc_norm(streamline)
        tracklet = get_tracklet_from_streamline(streamline, apt_param=.4)
        tracklet = np.array([interpolate_streamline((tracklet[i], self.patch_size)) for i in range(len(tracklet))])

        # # normalizing streamline, and getting distance from subject_center and tracklet
        # tracklet_len = self.patch_size*np.ones((len(tracklet)), dtype=np.int32)
        # tracklet_minus_center = tracklet - self.subject_id_to_subject_center[self.subject_ids[index]]
        # streamline = np.concatenate(tracklet, axis=0)
        # streamline = self.pc_norm(streamline)
        # tracklet = np.split(streamline, np.cumsum(tracklet_len)[:-1])


        # zero padding
        padding = len(tracklet)
        tracklet = self.zero_padding(tracklet)
        # tracklet_minus_center = self.zero_padding(tracklet_minus_center)
        # tracklet = self.repeat_last_padding(tracklet)
        # tracklet = self.repeat_all_padding(tracklet)
        # tracklet = self.pad_with_noise(tracklet)

        # normalizing tracklet with respect to the center points
        tracklet_center = np.mean(tracklet, axis=1)
        tracklet = tracklet - tracklet_center[:, None, :]

        data = {
            'tokens': torch.from_numpy(tracklet).float(),
            'center': torch.from_numpy(tracklet_center).float(),
            'padding': np.float32(padding),
            # 'tracklet_minus_center': torch.from_numpy(tracklet_minus_center).float()
        }
        if self.split[0] == 'test' or self.split[0] == 'testset':
            return np.float32(label),[str(self.subject_ids[index]), str(self.streamlines[index])],data
        return np.float32(label),0,data
    def __len__(self):
        return len(self.streamlines)

# yaml_file = "cfgs/dataset_configs/tracklet_center_aware.yaml"
# config = cfg_from_yaml_file(yaml_file)
# dataset = Tracklet_Brain_Norm(config)
# # tracklet, tracklet_center, padding = dataset[100]
# # x = tracklet_center[:,None,:][:padding]
# # print(tracklet.shape)
# # print(x.shape)
# # plotting_tracklet(x)
# for i in range(10):
#     label, _, data = dataset[0]
#     tracklet_minus_center = data['tracklet_minus_center'].numpy()
#     tracklet = data['tokens'].numpy()
#     tracklet_minus_center = np.concatenate(tracklet_minus_center, axis=0)
#     tracklet = np.concatenate(tracklet, axis=0)
#     plot_point_cloud(tracklet, tracklet_minus_center, filename=f"streamline_plots/tracklet_{i}", title=f"Tracklet {i}")