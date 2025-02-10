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
    # data_transforms_tracklet.StreamlineFlip(),
    data_transforms_tracklet.StreamlineRotate(),
])
def plotting_streamlines(streamlines, labels, file_name="default", title="Random Streamlines"):
    """
    Plots multiple 3D streamlines using Plotly.

    Parameters:
        streamlines (list of lists): A list of streamlines, where each streamline is a list of 3D coordinates.
                                     Each coordinate should be a list or tuple of (x, y, z).
        labels (list): A list of labels corresponding to each streamline.
        file_name (str): The file name for the saved plot.
        title (str): The title of the plot.
    """
    
    fig = go.Figure()
    
    # Loop through each streamline and add it to the plot
    for streamline, label in zip(streamlines, labels):
        streamline = list(zip(*streamline))  # Transpose to get x, y, z components
        fig.add_trace(go.Scatter3d(
            x=streamline[0],
            y=streamline[1],
            z=streamline[2],
            mode='lines',
            name=label  # Assign the label to each streamline for the legend
        ))

    # Update layout with axis labels, title, and legend
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(
            title="Labels"
        )
    )
    
    # Save the plot as an HTML file
    fig.write_html(f"{file_name}.html")

def load_stream(trk_path):
    """ Load streamlines from a SINGLE .trk file. """
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    return list(bundle_track.streamlines)

@DATASETS.register_module()
class Streamline_Patch_TrInF(Dataset):
    def __init__(self, config):
        logger = "Streamline_Patch_TrInF_Dataset"
        self.config = config

        self.data_dir = config.data_dir
        self.split = config.subset
        self.dataset = config.dataset
        self.num_workers = config.num_workers

        self.sample_per_subject = config.sample_per_subject
        self.samples_per_class = config.samples_per_class
        self.data_augmentation = config.data_augmentation
        self.test_subjects = config.get('test_subjects', None)

        self.patch_size = config.patch_size
        self.max_num_tokens = config.max_num_tokens
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        print_log(f"Patch size: {self.patch_size} and Number of Tokens: {self.max_num_tokens}", logger=logger)

        self.label_to_int_mapping = self.load_json(self.label_to_int_mapping_path)
        print_log(f"Starting to load streamlines...", logger=logger)
        self.load_streamlines_mp()
        print_log(f"Number of Unique Input for Model: {len(self.streamlines)}", logger=logger)
        print_log(f"Number of Unique Labels: {len(self.label_to_int_mapping)}", logger=logger)
        print_log(f"Number of Unique Subjects: {len(set(self.subject_ids))}", logger=logger)


    def load_json(self, path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        return json_data
    
    def find_duplicates(self, sub_arr):
        occurrences = {}
        duplicate_indices = []
        for idx, item in enumerate(sub_arr):
            arr_tuple = str(item)  # Convert item to string for hashing
            if arr_tuple in occurrences:
                duplicate_indices.append(idx)  # Store index of duplicate
            else:
                occurrences[arr_tuple] = idx  # Store first occurrence

        del occurrences
        return duplicate_indices
    def deduplication(self, arr):
        # arr = tracks
        # num_chunks = multiprocessing.cpu_count()
        # chunks = list(chunkify(arr, len(arr) // num_chunks))

        # with multiprocessing.Pool(processes=num_chunks) as pool:
        #     results = pool.map(find_duplicates, chunks)
        total_duplicate_indices = self.find_duplicates(arr)
        # total_duplicate_indices = [index for sublist in results for index in sublist]
        arr = [arr[i] for i in range(len(arr)) if i not in total_duplicate_indices]
        # labels = [labels[i] for i in range(len(labels)) if i not in total_duplicate_indices]
        # print(len(arr), "after deduplication")
        return arr
    
    def hash_streamline(self, streamline):
        return sha256(streamline.tobytes()).hexdigest()
    
    def load_subject(self,subject_data):
        """ Helper function to load streamlines of all classes for a particular subject. """
        subject, data_path, dataset, sample_per_subject, samples_per_class, self.max_num_tokens = subject_data
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
                # loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
                # downsample using set
                # loaded_streamlines = list(set(loaded_streamlines))
                loaded_streamlines = [loaded_streamlines[i] for i in np.random.choice(len(loaded_streamlines), samples_per_class, replace=False)]
            
            streamlines.extend(loaded_streamlines)
            streamline_to_class.update({self.hash_streamline(streamline): class_name for streamline in loaded_streamlines})

        if len(streamlines) > sample_per_subject:
            # streamlines = select_random_set_of_streamlines(streamlines, sample_per_subject)
            streamlines = [streamlines[i] for i in np.random.choice(len(streamlines), sample_per_subject, replace=False)]

        # randomlly shuffling streamlines and making it multiple of max_num_tokens by addition
        streamlines = self.deduplication(streamlines)
        np.random.shuffle(streamlines)
        num_streamlines = len(streamlines)
        num_tokens = self.max_num_tokens
        num_of_additional_streamlines = (num_tokens - (num_streamlines % num_tokens))%num_tokens
        streamlines = streamlines + streamlines[:num_of_additional_streamlines]
        labels = [streamline_to_class[self.hash_streamline(streamline)] for streamline in streamlines]

        # making groups of num_tokens streamlines
        streamlines = [
            [streamlines[j] for j in range(i, i + num_tokens)]
            for i in range(0, len(streamlines), num_tokens)
        ]
        labels = [
            [labels[j] for j in range(i, i + num_tokens)]
            for i in range(0, len(labels), num_tokens)
        ]

        return subject_id, streamlines, labels

    def load_streamlines_mp(self):
        """ Load streamlines from the dataset. 
            This function is parallelized using multiprocessing. 
            This function will only load .trk files. """

        streamlines = []
        subject_ids = []
        labels = []

        data_path = os.path.join(self.data_dir, self.split)
        subjects = [subject for subject in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, subject))]
        subjects = [subject for subject in subjects if self.test_subjects is None or subject in self.test_subjects]
        subject_data = [
            (subject, data_path, self.dataset, self.sample_per_subject, self.samples_per_class, self.max_num_tokens) for subject in subjects
        ]
        
        # for i, data_path in enumerate(data_paths):
        #     subjects_ = [subject for subject in os.listdir(data_path)
        #                  if os.path.isdir(os.path.join(data_path, subject))]
        #     if self.split[i] == 'test' or self.split[i] == 'testset':
        #         print(self.test_subjects)
        #         subjects_ = [subject for subject in subjects_ if self.test_subjects is None or subject in self.test_subjects]
        #         print(subjects_)
        #     subjects.extend(subjects_)

        #     subject_data_ = [(subject, data_path, datasets[i], self.sample_per_subject, self.samples_per_class)
        #                      for subject in subjects_]
        #     subject_data.extend(subject_data_)
        
        

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
        streamlines = self.streamlines[index]
        labels = self.labels[index]
        subject_id = self.subject_ids[index]

        #interpolating streamlines to have same number of points
        streamlines = [interpolate_streamline((streamline, self.patch_size)) for streamline in streamlines]
        streamlines = np.array(streamlines)

        # normalizing complete group of streamlines
        streamlines = streamlines.reshape(-1, 3)
        streamlines = self.pc_norm(streamlines)
        streamlines = streamlines.reshape(-1, self.patch_size, 3)

        # applying data augmentation
        if self.data_augmentation and self.split in ['train', 'trainset']:
        # if self.data_augmentation:
            streamlines = streamlines.reshape(-1, 3)
            streamlines = data_transforms(streamlines)
            streamlines = streamlines.reshape(-1, self.patch_size, 3)


        
        # getting centroids of each streamline
        centers = np.mean(streamlines, axis=1)
        point_cloud = streamlines.reshape(-1, 3)

        # normalizing streamlines with respect to centroid
        streamlines = streamlines - centers[:, None, :]

        data = {
            'streamlines': torch.tensor(streamlines).float(),
            'point_cloud': torch.tensor(point_cloud).float(),
            'center': torch.tensor(centers).float(),
        }
        labels = [self.label_to_int_mapping[label] for label in labels]

        if self.split in ['test', 'testset']:
            return np.float32(labels), [str(subject_id), str(self.streamlines[index])], data
            # return labels, [str(subject_id), str(self.streamlines[index])], data

        return np.float32(labels), 0, data
        # return labels, 0, data
    def __len__(self):
        return len(self.streamlines)

# yaml_file = "cfgs/dataset_configs/streamline_patch_TrInF.yaml"
# config = cfg_from_yaml_file(yaml_file)
# dataset = Streamline_Patch_TrInF(config)

# for i in range(10):
#     label, _ , data = dataset[0]
#     streamlines = data['streamlines']
#     plotting_streamlines(streamlines, label, file_name=f"streamline_plots/group_{i}", title=f"Streamline {i}")