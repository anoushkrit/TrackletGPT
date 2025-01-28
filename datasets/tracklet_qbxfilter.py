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
from collections import defaultdict
from multiprocessing import Pool
import plotly.graph_objs as go
import plotly.express as px

# Third-party libraries
import numpy as np
import torch
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
from utils.qbx_clustering import create_qbx_cluster_from_scratch_n_save

def plotting_tracklet(tracklet, file_name="default", title="Tracklet"):
    """
    Plots multiple 3D streamlines using Plotly.

    Parameters:
        streamlines (list of lists): A list of streamlines, where each streamline is a list of 3D coordinates.
                                     Each coordinate should be a list or tuple of (x, y, z).
        labels (list): A list of labels corresponding to each streamline.
        title (str): The title of the plot.
    """
    
    fig = go.Figure()
    # print(tracklet.shape)
    for track in tracklet:
        track = list(zip(*track))  # Transpose to get x, y, z components
        fig.add_trace(go.Scatter3d(
            x=track[0],
            y=track[1],
            z=track[2],
            mode='markers',
        ))

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
    
    fig.write_html(f"{file_name}.html")

def load_stream(trk_path):
    """ Load streamlines from a SINGLE .trk file. """
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    return list(bundle_track.streamlines)

@DATASETS.register_module()
class Tracklet_QBXFilter(Dataset):
    def __init__(self, config):
        logger = "Tracklet_Dataset"
        self.config = config

        self.data_dir = config.data_dir
        self.qbx_save_path = config.qbx_save_path
        self.split = config.subset
        self.dataset = config.dataset

        self.samples_per_subject = config.sample_per_subject
        self.samples_per_class = config.samples_per_class
        self.threshold = config.threshold
        self.test_subjects = config.get('test_subjects', None)

        self.max_num_tokens = config.max_num_tokens
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        self.patch_size = 12
        print_log(f"Patch size: {self.patch_size}", logger=logger)

        self.label_to_int_mapping = self.load_json(self.label_to_int_mapping_path)

        if isinstance(self.dataset, str):
            datasets = [self.dataset]
        else:
            datasets = self.dataset
        
        if isinstance(self.split, str):
            splits = [self.split]
        else:
            splits = self.split

        for dataset in datasets:
            if os.path.exists(os.path.join(self.qbx_save_path, dataset)):
                print_log(f"QBX clusters already exist for dataset {dataset}.", logger=logger)
                continue
            print_log(f"Creating QBX clusters for dataset {dataset} and split {splits}", logger=logger)
            create_qbx_cluster_from_scratch_n_save(
                dataset=dataset,
                mapping=self.label_to_int_mapping,
                raw_data_path=self.data_dir,
                save_path=os.path.join(self.qbx_save_path, dataset),
                subset = splits
            )
        
        print_log(f"Starting to load streamlines...", logger=logger)
        self.load_streamlines_mp()
        print_log(f"Total number of streamlines: {len(self.streamlines)}", logger=logger)
        print_log(f"Total number of subjects: {len(np.unique(self.subject_ids))}", logger=logger)



    def load_json(self, path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        return json_data
    
    def load_pickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
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
        return subject_id, streamlines, labels
    
    def load_subject_with_qbx_filter(self, subject_data):
        """
            Loads the data stored by the QBX clustering algorithm in utils.qbx_clustering. 
            This function will only load .pkl files.
            This will remove the streamlines which belong to cluster not obeying the threshold.
        """

        subject_id, qbx_pkl_path, threshold, samples_per_subject, samples_per_class = subject_data
        clusters_L1_L2_L3, streamlines, labels = self.load_pickle(qbx_pkl_path)
        clusters_L3 = np.array(clusters_L1_L2_L3)[:, 2]

        # initialize the cluster_id_to_idxs_mapping
        cluster_id_to_idxs_mapping = {
            'L3': defaultdict(list)
        }

        # populate the cluster_id_to_idxs_mapping
        for idx, cluster_id in enumerate(clusters_L3):
            cluster_id_to_idxs_mapping['L3'][cluster_id].append(idx)
        
        # consider only those streamlines which belong to clusters with more than threshold streamlines
        considered_streamline_idxs = []
        for cluster_id, idxs in cluster_id_to_idxs_mapping['L3'].items():
            if len(idxs) > threshold:
                considered_streamline_idxs.extend(idxs)
        
        # filter the streamlines and labels
        filtered_streamlines = [streamlines[idx] for idx in considered_streamline_idxs]
        filtered_labels = [labels[idx] for idx in considered_streamline_idxs]

        # randomly sample the streamlines per class
        streamlines_per_class = defaultdict(list)
        for idx, label in enumerate(filtered_labels):
            streamlines_per_class[label].append(filtered_streamlines[idx])
        for label in streamlines_per_class:
            if len(streamlines_per_class[label]) > samples_per_class:
                streamlines_per_class[label] = [streamlines_per_class[label][i] 
                                                for i in np.random.choice(len(streamlines_per_class[label]),
                                                                           samples_per_class, replace=False)]
        
        filtered_streamlines = []
        filtered_labels = []
        for label in streamlines_per_class:
            filtered_streamlines.extend(streamlines_per_class[label])
            filtered_labels.extend([label] * len(streamlines_per_class[label]))
        
        # randomly sample the streamlines per subject
        if len(filtered_streamlines) > samples_per_subject:
            random_indices = np.random.choice(len(filtered_streamlines), samples_per_subject, replace=False)
            filtered_streamlines = [filtered_streamlines[i] for i in random_indices]
            filtered_labels = [filtered_labels[i] for i in random_indices]


        return subject_id, filtered_streamlines, filtered_labels

    def load_streamlines_mp(self):
        
        """ 
            Load streamlines from the dataset. 
            This function is parallelized using multiprocessing. 
            This function will only load .trk files. 
        """

        # Handling edge cases
        if isinstance(self.dataset, list):
            if isinstance(self.split, str) or len(self.split) != len(self.dataset):
                raise ValueError("Length of 'dataset' and 'split' should be the same.")
            data_paths = [os.path.join(self.qbx_save_path, dataset, split) 
                          for dataset, split in zip(self.dataset, self.split)]
        else:
            if isinstance(self.split, list):
                raise ValueError("Length of 'dataset' and 'split' should be the same.")
            data_paths = [os.path.join(self.qbx_save_path, self.dataset, self.split)]

        streamlines = []
        subject_ids = []
        labels = []

        subjects = []
        subject_data = []
        for i, data_path in enumerate(data_paths):
            subjects_ = [subject.split('.')[0] for subject in os.listdir(data_path)
                         if os.path.join(data_path, subject).endswith('.pkl')]
            if self.split[i] == 'test' or self.split[i] == 'testset':
                subjects_ = [subject for subject in subjects_ if self.test_subjects is None or subject in self.test_subjects]
            subjects.extend(subjects_)

            # subject_data_ = [(subject, data_path, datasets[i], self.samples_per_subject, self.samples_per_class)
            #                  for subject in subjects_]
            subject_data_ = [(subject, 
                              os.path.join(data_path, f"{subject}.pkl"), 
                              self.threshold, 
                              self.samples_per_subject, 
                              self.samples_per_class)
                             for subject in subjects_]
            subject_data.extend(subject_data_)
        
        

        # Using multiprocessing to parallelize the loading of streamlines
        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.load_subject_with_qbx_filter, subject_data), total=len(subject_data), desc='Loading Streamlines'))

        for subject_id, subject_streamlines, subject_labels in results:
            subject_ids.extend([subject_id] * len(subject_streamlines))
            streamlines.extend(subject_streamlines)
            labels.extend(subject_labels)

        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
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

        # getting tracklet from streamline
        streamline = interpolate_streamline((streamline, 64))
        streamline = self.pc_norm(streamline)
        tracklet = get_tracklet_from_streamline(streamline, apt_param=.4)
        tracklet = np.array([interpolate_streamline((tracklet[i], self.patch_size)) for i in range(len(tracklet))])

        # zero padding
        padding = len(tracklet)
        tracklet = self.zero_padding(tracklet)
        # tracklet = self.repeat_last_padding(tracklet)
        # tracklet = self.repeat_all_padding(tracklet)
        # tracklet = self.pad_with_noise(tracklet)

        # normalizing tracklet with respect to the center points
        tracklet_center = np.median(tracklet, axis=1)
        tracklet = tracklet - tracklet_center[:, None, :]

        # plotting_tracklet(tracklet, file_name="tracklet")
        # plotting_tracklet(tracklet_center[:,None,:], file_name="tracklet_center")
        return np.float32(label),0,(torch.from_numpy(tracklet).float(), torch.from_numpy(tracklet_center).float(), np.float32(padding))
    def __len__(self):
        return len(self.streamlines)

# yaml_file = "cfgs/dataset_configs/tracklet_qbxfilter.yaml"
# config = cfg_from_yaml_file(yaml_file)
# dataset = Tracklet_QBXFilter(config)
# tracklet, tracklet_center, padding = dataset[100]
# x = tracklet_center[:,None,:][:padding]
# print(tracklet.shape)
# print(x.shape)
# plotting_tracklet(x)