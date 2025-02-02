# Standard libraries
import sys
import os
import json
import time
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
    for i, track in enumerate(tracklet):
        track = list(zip(*track))  # Transpose to get x, y, z components
        # if i == 0:
        #     fig.add_trace(go.Scatter3d(
        #         x=track[0],
        #         y=track[1],
        #         z=track[2],
        #         mode='markers',
        #     ))
        #     continue
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
class Tracklet_Sentence(Dataset):
    def __init__(self, config):
        logger = "Tracklet_Sentence_Dataset"
        self.config = config

        self.data_dir = config.data_dir
        self.qbx_save_path = config.qbx_save_path
        self.split = config.subset
        self.dataset = config.dataset
        self.num_workers = config.num_workers

        self.samples_per_subject = config.sample_per_subject
        self.samples_per_class = config.samples_per_class
        self.threshold = config.threshold
        self.considered_neighbours_count = config.considered_neighbours_count
        self.test_subjects = config.get('test_subjects', None)

        self.max_num_tokens = config.max_num_tokens
        self.EOS_token = config.EOS_token
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        self.patch_size = 12
        self.neighbouring_streamline_count = config.max_total_neighbours_count
        print_log(f"Patch size: {self.patch_size}", logger=logger)

        # Check if the threshold is greater than the considered neighbours count
        if self.threshold < self.considered_neighbours_count:
            raise ValueError("Threshold should be greater than the considered neighbours count.")
        
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
            for split in splits:
                if os.path.exists(os.path.join(self.qbx_save_path, dataset, split)):
                    print_log(f"QBX clusters already exist for dataset {dataset} and split {split}", logger=logger)
                    continue
                print_log(f"Creating QBX clusters for dataset {dataset} and split {split}", logger=logger)
                create_qbx_cluster_from_scratch_n_save(
                    dataset=dataset,
                    mapping=self.label_to_int_mapping,
                    raw_data_path=self.data_dir,
                    save_path=os.path.join(self.qbx_save_path, dataset),
                    subset = [split]
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
    
    def load_subject_with_qbx_cluster(self, subject_data):
        """
            Loads the data stored by the QBX clustering algorithm in utils.qbx_clustering. 
            This function will only load .pkl files.
            This will remove the streamlines which belong to cluster not obeying the threshold.
        """

        subject_id, qbx_pkl_path, threshold, samples_per_subject, samples_per_class, neighbouring_streamline_count = subject_data
        clusters_L1_L2_L3, streamlines, labels = self.load_pickle(qbx_pkl_path)
        clusters_L3 = np.array(clusters_L1_L2_L3)[:, 2]
        del clusters_L1_L2_L3

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
        
        # randomly sample samples_per_class streamlines for each label/class
        label_to_streamline_idxs = defaultdict(list)
        for idx in considered_streamline_idxs:
            label = labels[idx]
            label_to_streamline_idxs[label].append(idx)
        
        considered_streamline_idxs = []
        for label, idxs in label_to_streamline_idxs.items():
            if len(idxs) >= samples_per_class:
                idxs = np.random.choice(idxs, samples_per_class, replace=False)
            considered_streamline_idxs.extend(idxs)
        del label_to_streamline_idxs
        
        # randomly sample samples_per_subject streamlines for the subject
        if len(considered_streamline_idxs) >= samples_per_subject:
            considered_streamline_idxs = np.random.choice(considered_streamline_idxs, samples_per_subject, replace=False)
        
        # get the streamline neighbours
        streamline_neighbours = []
        for idx in considered_streamline_idxs:
            cluster_id = clusters_L3[idx]
            streamline_idxs_in_cluster = cluster_id_to_idxs_mapping['L3'][cluster_id]
            if len(streamline_idxs_in_cluster) > self.neighbouring_streamline_count:
                streamline_idxs_in_cluster = np.random.choice(streamline_idxs_in_cluster, neighbouring_streamline_count, replace=False)
            
            streamline_in_cluster = [streamlines[idx] for idx in streamline_idxs_in_cluster]
            streamline_neighbours.append(streamline_in_cluster)

        # get the streamlines and labels
        streamlines = [streamlines[idx] for idx in considered_streamline_idxs]
        labels = [labels[idx] for idx in considered_streamline_idxs]

        return subject_id, streamlines, labels, streamline_neighbours


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
        streamline_neighbours = []

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
                              self.samples_per_class,
                              self.neighbouring_streamline_count)
                             for subject in subjects_]
            subject_data.extend(subject_data_)
        
        

        # Using multiprocessing to parallelize the loading of streamlines
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.load_subject_with_qbx_cluster, subject_data), total=len(subject_data), desc='Loading Streamlines'))

        for subject_id, subject_streamlines, subject_labels, subject_streamline_neighbours in results:
            subject_ids.extend([subject_id] * len(subject_streamlines))
            streamlines.extend(subject_streamlines)
            labels.extend(subject_labels)
            streamline_neighbours.extend(subject_streamline_neighbours)

        self.streamlines = streamlines
        self.subject_ids = np.array(subject_ids)
        self.labels = labels
        self.streamline_neighbours = streamline_neighbours

    def zero_padding(self, tracklet):
        """ Zero pads the tracklet to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            tracklet = np.concatenate((tracklet, np.zeros((padding, self.patch_size, 3))), axis=0)
        else:
            print('problem')
        #     mod_tracklet = np.array(tracklet[:self.max_num_tokens])
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
        """
            Returns:
                group of streamlines
                label of main streamline
                padding (number of tokens to consider)
                index where EOS/[SEP] should be placed
        """

        streamline = self.streamlines[index]
        label = self.labels[index]
        label = self.label_to_int_mapping[label] # converting label to integer
        neighbour_streamlines = self.streamline_neighbours[index]

        # randomly sample neighbouring streamlines
        if len(neighbour_streamlines) > self.considered_neighbours_count:
            neighbour_streamlines = [
                neighbour_streamlines[i] for i in np.random.choice(len(neighbour_streamlines), self.considered_neighbours_count, replace=False)
            ]
        # grouping main streamline with neighbours to do normalization of PC
        grouped_streamlines = [streamline] + neighbour_streamlines
        grouped_streamlines = [np.array(streamline) for streamline in grouped_streamlines]
        grouped_streamlines = np.concatenate(grouped_streamlines, axis=0)
        grouped_streamlines = self.pc_norm(grouped_streamlines)

        # getting back streamline and neighbouring streamlines
        streamline = grouped_streamlines[:len(streamline)]
        counted_pts = len(streamline)
        for i in range(len(neighbour_streamlines)):
            neighbour_streamlines[i] = grouped_streamlines[
                counted_pts: counted_pts + len(neighbour_streamlines[i])
            ]
            counted_pts += len(neighbour_streamlines[i])
        
        # getting the tracklet
        streamline = interpolate_streamline((streamline, 64))
        tracklet = get_tracklet_from_streamline(streamline, apt_param=0.4)
        tracklet = np.array([interpolate_streamline((tracklet[i], self.patch_size)) for i in range(len(tracklet))])

        neighbour_streamlines = [interpolate_streamline((neighbour_streamline, 64))
                                 for neighbour_streamline in neighbour_streamlines]
        neighbour_tracklets = [get_tracklet_from_streamline(neighbour_streamline, apt_param=0.4)
                                for neighbour_streamline in neighbour_streamlines]
        neighbour_tracklets = [
            np.array([interpolate_streamline((neighbour_tracklet[i], self.patch_size)) for i in range(len(neighbour_tracklet))])
            for neighbour_tracklet in neighbour_tracklets
        ]
        
        # initializing the sentence
        EOS_idxs = np.zeros(len(neighbour_tracklets), dtype=np.int32)
        sentence_len = len(tracklet)
        for i in range(len(neighbour_tracklets)):
            sentence_len += len(neighbour_tracklets[i])
            if self.EOS_token: sentence_len += 1
        sentence = np.zeros((sentence_len, self.patch_size, 3))

        # adding the main tracklet
        sentence[:len(tracklet)] = tracklet.copy()
        EOS_idxs[0] = len(tracklet)-1
        if self.EOS_token:
            sentence[len(tracklet)] = np.zeros((self.patch_size, 3))
            EOS_idxs[0] = len(tracklet)

        # adding the neighbouring tracklets
        for i in range(len(neighbour_tracklets)):
            sentence[1+EOS_idxs[i]:1+EOS_idxs[i]+len(neighbour_tracklets[i])] = neighbour_tracklets[i].copy()
            if i+1 < len(neighbour_tracklets):
                if self.EOS_token:
                    sentence[1+EOS_idxs[i]+len(neighbour_tracklets[i])] = np.zeros((self.patch_size, 3))
                    EOS_idxs[i+1] = 1 + EOS_idxs[i] + len(neighbour_tracklets[i])
                else:
                    EOS_idxs[i+1] = EOS_idxs[i] + len(neighbour_tracklets[i])
        
        for i in range(len(sentence)-1):
            if np.all(sentence[i]==0) or np.all(sentence[i+1]==0):
                print('zero')
            if np.all(sentence[i]==sentence[i+1]):
                print('same')

        # padding the sentence
        padding = len(sentence)
        sentence = self.zero_padding(sentence)

        # getting centroids of the sentence and normalizaing them
        sentence_centroids = np.median(sentence, axis=1)
        sentence = sentence - sentence_centroids[:, None, :]

        data = {
            'tokens': torch.from_numpy(sentence).float(),
            'center': torch.from_numpy(sentence_centroids).float(),
            'padding': np.float32(padding),
            'EOS_idxs': torch.tensor(EOS_idxs).float()
        }
        if self.split[0] == 'test' or self.split[0] == 'testset':
            return np.float32(label),[str(self.subject_ids[index]), str(self.streamlines[index])], data
        return np.float32(label), 0, data
        

    def __len__(self):
        return len(self.streamlines)

# yaml_file = "cfgs/dataset_configs/tracklet_qbxfilter.yaml"
# config = cfg_from_yaml_file(yaml_file)
# dataset = Tracklet_Sentence(config)
# time.sleep(100)
# for i in range(20):
#     idx = np.random.randint(len(dataset))
#     label, _, (sentence, sentence_centroids, padding, SEP_idxs) = dataset[idx]

#     plotting_tracklet(sentence_centroids.unsqueeze(1).numpy(), file_name=f"streamline_plots/{label.item()}", title=f"Label: {label.item()}")

# max_length = 0
# for i in range(10000):
#     label, _, (sentence, sentence_centroids, padding, SEP_idxs) = dataset[i]
#     max_length = max(max_length, len(sentence))
# print(max_length)