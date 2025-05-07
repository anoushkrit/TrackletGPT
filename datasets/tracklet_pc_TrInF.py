# Standard libraries
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import datetime
import time
import gc
import pickle
import copy
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count, set_start_method

# Third-party libraries
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from blist import blist
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

from memory_profiler import profile
from dipy.tracking.distances import approx_polygon_track as apt

# Project-specific imports
from utils.config import *
from utils.logger import *
from utils.tracklet_ops import get_tracklet_from_streamline
from utils.streamline_ops import *
from utils.qbx_clustering import create_qbx_cluster_from_scratch_n_save
from datasets import data_transforms_tracklet
from .build import DATASETS

data_transforms = transforms.Compose([
    # data_transforms_tracklet.StreamlineFlip(),
    data_transforms_tracklet.StreamlineRotate(),
])
#%%
def load_data(args):
    """
    Load data from the pickle file and create the point cloud of the cluster
    Also, randomly select sample_per_class_per_subject from each class
    Args:
        args: tuple containing (pth, pts_per_PC_sample, streamline_threshold_per_cluster, sample_per_class_per_subject, test)
    Returns:
        streamlines: list of streamlines
        labels: list of labels
        cluster_id: list of cluster_id
        pc_cluster: numpy array of point cloud of cluster
    """
    # current, peak = tracemalloc.get_traced_memory()
    # print(f'current memory usage at start of load_data is {current / 10**6}MB; Peak was {peak / 10**6}MB')
    pth, pts_per_PC_sample, streamline_threshold_per_cluster, sample_per_class_per_subject, test, subject = args
    level = 3
    subject_data = {}
    subject_id = os.path.basename(pth).split('.')[0]

    # reading the data from the pickle file saved by create_qbx_cluster_from_scratch_n_save
    with open(pth, 'rb') as f:
        data = pickle.load(f)
    data_x = np.array(data[0])
    subject_data['streamlines'] = blist(data[1])
    subject_data['label'] = np.array(data[2])
    subject_data['cluster_ids_l1'] = data_x[:,level-2]
    subject_data['cluster_ids_l2'] = data_x[:,level-1]
    subject_data['cluster_ids_l3'] = data_x[:,level]
    del data_x
    del data
    gc.collect()

    # tic = time.time()
    # getting cluster from go back 3L method cluster formation of qbx
    # print(streamline_threshold_per_cluster)
    cluster, pc_cluster_labels = create_clusters_QBX_3L_per_subject((subject_data, subject_id, streamline_threshold_per_cluster, test))
    # print(cluster[0])
    # toc = time.time()
    # print('step 1 done', toc-tic)
    tic = time.time()
    del subject_data['cluster_ids_l1']
    del subject_data['cluster_ids_l2']
    del subject_data['cluster_ids_l3']
    gc.collect()

    # creating point cloud of cluster from streamlines belonging to same cluster
    pc_cluster = np.zeros((len(cluster), pts_per_PC_sample, 3))
    for i in range(len(cluster)):
        pc_cluster[i] = concat_streamlines((subject_data, cluster[i], False, pts_per_PC_sample))
    
    toc = time.time()
    # print('step 2 done', toc-tic)
    # tic = time.time()
    # counting the number of streamlines in left after cluster formation
    tot_remaining_streamlines = 0
    for i in range(len(cluster)):
        tot_remaining_streamlines += len(cluster[i])

    # creating a array of tuple having(streamline, cluster_id, label of streamline)
    label_to_streamline_cluster_id = {}
    for i in range(len(cluster)):
        for j in cluster[i]:
            j = int(j.split('_')[1])
            if subject_data['label'][j] not in label_to_streamline_cluster_id:
                label_to_streamline_cluster_id[subject_data['label'][j]] = []
            label_to_streamline_cluster_id[subject_data['label'][j]].append((j, i))

    # toc = time.time()
    # print('step 3 done', toc-tic)
    # tic = time.time()
    # converting list to numpy array
    for label in label_to_streamline_cluster_id.keys():
        label_to_streamline_cluster_id[label] = np.array(label_to_streamline_cluster_id[label], dtype=np.int32)

    # randomly selecting sample_per_class from each label
    if not test:
        all_labels = list(label_to_streamline_cluster_id.keys())
        for label in all_labels:
            if len(label_to_streamline_cluster_id[label]) > sample_per_class_per_subject:
                label_to_streamline_cluster_id[label] = label_to_streamline_cluster_id[label][
                    np.random.choice(len(label_to_streamline_cluster_id[label]), sample_per_class_per_subject, replace=False)
                ]
    
    # toc = time.time()
    # print('step 4 done', toc-tic)
    # tic = time.time()
    streamlines = []
    labels = []
    cluster_id = []
    for label in label_to_streamline_cluster_id.keys():
        for i in range(len(label_to_streamline_cluster_id[label])):
            streamlines.append(subject_data['streamlines'][label_to_streamline_cluster_id[label][i][0]])
            labels.append(label)
            cluster_id.append(label_to_streamline_cluster_id[label][i][1])

    # toc = time.time()
    # print('step 5 done', toc-tic)
    del subject_data

    return streamlines, labels, cluster_id, pc_cluster, subject

def concat_streamlines(args):
    data, indices, use_resampled_streamline, pts_per_sample = args
    subject_id = indices[0].split('_')[0]      
    indices = [int(i.split('_')[1]) for i in indices]

    club_streamlines = [data['streamlines'][idx] for idx in indices]
    club_streamlines = np.concatenate(club_streamlines, axis=0)
    
    rep_count = 1
    if len(club_streamlines) < pts_per_sample:
        rep_count = int(np.ceil(pts_per_sample/len(club_streamlines)))
    club_streamlines = np.tile(club_streamlines, (rep_count, 1))

    if len(club_streamlines) > pts_per_sample:
        club_streamlines = club_streamlines[
            np.random.choice(len(club_streamlines), pts_per_sample, replace=False)
        ]
    elif len(club_streamlines) < pts_per_sample:
        print('why')
    # club_streamlines = club_streamlines[
    #     np.random.choice(len(club_streamlines), pts_per_sample, replace=False)
    # ]
    # # club_streamlines = pc_norm(club_streamlines)
    # original_club_streamlines = copy.deepcopy(club_streamlines)
    # while len(club_streamlines)+len(original_club_streamlines) < pts_per_sample:
    #     club_streamlines = np.concatenate((club_streamlines,original_club_streamlines))

    
    return club_streamlines
@DATASETS.register_module()
class Tracklet_pc_TrInF(Dataset):

    def __init__(self,config):
        logger = "Tracklet_with_Point_Cloud_Dataset_TrInF"
        self.config = config

        self.data_dir = config.data_dir
        self.qbx_dir = config.qbx_save_dir
        self.dataset = config.dataset
        self.split = config.subset
        self.num_workers = config.num_workers

        self.samples_per_class_per_subject = config.samples_per_class_per_subject
        self.pts_per_PC_sample = config.pts_per_PC_sample
        self.data_transformation = config.data_transformation
        self.test_subjects = config.get('test_subjects', None)

        self.max_num_tokens = config.max_num_tokens
        self.num_tokens_for_pc = config.num_tokens_for_pc
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        self.patch_size = 32

        self.streamline_threshold_per_cluster = config.streamline_threshold_per_cluster
        self.label_to_int_mapping_path = config.label_to_int_mapping_path

        self.label_to_int_mapping = self.load_json(self.label_to_int_mapping_path)

        # check if qbx_data is prepared of not
        if not os.path.exists(self.qbx_dir):
            print_log(f'[DATASET] Data path {self.data_path} does not exist', logger=logger)
            print_log(f'[DATASET] Creating QBX clusters', logger=logger)
            create_qbx_cluster_from_scratch_n_save(
                dataset='TractoInferno',
                mapping=self.label_to_int_mapping,
                save_path=self.qbx_dir,
                raw_data_path=self.data_dir,
                split=[self.split]
            )
        
        self.load_data_tracklet_pc_TrInF()

    def load_json(self, path):
        with open(path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def load_data_tracklet_pc_TrInF(self):

        data_dir = os.path.join(self.qbx_dir, self.dataset, self.split)
        subjects = [subject for subject in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, subject))]
        # print(subjects)
        if self.split in ['testset', 'test'] and type(self.test_subjects) == list: # using only the given subjects
            subjects = [subject for subject in subjects if subject.split('.')[0] in self.test_subjects]
        # print(subjects)
        
        inp = []
        for subject in subjects:
            inp.append((os.path.join(data_dir,subject), 
                        self.pts_per_PC_sample,
                        self.streamline_threshold_per_cluster,
                        self.samples_per_class_per_subject,
                        self.split in ['testset', 'test'],
                        subject.split('.')[0]
                        ))
        with Pool(self.num_workers) as p:
            # data = p.map(load_data, inp)
            data = list(tqdm(p.imap(load_data, inp), total=len(inp)))
        # return streamlines, labels, cluster_id, pc_cluster
        self.streamlines = []
        self.labels = []
        self.subjects = []
        self.cluster_id = []
        self.pc_cluster = {}

        for streamlines, labels, cluster_id, pc_cluster, subject in data:
            self.streamlines.extend(streamlines)
            self.labels.extend(labels)
            self.cluster_id.extend(cluster_id)
            self.subjects.extend([subject]*len(streamlines))
            self.pc_cluster[subject] = pc_cluster
        
        # for i in tqdm(range(len(inp))):
        #     streamlines, labels, cluster_id, pc_cluster = load_data(inp[i])
        #     self.streamlines.extend(streamlines)
        #     self.labels.extend(labels)
        #     self.cluster_id.extend(cluster_id)
        #     self.subjects.extend([subjects[i]]*len(streamlines))
        #     self.pc_cluster[subjects[i]] = pc_cluster
        # self.pc_cluster = np.array(self.pc_cluster)



        # for i in range(len(data)):
        #     self.pc_cluster.extend(data[i][0])
        #     self.pc_cluster_labels.extend(data[i][1])
        #     self.cluster.extend(data[i][2])
        #     self.cluster_labels.extend(data[i][3])
        #     self.streamline_labels[subjects[i].split('.')[0]] = data[i][5]
        #     if self.split == 'testset':
        #         self.streamlines[subjects[i].split('.')[0]] = data[i][4]
        

                
        # # if self.split not in ['testset', 'test']:
        # #     with open(os.path.join(self.save_path, f'{self.split}.pkl'), 'wb') as f:
        # #         pickle.dump((self.pc_cluster, self.pc_cluster_labels, self.cluster, self.cluster_labels, self.streamline_labels), f)
        # else:
    
    def zero_padding(self, tracklet):
        """ Zero pads the tracklet to the maximum number of tokens. """
        if len(tracklet) < self.max_num_tokens:
            padding = self.max_num_tokens - len(tracklet)
            tracklet = np.concatenate((tracklet, np.zeros((padding, self.patch_size, 3))), axis=0)
        return tracklet

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __getitem__(self, index):
        streamline = self.streamlines[index]
        label = self.label_to_int_mapping[self.labels[index]]
        cluster_id = int(self.cluster_id[index])
        subject = self.subjects[index]
        pc = self.pc_cluster[subject][cluster_id]

        # interpolate and normalize streamline+pc
        streamline = interpolate_streamline((streamline, 64))
        pc = np.concatenate((streamline, pc), axis=0)
        pc = self.pc_norm(pc)
        if self.data_transformation and self.split in ['train', 'trainset']:
            pc = data_transforms(pc)
        streamline = pc[:-self.pts_per_PC_sample]
        pc = pc[-self.pts_per_PC_sample:]

        # getting tracklet from the streamline
        tracklet = get_tracklet_from_streamline(streamline, apt_param=0.8)
        tracklet = np.array([interpolate_streamline((tracklet[i], self.patch_size)) for i in range(len(tracklet))])

        # zero padding the tracklet
        padding = np.full((self.max_num_tokens), True)
        padding[-self.num_tokens_for_pc:] = False
        padding[:len(tracklet)] = False
        tracklet = self.zero_padding(tracklet)

        # center points 
        tracklet_center = np.mean(tracklet, axis=1)
        tracklet = tracklet - tracklet_center[:, None, :]

        data = {
            'tracklet': torch.tensor(tracklet, dtype=torch.float32),
            'tracklet_center': torch.tensor(tracklet_center, dtype=torch.float32),
            'pc': torch.tensor(pc, dtype=torch.float32),
            'padding': torch.tensor(padding, dtype=torch.bool),
        }
        label = torch.tensor(label, dtype=torch.int32)
        
        return label, 0, data
        # return self.labels[index], 0, data


    def __len__(self):
        return len(self.streamlines)
        
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

if __name__ == '__main__':
    config_path = 'cfgs/dataset_configs/tracklet_pc_TrInF.yaml'
    config = cfg_from_yaml_file(config_path)
    dataset = Tracklet_pc_TrInF(config)

    # print(len(dataset.streamlines), len(dataset.labels), len(dataset.cluster_id), len(dataset.pc_cluster))
    # subject_cluster = {}
    # for i in range(len(dataset.cluster_id)):
    #     if dataset.subjects[i] not in subject_cluster:
    #         subject_cluster[dataset.subjects[i]] = 0
    #     subject_cluster[dataset.subjects[i]] = max(subject_cluster[dataset.subjects[i]], dataset.cluster_id[i])
    
    # for subject in subject_cluster.keys():
    #     print(f'{subject}: {subject_cluster[subject]}, {len(dataset.pc_cluster[subject])}')

    # for i in range(200):
    #     idx = np.random.randint(len(dataset))
    #     label, _, data = dataset[idx]
    #     tracklet_center = data['tracklet_center']
    #     pc = data['pc']
    #     plot_point_cloud(pc.numpy(), red_points=tracklet_center.numpy(), filename=f'streamline_plots/{label}_{i}', title=f'Label: {label}')

    subject_to_label_to_streamlines = defaultdict(lambda: defaultdict(list))
    os.makedirs('streamline_plots', exist_ok=True)
    label_to_count = {}
    for i in range(len(dataset.streamlines)):
        subject_to_label_to_streamlines[dataset.subjects[i]][dataset.labels[i]].append(dataset.streamlines[i])
    for subject in subject_to_label_to_streamlines.keys():
        for label in subject_to_label_to_streamlines[subject].keys():
            # print(f'{subject}: {label} - {len(subject_to_label_to_streamlines[subject][label])}')
            if label not in label_to_count:
                label_to_count[label] = 0
            label_to_count[label] += len(subject_to_label_to_streamlines[subject][label])
            subject_to_label_to_streamlines[subject][label] = np.concatenate(subject_to_label_to_streamlines[subject][label], axis=0)
            subject_to_label_to_streamlines[subject][label] = subject_to_label_to_streamlines[subject][label][
                np.random.choice(len(subject_to_label_to_streamlines[subject][label]), 2048, replace=True)
            ]
            # print(subject_to_label_to_streamlines[subject][label].shape)
            # plot_point_cloud(subject_to_label_to_streamlines[subject][label], filename=f'streamline_plots/{subject}_{label}', title=f'Subject: {subject}, Label: {label}')
        # break
    print(label_to_count)
