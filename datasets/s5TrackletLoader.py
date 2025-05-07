#%%

import pickle
import numpy as np 
import os
import sys 
sys.path.append('../')
import scipy.interpolate
import torch.utils.data as data
from utils.streamline_ops import bicubic_interpolate, interpolate_streamlines, get_named_colors
import torch
import plotly.graph_objects as go
from utils import streamline_ops
import scipy
import numpy as np
from dipy.tracking.distances import approx_polygon_track as apt
from dipy.tracking.streamline import set_number_of_points
# from scipy.interpolate import make_interp_spline
# from scipy import interpolate

#%%

def solid_angle_of_cone(p1, p2):
    v = p2 - p1
    v_mag = np.linalg.norm(v)
    v_hat = v / v_mag
    theta = np.arccos(v_hat[2]) # only calculating the theta angle with the z axis and the
    # z angle of the line segment
    solid_angle = 2 * np.pi * (1 - np.cos(theta))
    return solid_angle 

def scipy_interpolate_2D(x, y, degree=3):
    spline = scipy.interpolate.make_interp_spline(x, y, k=degree)
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = spline(x_new)
    return x_new, y_new

def scipy_interpolate_3D(x, y, z, num_points, s=0, k=3):
    tck, u = scipy.interpolate.splprep([x, y, z], s=s, k=k)
    normalize=None
    if normalize: 
        u_new = np.linspace(0, 1, num=num_points)
    else:
        u_new = np.linspace(x.min(), x.max(), num=num_points)
    # x_new = np.linspace(x.min(), x.max(), num_points)
    x_new, y_new, z_new = scipy.interpolate.splev(u_new, tck)
    return x_new, y_new, z_new
def visualize_tracks(tracks, points, save=False):
    fig = go.Figure()
    named_colors = get_named_colors()
    static_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    if points is not None:
        for p, point in enumerate(points): 
            x_values = point[:, 0]  # X-axis values
            y_values = point[:, 1]  # Y-axis values
            z_values = point[:, 2]  # Z-axis values
            fig.add_trace(go.Scatter3d(x=x_values, 
                                    y=y_values, 
                                    z=z_values,
                                    mode='markers', 
                                    marker=dict(size=2, color=static_colors[p% len(static_colors)]), 
                                    name='original_points'))

    if tracks is not None:
        for i, track in enumerate(tracks): 
            x_values = track[:, 0]  # X-axis values
            y_values = track[:, 1]  # Y-axis values
            z_values = track[:, 2]  # Z-axis values

            color_index = i % len(named_colors) + 10  # Calculate the color index based on the current iteration
            fig.add_trace(go.Scatter3d(x=x_values, 
                                        y=y_values, 
                                        z=z_values,
                                        mode='lines', 
                                        line=dict(color=named_colors[color_index], width=4), 
                                        name=f'track_{i}'))

    if tracks is None and points is None:
        raise ValueError("Both Tracks and Points are None")
    

    fig.update_layout(
        title='3D Plot of Streamlines',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        legend_title='Lines'
    )
    if save is False: 
        fig.show()
    else: 
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.write_html(f"plots/track_{np.random.randint(0, 1000)}.html")

def find_matching_indices(arr1, arr2):
    """where arr1 is the smaller 
    and arr2 is the larger array"""
    match_array = [] 
    for i in range(arr1.shape[0]):
        matches = np.where(np.all(np.isclose(arr1[i], arr2, atol=1e-3), axis=1))[0]
        if not matches.size == 0:
            match_array.append(matches[0])
    return match_array

def bspline_split(streamline, interp_points=[64, True]): 
    interp_streamline = streamline_ops.interpolate_streamlines(streamline.reshape(1,15,3),interp_points[0])[0]
    indices = [np.where((interp_streamline == streamline[i]).all(axis=1))[0][0] for i in range(len(streamline))]
    if interp_points[1]:
        return indices, interp_streamline
    else: 
        return indices

class s5TrackletLoader(data.Dataset):
    def __init__(self, input_path, batch_mode = False, split='train', interpolate=[False, 40]):
        split = "train"
        with open(os.path.join(input_path, '{}.pickle'.format(split)), 'rb') as file:
            data_dict = pickle.load(file)

        self.feat = data_dict['feat']
        self.labels = data_dict['label']
        self.label_names = data_dict['label_name']
        self.subject_ids = data_dict['subject_id']
        self.augment = False
        self.zero_pad, self.interpolate_tracklets, = False, True
        self.batch_mode = batch_mode
        del data_dict

        if interpolate[0]:
            self.labels, self.subject_ids , self.feat, _ , self.indices= bicubic_interpolate(labels = self.labels,
                                                        subject_ids= self.subject_ids ,
                                                        features = self.feat,
                                                        ras_feature=self.feat,
                                                        indices = self.indices,
                                                        num_point_per_fiber=interpolate[1])
             
    def batch_bspline_split(self, streamlines):
        return indices

    def conical_split(self, streamline):
        
        indices = []
        return indices
    
    def create_conical_split_patches(streamline, slope_arr, threshold):
        patches = []
        current_patch = [streamline[0]]  # Start with the first point
        is_above_threshold = abs(slope_arr[0]) > threshold  # Determine the initial threshold condition
        
        for i in range(1, len(slope_arr) + 1):
            current_slope = abs(slope_arr[i - 1])
            # Check if the slope crosses the threshold boundary (either above or below)
            if (is_above_threshold and current_slope <= threshold) or (not is_above_threshold and current_slope > threshold):
                patches.append(np.array(current_patch))
                current_patch = []
                is_above_threshold = current_slope > threshold  # Update the condition
            
            current_patch.append(streamline[i])
        if current_patch:
            patches.append(np.array(current_patch))
        
        return patches

    def bend_split(self, streamline):
        indices = []
        return indices

    def gradient_split(self, streamline):
        indices = []
        return indices
        
    def get_tracklets(self, indices, streamline, overlap = 0):
        tracklets = [] 
        pad=False 
        interpolate_tracklets = True

        tracklet_indices = []   # NOTE: These are the tracklet indices which were originally in the streamline and 
        # are the ones which will get modified during augmentation and overlap of streamlines

        for i in range(len(indices)-1) :
            if overlap is None: 
                tracklets.append(streamline[indices[i]:indices[i+1]])
            else: 
                if indices[i] == 0: # streamline start case
                    # + 1 is done to handle indexing in python to add all the indices
                    tracklet_indices.append(str(streamline[indices[i]:indices[i+1]  + 1 + overlap]))
                    tracklets.append(streamline[indices[i]:indices[i+1]  + 1 + overlap])
                elif indices[i] == len(streamline): # streamline termination case 
                    tracklet_indices.append(str(streamline[indices[i] - overlap: indices[i+1] ]))
                    tracklets.append(streamline[indices[i] - overlap: indices[i+1] ])
                else:
                    tracklet_indices.append(str(streamline[indices[i] - overlap : indices[i+1] + 1  + overlap]))
                    tracklets.append(streamline[indices[i] - overlap: indices[i+1] +1 + overlap])
        # Clean the tracklets based on the empty arrays in the list of tracklets
        tracklets = self.remove_empty_tracks(tracklets)
        return tracklets, tracklet_indices

    def remove_empty_tracks(self, tracklets):
        return [tracklet for tracklet in tracklets if tracklet.shape[0] != 0]

    def zero_pad_tracklets(self, tracklets, max_length):
        """pad tracklets with zeros"""
        # NOTE: Doesn't work that good, sort of a make shift solution 
        for i in range(len(tracklets)):
            if len(tracklets[i]) < max_length:
                padding = np.zeros((max_length - len(tracklets[i]), 3))
                tracklets[i] = np.concatenate((tracklets[i], padding), axis=0)
        return tracklets
    

    def augment_tracklets(self, tracklets):
        # TODO: bidirectionality in tracklets OR streamline
        for i in range(len(tracklets)):
            tracklet = tracklets[i]
            rotation_angle = np.random.uniform(0, 2*np.pi)
            rotation_axis = np.random.rand(3)
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation_matrix = scipy.spatial.transform.Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()
            tracklet = np.dot(tracklet, rotation_matrix)
            noise = np.random.normal(0, 0.1, tracklet.shape)
            tracklet += noise
            tracklets[i] = tracklet

        return tracklets
    
    def linguistic_tokens():
        token = 0
        # TODO: <EOS> token 
        # TODO: <SEP> token
        # TODO: <CLS> token
        # TODO: <PAD> token
        # TODO: <UNK> token
        # TODO: <MASK> token
        return token
    
    def augment_sequences(self, sequence):
        sequences = []
        sequences[0] = sequence
        sequences[1] = torch.flip(sequence, [0])

        # TODO: use apply_transform in DIPY for all the augmentations
        return sequences
    
    def undersample_streamline(self, streamline):
        """return streamline of (15,3) or (12,3)"""
        #TODO: use approx polygon track for this 
        return streamline
    
    def tracklet_embedding(self, tracklets, embedding_model, tembed_dim):
        # tracklet_embeddings = torch.zeros(len(tracklets), tembed_dim)
        embedding_model.eval()
        # This will take 15 tracklets inside this in the form of a batch 
        # (B, num_points_per_tracklet, 3) -> (B, tembed_dim) 
        tracklet_embeddings = embedding_model.predict(tracklets)

        return tracklet_embeddings

    def __len__(self):
        if self.augment is False:
            return len(self.feat)

    def __getitem__(self, idx, overlap = 2):
        streamline = self.feat[idx]
        label = self.labels[idx]
        subject = self.subject_ids[idx]
        label_name = self.label_names[idx]
        pad_sequence = False
        crucial_points = apt(streamline, 0.40)
        indices = find_matching_indices(crucial_points , streamline)
        tracklets, tracklet_indices = self.get_tracklets(indices, streamline, overlap = overlap)

        if self.zero_pad == True:
            # For zero padding tracklets
            tracklets = self.zero_pad_tracklets(tracklets, 5)
        # print(tracklets)
        # print(len(tracklets))
        # [print(track.shape) for track in tracklets]
        if self.interpolate_tracklets == True: 
            tracklets = interpolate_streamlines(tracklets, pps = 5) #NOTE: here points per tracklet is set to 5 

        if self.augment == True:
            tracklets = self.augment_tracklets(tracklets)

        #TODO: Pad sequence, use this after batch loading
        #BUG: Can we do batch loading in the dataloader?
        if pad_sequence == True: 
            from torch.nn.utils.rnn import pad_sequence
            batch_tracklets = pad_sequence(batch_tracklets, batch_first=True)
            #NOTE: 4D array (B, num_tracklets, nppt, 3)
            return batch_tracklets, batch_labels, batch_label_names, batch_subjects
        else: 
            return tracklets , label , label_name, subject
        
        # TODO: Augmentation per epoch would be better



#%%

if __name__ == '__main__':
    # config_file_path = "/neuro/tracto/TractoBERT/cfgs/dataset_configs/streamline_patch.yaml"
    input_path = "/neuro/tractcloud-data/TrainData_800clu800ol/"
    s5 = s5TrackletLoader(input_path = input_path, split='train', interpolate=[False, 40])
    s5[0]

    streamline = s5.feat[1298]
    interp_points = [64, True]
    interp_streamline = streamline_ops.interpolate_streamlines(streamline.reshape(1,15,3),interp_points[0])[0]
    crucial_points = apt(interp_streamline, 0.40)
    apt_interp_streamline = streamline_ops.interpolate_streamlines(crucial_points.reshape(1,len(crucial_points),3), 100)[0]
    visualize_tracks([interp_streamline, apt_interp_streamline ], [ crucial_points], save=True)
    indices = find_matching_indices(crucial_points,interp_streamline)
    print(indices)
    tracklets, _= s5.get_tracklets(indices, interp_streamline, overlap=2)

    visualize_tracks(tracklets, None, save=False)
    visualize_tracks([interp_streamline[indices[1] + 1 - 2 : indices[2] + 1 + 2 ], 
                        interp_streamline[indices[2] + 1 - 2 : indices[3] + 1 + 2 ]], 
                        [streamline], save=False)
# %%
