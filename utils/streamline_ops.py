import os 
import numpy as np
from pathlib import Path
from dipy.io.streamline import load_trk
from nibabel.streamlines.array_sequence import ArraySequence
import plotly.graph_objects as go
from dipy.segment.bundles import select_random_set_of_streamlines
from dipy.segment.clustering import QuickBundlesX, QuickBundles
from dipy.segment.metric import (
    AveragePointwiseEuclideanMetric, SumPointwiseEuclideanMetric, CosineMetric)
from dipy.segment.featurespeed import VectorOfEndpointsFeature, CenterOfMassFeature
from dipy.tracking.distances import approx_polygon_track as apt
# https://docs.dipy.org/stable/reference/dipy.segment.html#centerofmassfeature
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.interpolate import CubicSpline
# import open3d as o3d
# from dipy.viz import window, actor
#%% 

def visualise_bundle(viz_tracks, name, interactive, separate=True ):
    scene = window.Scene()
    scene.SetBackground(0,0,0)
    if separate: 
        viz_tracks = [np.asarray(track) for track in viz_tracks]
        for track in viz_tracks:
            scene.add(actor.line([track], colors= colormap.line_colors, fake_tube=True, linewidth=2))
    else:
        scene.add(actor.line(viz_tracks, fake_tube=True, linewidth=2))
    scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                    position=(-360.11, -30.46, -40.44),
                    view_up=(-0.03, 0.028, 0.89))
    if interactive:
        window.show(scene)
    else:
        window.record(scene, out_path=name+".png", size=(2000,2000))

def get_total_points(tracks):
    return np.vstack(tracks).shape[0]

def merge_tracks(tracks): 
    return


def interpolate_streamlines(tracks, pps, c = 3):
    """
    streamlines: list
    pps: int (points per streamline)
    c: channel is 3 
    """
    a = len(tracks)
    resampled_tracks = np.zeros((a,pps,c))

    # TODO: Multiprocessing this loop to speed up
    for i,track in enumerate(tracks):
        # old_points = np.linspace(0, 1, track.shape[0])
        x_old = np.linspace(track[:, 0].min(), track[:, 0].max(), track.shape[0])
        y_old = np.linspace(track[:, 1].min(), track[:, 1].max(), track.shape[0])
        z_old = np.linspace(track[:, 2].min(), track[:, 2].max(), track.shape[0])

        x_new = np.linspace(track[:, 0].min(), track[:, 0].max(), pps)
        y_new = np.linspace(track[:, 1].min(), track[:, 1].max(), pps)
        z_new = np.linspace(track[:, 2].min(), track[:, 2].max(), pps)

        x_interp = CubicSpline(x_old, track[:, 0], bc_type='natural')
        y_interp = CubicSpline(y_old, track[:, 1], bc_type='natural')
        z_interp = CubicSpline(z_old, track[:, 2], bc_type='natural')
        
        # Resample streamline
        resampled_track = np.vstack([
            x_interp(x_new),
            y_interp(y_new),
            z_interp(z_new)
        ]).T
        
        resampled_tracks[i] = resampled_track
    return resampled_tracks

def get_named_colors():
    named_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
        'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
        'blueviolet', 'brown', 'burlywood', 'cadetblue',
        'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
        'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
        'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey',
        'green', 'greenyellow', 'honeydew', 'hotpink',
        'indianred', 'indigo', 'ivory', 'khaki',
        'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgrey', 'lightgreen', 'lightpink',
        'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime',
        'limegreen', 'linen', 'magenta', 'maroon',
        'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
        'moccasin', 'navajowhite', 'navy', 'oldlace',
        'olive', 'olivedrab', 'orange', 'orangered',
        'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
        'palevioletred', 'papayawhip', 'peachpuff', 'peru',
        'pink', 'plum', 'powderblue', 'purple',
        'rebeccapurple', 'red', 'rosybrown', 'royalblue',
        'saddlebrown', 'salmon', 'sandybrown', 'seagreen',
        'seashell', 'sienna', 'silver', 'skyblue',
        'slateblue', 'slategray', 'slategrey', 'snow',
        'springgreen', 'steelblue', 'tan', 'teal',
        'thistle', 'tomato', 'turquoise', 'violet',
        'wheat', 'white', 'whitesmoke', 'yellow',
        'yellowgreen'
    ]
    return named_colors
def visualize_tracks(tracks, points, static_color=None, save=False):
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

            if static_color is None: 
                color_index = i % len(named_colors)  # Calculate the color index based on the current iteration
            else:
                color_index = -1 
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

def load_TractoInferno(tracts_directory):
    tracks = [] 
    labels = [] 
    import re        
    for path in tqdm(Path(tracts_directory).iterdir()):
        path = path / "tractography"
        for p in tqdm(Path(path).iterdir()):
            subject = re.search(r'\d{4}', p).group()
            print(subject)
            if p.is_file() and path.suffix == '.trk':
                label = re.search(r'__(.*?)\.trk', p).group(1)
                print(label)
                tractogram = load_trk(str(path), "same", bbox_valid_check=False)
                num_streamlines = len(tractogram.streamlines)
        with open(os.path.join(tracts_directory, f"{subject}.txt"), "a") as f:
            
            f.write(f"Number of Streamline in {label} is {num_streamlines}\n")

        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
    return tracks, labels

def load_HCP(tracts_directory):
    tracks = [] 
    labels = [] 
    import re
    dataset = "HCP"
    subject = re.search(r'\d{6}', tracts_directory).group()
    print(f"Loading streamlines for {subject} for {dataset} dataset")
    for path in tqdm(Path(tracts_directory).iterdir()):
        if path.is_file() and path.suffix == '.trk':
            label = str(path).split('/')[-1].split('.')[0]
            tractogram = load_trk(str(path), "same", bbox_valid_check=False)
            num_streamlines = len(tractogram.streamlines)
            for s in tractogram.streamlines:
                labels.append(label)
                tracks.append(s)
    return tracks, labels


def downsample_streamline(args):
    streamline, pts = args
    """
    Input:
        streamline: list of points
        pts: number of points to downsample or upsample to
    """
    streamline = apt(streamline,.1)
    if len(streamline) != pts: 
        resampled_streamline = interpolate_streamline((streamline, pts))
    else:
        resampled_streamline = streamline
    return resampled_streamline

def interpolate_streamline(args):
    streamline, pts = args
    """
    Input:
        streamline: list of points
        pts: number of points to downsample or upsample to
    """

    x_old = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), streamline.shape[0])
    y_old = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), streamline.shape[0])
    z_old = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), streamline.shape[0])

    x_new = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), pts)
    y_new = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), pts)
    z_new = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), pts)
    
    x_interp = CubicSpline(x_old, streamline[:, 0], bc_type='natural')
    y_interp = CubicSpline(y_old, streamline[:, 1], bc_type='natural')
    z_interp = CubicSpline(z_old, streamline[:, 2], bc_type='natural')
    
    resampled_streamline = np.vstack([
        x_interp(x_new),
        y_interp(y_new),
        z_interp(z_new)
    ]).T
    
    return resampled_streamline

def choose_upORdownsample_streamline(args):
    streamline, pts = args
    """
    Input:
        streamline: list of points
        pts: number of points to downsample or upsample to
    """
    if pts > len(streamline):
        return interpolate_streamline((streamline, pts))
    else:
        return downsample_streamline((streamline, pts))

def create_clusters_QBX_3L_per_subject(args):
    data, subject_id, streamline_threshold_per_cluster, test = args

    """
    Input:
        data: dict containing cluster_ids_l1, cluster_ids_l2, cluster_ids_l3, label
        subject_id: subject_id
        streamline_threshold_per_cluster: threshold for minimum number of streamlines in a cluster
        test: whether these clusters are for testset or not

    Output:
        clusters: list of list of streamlines
        pc_cluster_labels: list of labels for each cluster
    """

    cluster_id_to_idx_mapping = {
        'L1': defaultdict(list),
        'L2': defaultdict(list),
        'L3': defaultdict(list),
    }

    for idx, cid in enumerate(data['cluster_ids_l1']):
        cluster_id_to_idx_mapping['L1'][cid].append(idx)

    for idx, cid in enumerate(data['cluster_ids_l2']):
        cluster_id_to_idx_mapping['L2'][cid].append(idx)

    for idx, cid in enumerate(data['cluster_ids_l3']):
        cluster_id_to_idx_mapping['L3'][cid].append(idx)
    
    included_streamlines = np.zeros(len(data['label'])) # none included here
    
    clusters = []
    pc_cluster_labels = []
    cluster_ids_l3, cluster_count_l3 = np.unique(data['cluster_ids_l3'], return_counts=True)

    cluster_ids_l2 = []
    for cluster_id in cluster_ids_l3[cluster_count_l3 <  streamline_threshold_per_cluster]:
        cluster_ids_l2.append(data['cluster_ids_l2'][cluster_id_to_idx_mapping['L3'][cluster_id][0]])
    cluster_ids_l2 = np.unique(cluster_ids_l2)
    cluster_count_l2 = np.zeros(len(cluster_ids_l2))
    for i,cluster_id in enumerate(cluster_ids_l2):
        cluster_count_l2[i] = len(cluster_id_to_idx_mapping['L2'][cluster_id])
    
    cluster_ids_l1 = []
    for cluster_id in cluster_ids_l2[cluster_count_l2 <  streamline_threshold_per_cluster]:
        cluster_ids_l1.append(data['cluster_ids_l1'][cluster_id_to_idx_mapping['L2'][cluster_id][0]])
    cluster_ids_l1 = np.unique(cluster_ids_l1)
    cluster_count_l1 = np.zeros(len(cluster_ids_l1))
    for i,cluster_id in enumerate(cluster_ids_l1):
        cluster_count_l1[i] = len(cluster_id_to_idx_mapping['L1'][cluster_id])
    
    for cluster_id in cluster_ids_l1:
        cluster_streamline_idx = cluster_id_to_idx_mapping['L1'][cluster_id]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< streamline_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    for cluster_id in cluster_ids_l2:
        cluster_streamline_idx = cluster_id_to_idx_mapping['L2'][cluster_id]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< streamline_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    for cluster_id in cluster_ids_l3:
        cluster_streamline_idx = cluster_id_to_idx_mapping['L3'][cluster_id]
        if included_streamlines[cluster_streamline_idx[0]] == 1:
            continue
        if test==False and len(cluster_streamline_idx)< streamline_threshold_per_cluster:
            continue
        included_streamlines[cluster_streamline_idx] = 1
        u_labels, u_counts = np.unique(data['label'][cluster_streamline_idx], return_counts=True)
        label = u_labels[np.argmax(u_counts)]
        cluster_streamline_idx = [f'{subject_id}_{i}' for i in cluster_streamline_idx]
        clusters.append(cluster_streamline_idx)
        pc_cluster_labels.append(label)

    return (clusters, pc_cluster_labels)

def poisson_surface_reconstruction(point_cloud_np, depth=9):
    import open3d as o3d
    # Convert the numpy array to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    
    # Perform Poisson Surface Reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth)
    
    # Optional: Apply density threshold to remove low-density regions
    density_threshold = np.percentile(densities, 95)  # Example: keep 95% of high-density regions
    vertices_to_keep = densities > density_threshold
    mesh.remove_vertices_by_mask(~vertices_to_keep)
    
    # Extract vertices and convert to numpy array
    vertices_np = np.asarray(mesh.vertices)
    return vertices_np

def cal_local_feat(cur_feat, k_ds_rate=0.1, k =20 , use_endpoints_dist = True, cal_equiv_dist=False):
    import torch
    
    """ Calculate the local feature for all streamlines in cur_feat
    Args:
        cur_feat: [n_fiber, n_feat, n_point]
        k_ds_rate: the rate of downsample the fibers to calculate the distance matrix. 1 means no downsample
        k: the number of nearest neighbor streamlines (local)
        use_endpoints_dist (bool): whether to use the distance between endpoints to calculate the distance matrix
        cal_equiv_dist (bool): whether to calculate the equivalent distance matrix

    Returns:
        cur_local_feat: [n_fiber*k, n_feat, n_point], features of the k nearest neighbor streamlines
    """
    # local feat
    # [n_fiber, k], [n_fiber, k], [n_fiber,n_feat,n_point], [n_fiber,n_feat,n_point]
    near_idx, near_flip_mask, ds_cur_feat, ds_cur_feat_equiv = dist_mat_knn(
                        torch.from_numpy(cur_feat), k_ds_rate, k, use_endpoints_dist, cal_equiv_dist)
    
    #selecting downsampled local features from org, by selecting the near index streamlines
    cur_local_feat_org = ds_cur_feat[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    cur_local_feat_equiv = ds_cur_feat_equiv[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    near_flip_mask = near_flip_mask.reshape(-1)[:,None,None]  # [n_fiber*k, 1, 1]
    near_nonflip_mask = 1-near_flip_mask
    cur_local_feat = cur_local_feat_org*near_nonflip_mask + cur_local_feat_equiv*near_flip_mask  # [n_fiber*k, n_feat, n_point]

    return cur_local_feat 


def dist_mat_knn(brain_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    import torch
    from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   

    """ 
        calculate the distance between streamlines (fibers) and then find the neighbor streamlines (fibers)
        input (self) 
            brain_feat: [n_fiber, n_feat, n_point]
            k_ds_rate (float): the rate of downsample the fibers to calculate the distance matrix
            k (int): the number of nearest neighbors
            use_endpoints_dist (bool): whether to use endpoints distance to calculate the nearest neighbor streamlines (fibers)
            cal_equiv_dist (bool): whether to calculate the equivalent distance
        output 
            idx (the nearest idx): [n_fiber, k]
            flip_mask (the flip mask): [n_fiber, k]. whether the nearest fiber feature should use flipped fiber (reverse order)
            ds_brain_feat (the downsampled brain_feat): [n_fiber, n_feat, n_point]
            ds_brain_feat_equiv (the downsampled equivalent (reverse order) brain_feat): [n_fiber, n_feat, n_point]
    """
        
    # calculate the distance matrix. Take the minus since we wanna find the smallest distance using topk.
    if 0 < k_ds_rate < 1:
        num_ds_feat =  int(brain_feat.shape[0]*k_ds_rate)
        ds_indices = np.random.choice(brain_feat.shape[0], size=num_ds_feat, replace=False)
        downsample_feat = brain_feat[ds_indices,:,:]  # [n_ds_fiber, n_point, n_feat]
    else:
        downsample_feat = brain_feat
    if use_endpoints_dist:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation_endpoints(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber) for dist_mat
    else:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber)
    
    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]   # (N_fiber, k). largest is False then the k smallest elements are returned.
    near_idx = topk_idx[:,:]   # (N_fiber, k)
    # print(dist_mat[0,near_idx[0,:]])
    near_flip_mask = torch.gather(flip_mask, dim=1, index=near_idx) # (N_fiber, k). The flip mask of the info fibers (neighbor).

    return near_idx.numpy(), near_flip_mask.numpy(), ds_brain_feat.numpy(), ds_brain_feat_equiv.numpy()

def bicubic_interpolate_single_streamline(label, sub_id, feature,ras_feature, indices, num_point_per_fiber):
    original_x = np.arange(feature.shape[0])
    new_x = np.linspace(0, original_x[-1], num_point_per_fiber)
    num_dims = feature.shape[1]
    interpolated_streamline = np.zeros((num_point_per_fiber, num_dims))

    for dim in range(num_dims):
        cs = CubicSpline(original_x, feature[:, dim], bc_type='natural')
        interpolated_streamline[:, dim] = cs(new_x)

    return label, sub_id, interpolated_streamline,ras_feature, indices 

def bicubic_interpolate(labels, subject_ids, features, ras_feature, indices, num_point_per_fiber):
    """
    Interpolates a set of streamlines using bicubic interpolation with multiprocessing

    Args:
        labels (numpy.ndarray): An array of labels.
        subject_ids (numpy.ndarray): An array of subject IDs.
        features (numpy.ndarray): An array of features.
        ras_feature (numpy.ndarray): An array of RAS features or original streamlines which 
        won't be interpolated 
        indices (numpy.ndarray, optional): An array of indices.

    Returns:
        tuple: A tuple containing the interpolated labels, subject IDs, features,
               RAS features, and indices.

    """
    import multiprocessing
    arg_streamlines = [(labels[i],subject_ids[i] ,features[i],ras_feature[i], indices[i], num_point_per_fiber) for i in range(len(features))]
    with multiprocessing.Pool() as pool:
        result_list = pool.starmap(bicubic_interpolate_single_streamline, arg_streamlines)
    labels, subject_ids, features, ras_feature, indices = zip(*result_list)
    return np.array(labels), np.array(subject_ids), np.array(features),np.array(ras_feature), np.array(indices)

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def shuffle(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]
    return data, labels, idx