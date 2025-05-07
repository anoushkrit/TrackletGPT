from dipy.tracking.distances import approx_polygon_track as apt

from collections import defaultdict
import numpy as np

def find_matching_indices(arr1, arr2):
    """where arr2 is the smaller 
    and arr1 is the larger array"""
    match_array = [] 
    # to check if the index is already booked or not
    # booked = np.zeros(arr1.shape[0]+2, dtype=bool)
    for i in range(arr2.shape[0]):
        matches = np.where(np.all(np.isclose(arr2[i], arr1, atol=1e-3), axis=1))[0]
        # if not matches.size == 0 and booked[matches[0]]==0 and booked[matches[0]+2]==0 and booked[matches[0]+1]==0:
        if not matches.size == 0:
            # booked[matches[0]+1]s = True
            match_array.append(matches[0])
    return match_array

def find_matching_indices_hash(large_pts: np.ndarray,
                               small_pts: np.ndarray,
                               tol: float = 1e-3):
    # 1) Compute integer cell coordinates
    inv = 1.0 / tol
    cell_coords = np.floor(large_pts * inv).astype(int)
    
    # 2) Build hash: cell → list of large‐pts indices
    hash_table = defaultdict(list)
    for idx, cell in enumerate(cell_coords):
        key = tuple(cell)
        hash_table[key].append(idx)
    
    # np.array to check if that index is booked or not
    booked = np.zeros(2+large_pts.shape[0], dtype=bool)
    # 3) For each small_pt, check its cell and 26 neighbors
    result = []
    for pt in small_pts:
        c = tuple(np.floor(pt * inv).astype(int))
        found = -1
        # iterate over the 3×3×3 neighborhood of cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_key = (c[0]+dx, c[1]+dy, c[2]+dz)
                    for idx in hash_table.get(neighbor_key, ()):
                        if booked[idx]==0 and booked[idx+2]==0 and np.linalg.norm(large_pts[idx] - pt) <= tol:
                            found = idx
                            booked[idx+1] = True
                            break
                    if found != -1:
                        break
                if found != -1:
                    break
            if found != -1:
                break
        result.append(found)
    return result

def remove_empty_tracks(tracklets):
    return [tracklet for tracklet in tracklets if tracklet.shape[0] != 0]
    
def get_tracklets(indices, streamline, overlap = 0):
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
        tracklets = remove_empty_tracks(tracklets)
        return tracklets, tracklet_indices

def get_tracklet_from_streamline(streamline, apt_param=0.4) -> list:
    """Get a tracklet from a streamline."""

    crucial_points = apt(streamline, apt_param)

    # find indices of crucial points in the streamline in most optimal way
    crucial_point_indices = find_matching_indices(streamline, crucial_points)
    # crucial_point_indices = sorted(crucial_point_indices)
    # crucial_point_indices = find_matching_indices_hash(streamline, crucial_points)
    # print(crucial_point_indices)
    # remove crucial point if they are adjacent
    for i in range(len(crucial_point_indices)-1, 1, -1):
        if crucial_point_indices[i] - crucial_point_indices[i-1] <= 1:
            crucial_point_indices.pop(i)
    # get tracklets from the streamline
    tracklets, tracklet_indices = get_tracklets(crucial_point_indices, streamline, overlap=2)

    return tracklets
