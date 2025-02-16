from dipy.tracking.distances import approx_polygon_track as apt

import numpy as np

def find_matching_indices(arr1, arr2):
    """where arr2 is the smaller 
    and arr1 is the larger array"""
    match_array = [] 
    for i in range(arr2.shape[0]):
        matches = np.where(np.all(np.isclose(arr2[i], arr1, atol=1e-3), axis=1))[0]
        if not matches.size == 0:
            match_array.append(matches[0])
    return match_array

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
    # print(crucial_point_indices)
    # remove crucial point if they are adjacent
    for i in range(len(crucial_point_indices)-1, 1, -1):
        if crucial_point_indices[i] - crucial_point_indices[i-1] <= 1:
            crucial_point_indices.pop(i)

    # get tracklets from the streamline
    tracklets, tracklet_indices = get_tracklets(crucial_point_indices, streamline, overlap=2)

    return tracklets
