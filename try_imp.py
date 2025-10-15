#%%
import pickle
import os
import json
import ast
import numpy as np
from multiprocessing import Pool
import gc
from tqdm import tqdm

root_path = 'predictions/'
save_path = 'predictions/true_pred_streamlines'
def string_to_list(streamline):
    list_str = ','.join(streamline.split())
    clean_list_str = list_str.replace('[,', '[').replace(',]', ']')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    clean_list_str = clean_list_str.replace(',,,,', ',').replace(',,,', ',').replace(',,', ',')
    clean_list_str = clean_list_str.replace(',', ', ')
    clean_list_str = clean_list_str.replace(', dtype=float32)', '').replace('array([', '[')
    streamline = list(ast.literal_eval(clean_list_str))
    
    return streamline


for dataset in os.listdir(root_path):
    if dataset not in ['TractoInferno', '105HCP']:
        continue
    for model in os.listdir(os.path.join(root_path, dataset)):
        for method in os.listdir(os.path.join(root_path, dataset, model)):
            for exp in os.listdir(os.path.join(root_path, dataset, model, method)):
                print('Processing:', dataset, model, method, exp)
                os.makedirs(os.path.join(save_path, dataset, model, method, exp), exist_ok=True)
                exp_path = os.path.join(root_path, dataset, model, method, exp)
                pkl_path = os.path.join(exp_path, 'pkls', f'test_pred__{model}.pkl')
                if not pkl_path.endswith('.pkl'):
                    continue
                
                if dataset == 'TractoInferno':
                    with open('label_int_mappings/label_int_mapping_TrInF.json', 'r') as f:
                        label_int_mapping_TrInF = json.load(f)
                    int_label_mapping = {v: k for k, v in label_int_mapping_TrInF.items()}
                elif dataset == '105HCP':
                    with open('label_int_mappings/label_int_mapping_105HCP.json', 'r') as f:
                        label_int_mapping_105HCP = json.load(f)
                    int_label_mapping = {v: k for k, v in label_int_mapping_105HCP.items()}
                else:
                    raise NotImplementedError(f'No label_int_mapping for {dataset}')
                
                if True:
                    with open(pkl_path, 'rb') as f:
                        true_labels, predicted_labels, streamlines, subjects = pickle.load(f)
                    
                    with Pool() as p:
                        streamlines = list(tqdm(p.imap(string_to_list, streamlines), total=len(streamlines), desc='Converting streamlines to list'))
                    print('Total streamlines:', len(streamlines))
                    
                    subjects = np.array(subjects)
                    unique_subjects = np.unique(subjects)

                    # true_labels = [int_label_mapping[t] for t in true_labels]
                    # predicted_labels = [int_label_mapping[t] for t in predicted_labels]
                    true_labels = [int_label_mapping[t] for t in true_labels]
                    predicted_labels = [int_label_mapping[t] for t in predicted_labels]

                    for sub in unique_subjects:
                        if sub not in ['599469', 'sub-1006']:
                            continue
                        subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
                        true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
                        predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]

                        unique_true_labels_subject = np.unique(true_labels_subject)
                        unique_pred_labels_subject = np.unique(predicted_labels_subject)
                        for label in tqdm(unique_pred_labels_subject, desc=f'Processing pred subject {sub}'):
                            if os.path.exists(f'{save_path}/{dataset}/{model}/{method}/{exp}/{sub}/{label}/pred_streamlines.pkl'):
                                continue
                            pred_streamlines = [s for s,t in zip(subject_streamlines, predicted_labels_subject) if t == label]

                            # pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                            pred_save_path = os.path.join(save_path, dataset, model, method, exp, sub, label)
                            os.makedirs(pred_save_path, exist_ok=True)
                            with open(f'{pred_save_path}/pred_streamlines.pkl', 'wb') as f:
                                pickle.dump(pred_streamlines, f)
                        
                        for label in tqdm(unique_true_labels_subject, desc=f'Processing true subject {sub}'):
                            if os.path.exists(f'{save_path}/{dataset}/{model}/{method}/{exp}/{sub}/{label}/true_streamlines.pkl'):
                                continue
                            true_streamlines = [s for s,t in zip(subject_streamlines, true_labels_subject) if t == label]

                            # pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                            true_save_path = os.path.join(save_path, dataset, model, method, exp, sub, label)
                            os.makedirs(true_save_path, exist_ok=True)
                            with open(f'{true_save_path}/true_streamlines.pkl', 'wb') as f:
                                pickle.dump(true_streamlines, f)
                    
                    del true_labels, predicted_labels, streamlines, subjects
                    gc.collect()
                # elif 'qbx' in methods:
                #     with open(method_path, 'rb') as f:
                #         true_labels_str, predicted_cluster_labels, subjects_str, streamline_cluster_label_str, raw_streamlines_cluster_str = pickle.load(f)
                
                #     with Pool() as p:
                #         cluster_subjects = p.map(string_to_list, subjects_str)
                #     with Pool() as p:
                #         streamline_cluster_label = p.map(string_to_list, streamline_cluster_label_str)
                #     with Pool() as p:
                #         raw_streamlines_cluster = p.map(string_to_list, raw_streamlines_cluster_str)

                #     streamlines = []
                #     predicted_labels = []
                #     true_labels = []
                #     subjects = []
                #     for i in range(len(cluster_subjects)):
                #         for j in range(len(cluster_subjects[i])):
                #             streamlines.append(raw_streamlines_cluster[i][j])
                #             predicted_labels.append(predicted_cluster_labels[i])
                #             true_labels.append(streamline_cluster_label[i][j])
                #             subjects.append(cluster_subjects[i][j])
                #     subjects = np.array(subjects)

                #     true_labels = [int_label_mapping[t] for t in true_labels]
                #     predicted_labels = [int_label_mapping[t] for t in predicted_labels]

                #     unique_subjects = np.unique(subjects)
                #     for sub in unique_subjects:
                #         if os.path.exists(f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}'):
                #             continue
                #         if sub not in ['599469', 'sub-1006']:
                #             continue
                #         subject_streamlines = [s for s,t in zip(streamlines, subjects) if t == sub]
                #         true_labels_subject = [t for t, s in zip(true_labels, subjects) if s == sub]
                #         predicted_labels_subject = [t for t, s in zip(predicted_labels, subjects) if s == sub]

                #         unique_true_labels_subject = np.unique(true_labels_subject)
                #         unique_pred_labels_subject = np.unique(predicted_labels_subject)
                #         for label in unique_pred_labels_subject:
                #             pred_streamlines = [s for s,t in zip(subject_streamlines, predicted_labels_subject) if t == label]

                #             pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                #             os.makedirs(pickle_path, exist_ok=True)
                #             with open(f'{pickle_path}/pred_streamlines.pkl', 'wb') as f:
                #                 pickle.dump(pred_streamlines, f)
                        
                #         for label in unique_true_labels_subject:
                #             true_streamlines = [s for s,t in zip(subject_streamlines, true_labels_subject) if t == label]

                #             pickle_path = f'{save_path}/{"_".join(methods.split(".")[0].split("_")[2:])}/{sub}/{label}'
                #             os.makedirs(pickle_path, exist_ok=True)
                #             with open(f'{pickle_path}/true_streamlines.pkl', 'wb') as f:
                #                 pickle.dump(true_streamlines, f)
                            
                    
                #     del true_labels_str, predicted_cluster_labels, subjects_str, streamline_cluster_label_str, raw_streamlines_cluster_str
                #     del cluster_subjects, streamline_cluster_label, raw_streamlines_cluster
                #     del streamlines, predicted_labels, true_labels, subjects
                #     gc.collect()
                # else:
                #     raise NotImplementedError(f'No label_int_mapping for {methods}')
                

#%%
# import pandas as pd
# import json

# json_path = 'predictions_new/single_GPTM_105HCP71__TrInF32/dice_scores_single_105HCP71__TrInF32.json'
# with open(json_path, 'r') as f:
#     fiesta_results = json.load(f)

# fiesta_results = fiesta_results['sub-1006']
# fiesta_scores = []
# dice = {}
# overlap = {}
# overreach = {}
# for class_ in fiesta_results.keys():
#     dice[class_] = fiesta_results[class_]['dice']
#     overlap[class_] = fiesta_results[class_]['overlap']
#     overreach[class_] = fiesta_results[class_]['overreach']

# fiesta_scores.append(dice)
# fiesta_scores.append(overlap)
# fiesta_scores.append(overreach)
# fiesta_df = pd.DataFrame(fiesta_scores, index=['Dice', 'Overlap', 'Overreach'])

# # Sort columns alphabetically
# fiesta_df = fiesta_df.reindex(sorted(fiesta_df.columns), axis=1)

# fiesta_df.to_csv(json_path.replace('.json', '.csv'))