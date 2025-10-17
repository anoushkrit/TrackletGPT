import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dice import *    
import pickle
import argparse
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from utils.dice import hcp_scores
import ast
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import csv
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import defaultdict



def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def calculate_weighted_average(csv_path, supports: dict) -> pd.DataFrame:
    # Read csv file
    df = pd.read_csv(csv_path)

    # if Unnamed: 0 column exists and class name coloumn does not exist rename Unnamed: 0 to class name
    if 'Unnamed: 0' in df.columns and 'class name' not in df.columns:
        df.rename(columns={'Unnamed: 0': 'class name'}, inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'weighted avg' in df.columns:
        df.drop('weighted avg', axis=1, inplace=True)
    if 'weighted avg' in df['class name'].values:
        df = df[df['class name'] != 'weighted avg']
    
    # calculating weighted average across all columns/subjects
    weighted_avg = {}
    for subject in df.columns:
        if subject == 'class name':
            continue
        weighted_avg[subject] = 0
        for _class in df['class name']:
            value = df[df['class name'] == _class][subject].values[0]
            if np.isnan(value):
                continue
            support = supports[subject][_class]
            overall_support = supports[subject]['overall_support']
            weighted_avg[subject] += value * (support / overall_support)

    weighted_avg['class name'] = 'weighted avg'
    df = pd.concat([df, pd.DataFrame(weighted_avg, index=[0])], ignore_index=True)

    # calculating average across all classes/rows
    weighted_avg = {}
    for _class in df['class name']:
        weighted_avg[_class] = 0
        overall_support = 0
        for subject in df.columns:
            if subject == 'class name':
                continue
            value = df[df['class name'] == _class][subject].values[0]
            if np.isnan(value):
                continue
            if _class != 'weighted avg':
                support = supports[subject][_class]
            else:
                support = supports[subject]['overall_support']
            overall_support += support
            weighted_avg[_class] += value * support
        weighted_avg[_class] /= overall_support
    
    df['weighted avg'] = [weighted_avg[_class] for _class in df['class name']]

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a model with specified parameters.")
    
    # Add arguments
    parser.add_argument("--model", required=True, help="Name of the model to use.")
    parser.add_argument("--exp_name", required=True, help="Experiment name.")
    parser.add_argument("--exp_subset", required=True, help="Experiment subset.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--mapping_path", required=True, help="Path to the label int mapping file.")

    # Parse arguments
    args = parser.parse_args()

    # output_path
    output_dir = f'predictions/{args.exp_subset}/{args.model}/{args.dataset_name}/test_{args.exp_name}/results'

    # list csv files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    csv_files_path = [os.path.join(output_dir, f) for f in csv_files]

    # reading json and creating support mapping
    json_file_path = [f for f in os.listdir(output_dir) if f.endswith('.json')][0]
    json_file_path = os.path.join(output_dir, json_file_path)

    json_data = load_json(json_file_path)

    supports = {}
    for key in json_data.keys():
        overall_support = 0
        supports[key] = {}
        for _class in json_data[key][1]:
            if _class in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            supports[key][_class] = json_data[key][1][_class]['support']
            overall_support += supports[key][_class]
        supports[key]['overall_support'] = overall_support
        
    # calculating weighted average for all csv files
    for csv_file in csv_files_path:
        try:
            df = calculate_weighted_average(csv_file, supports)
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error in calculating weighted average for {csv_file}")
            print(e)
            continue
