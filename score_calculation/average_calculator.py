"""
Calculates average for all the csv files in the given directory
Stores the results in a new csv file with sorted columns
"""

import os
import pandas as pd
import argparse


def read_csv(path):
    """
    Reads a csv file and returns a pandas DataFrame
    """
    return pd.read_csv(path)

def calculate_average(df):
    """
    Calculates the average of the given DataFrame across rows and ignores NaN values
    """
    return df.mean(axis=1, skipna=True, numeric_only=True)
def drop_columns(df, columns):
    """
    Drops the columns with the given names from the DataFrame
    """
    return df.drop(columns, axis=1, errors='ignore')

def drop_rows(df, rows):
    """
    Drops the rows with the given indices from the DataFrame. if the index is not present, it will be ignored
    """
    return df.drop(rows, axis=0, errors='ignore')

def set_index(df, index):
    """
    Sets the index of the DataFrame to the given column
    """
    return df.set_index(index)
def sort_columns(df):
    """
    Sorts the columns of the DataFrame
    """
    return df.reindex(sorted(df.columns), axis=1)

def sort_rows(df):
    """
    Sorts the rows of the DataFrame
    """
    return df.sort_index()


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
    path = f'predictions_encodernano/{args.exp_subset}/{args.model}/{args.dataset_name}/test_{args.exp_name}/results'
    # path = 'predictions_encodernano/105HCP/TractoGPT-M/Tracklet_Sentence/test_7_from_12_threshold_7/results'

    metric_order = ['dice', 'overlap', 'overreach', 'precesion', 'recall', 'f1']
    # Read all the csv files
    csvs = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            if file.split('__')[-1].split('.')[0].split('-')[0] in metric_order:
                csvs.append(file)
    dataframes = []
    for csv in csvs:
        try:
            metric = csv.split('__')[-1].split('.')[0].split('-')[0]
            if metric in metric_order:
                dataframes.append(read_csv(os.path.join(path, csv)))
            else:
                print(f"Skipping {csv}, metric not in order: {metric}")
        except Exception as e:
            print(f"Error processing {csv}: {e}")

    
    # Calculate average
    for i, df in enumerate(dataframes):
        df_name = csvs[i].split('__')[-1].split('.')[0].split('-')[0]
        # check if class name is present in the df
        if 'class name' in df.columns:
            df = set_index(df, 'class name')
        else:
            df.rename(columns={'Unnamed: 0': 'class name'}, inplace=True)
            df = set_index(df, 'class name')
        df = drop_columns(df, ['weighted avg'])
        df = drop_rows(df, ['weighted avg'])
        df = sort_rows(df)
        df = calculate_average(df)
        df.name = df_name
        dataframes[i] = df
    
    # Combine all the averages into a single DataFrame
    df = pd.concat(dataframes, axis=1)
    df = df.T
    df = sort_columns(df)
    df = df.round(4)
    # make the order of rows wiht metric_order
    df = df.reindex(metric_order)

    # Save the DataFrame to a new csv file
    df.to_csv(os.path.join(path, 'average.csv'))

