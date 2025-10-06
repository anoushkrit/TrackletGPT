import os
import pickle as pkl

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def get_true_pred_labels(true_pred_dir, true_pred_files):
    true_labels = []
    pred_labels = []
    for file in true_pred_files:
        true_pred = load_pickle(os.path.join(true_pred_dir, file))
        true_labels.extend(true_pred[0])
        pred_labels.extend(true_pred[1])
    return true_labels, pred_labels

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    """
    Generates and saves a confusion matrix heatmap without numbers, and automatically uses unique labels 
    from the true and predicted values for the x and y axes.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels
    filename (str): The filename to save the plot (default is 'confusion_matrix.png')
    """
    # Get unique labels from true and predicted values
    labels = sorted(set(y_true).union(set(y_pred)))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create a heatmap using seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap='hot', xticklabels=labels, yticklabels=labels)
    # plt.grid(True, which='both', linestyle='--', linewidth=1)

    # Labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')

    # Save the plot as a PNG file
    plt.savefig(filename)

    # Show the plot
    plt.show()

if __name__ == '__main__':

    true_pred_dir = 'predictions_encodernano/TractoInferno/TractoGPT-M/Tracklet_Sentence/test_7_from12_threshold_7/results/true_pred_labels'
    true_pred_files = os.listdir(true_pred_dir)


    true_labels, pred_labels = get_true_pred_labels(true_pred_dir, true_pred_files)

    remove_labels = ['CC_Fr_2', 'CC_Pa','CC_Pr_Po', 'POPT_L', 'POPT_R', 'PYT_L', 'PYT_R']
    # for label in remove_labels:
    #     true_labels = [l for l in true_labels if l != label]
        # pred_labels = [l for l in pred_labels if l != label]
    
    # print the accuracy
    acc = 0
    for i in range(len(true_labels)):
        if true_labels[i] in remove_labels:
            continue
        if true_labels[i] == pred_labels[i]:
            acc += 1
    
    for label in remove_labels:
        true_labels = [l for l in true_labels if l != label]
    print(acc/len(true_labels))
    # plot_confusion_matrix(true_labels, pred_labels, f'{os.path.dirname(true_pred_dir)}/confusion_matrix.png')
    # print(np.unique(true_labels, return_counts=True))
    # print(np.unique(pred_labels, return_counts=True))
    # print(confusion_matrix(true_labels, pred_labels))
    