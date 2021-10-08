from __future__ import print_function
import os, sys
import json
import argparse
import statistics
import numpy as np
import pickle
import torch
import random
from collections import Counter
import time
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import submodlib
from submodlib.helper import create_kernel
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import GraphCutMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction

# random.seed(42)

accent_map = {"ABA":"arabic","SKA":"arabic","YBAA":"arabic","ZHAA":"arabic",
              "BWC":"chinese","LXC":"chinese","NCC":"chinese","TXHC":"chinese",
              "ASI":"hindi","RRBI":"hindi","SVBI":"hindi","TNI":"hindi",
              "HJK":"korean","HKK":"korean","YDCK":"korean","YKWK":"korean",
              "EBVS":"spanish","ERMS":"spanish","MBMPS":"spanish","NJS":"spanish",
              "HQTV":"vietnamese","PNV":"vietnamese","THV":"vietnamese","TLV":"vietnamese"
              }

accent_short_forms = {"hindi":"HIN", "korean":"KOR", "vietnamese":"VTN", "arabic":"ARB", "chinese":"CHN", "spanish":"ESP"}
composed_accent_map = {k: accent_short_forms.get(v) for k, v in accent_map.items()}

def legend_tag(label_map):
    # if label_map in accent_map:
    if False:
        return "{}: {}".format(label_map,composed_accent_map[label_map])
    else:
        return label_map

def _color_map(dirs):
    color_map = {}
    accent_color_map = {}
    for accent, color in zip(['arabic', 'hindi', 'chinese', 'spanish', 'korean', 'vietnamese'],
                             ['c', 'm', 'y', 'limegreen', 'orange', 'b']):
        accent_color_map[accent] = color
    for _dir, color in zip(['query_set', 'selected_set', 'test_set'], ['k', 'r', 'dimgray']):
        color_map[_dir] = color
    for _dir in dirs:
        color_map[_dir] = accent_color_map[accent_map[_dir]]
    for accent in ['arabic', 'hindi', 'chinese', 'spanish', 'korean', 'vietnamese']:
        for _dir in dirs:
            if accent_map[_dir]==accent:
                color_map[_dir] = accent_color_map[accent]
    return color_map

def _plot_PCA(X, y, _dirs, pca_result, extra_result, label_map, _ax, color_map, marker_sizes):
    feat_cols = ['dim'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    label_map = dict(enumerate(_dirs))
    df['label'] = df['y'].apply(lambda i: legend_tag(label_map[i]))
    df['pca-one'] = np.concatenate([pca_result[:,0], extra_result[:, 0]], axis=0)
    df['pca-two'] = np.concatenate([pca_result[:,1], extra_result[:, 1]], axis=0)
    g = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=color_map, 
        data=df,
        legend="full",
        alpha=0.6,
        s=marker_sizes,
        ax = _ax
    )
    g.legend(loc='center right', bbox_to_anchor=(0.05, 0.9))

def _plot_TSNE(X, y, _dirs, _ax, color_map, markers, marker_sizes):
    feat_cols = ['dim'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y

    label_map = dict(enumerate(_dirs))
    df['label'] = df['y'].apply(lambda i: legend_tag(label_map[i]))

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=color_map, 
        data=df,
        legend="full",
        alpha=0.6,
        s=marker_sizes,
        markers=markers,
        ax = _ax
    )
    g.legend(loc='center right', bbox_to_anchor=(0.05, 0.9))


def plot_TSNE(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices):
    n = len(dirs)
    selected_features = ground_features[selected_indices]
    color_map = _color_map(dirs)
    fig = plt.figure(figsize=(30,30))

#    _ax = fig.add_subplot(3,2,1)
    _ax = fig.add_subplot(1,2,1)
    X = np.concatenate([ground_features, query_features, selected_features], axis=0)
    y = np.concatenate([ground_features_Y, np.zeros((len(query_features), 1))+n, np.zeros((len(selected_features), 1))+n+1], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30, np.zeros((len(query_features),))+40, np.zeros((len(selected_features),))+40], axis=0)
    _plot_TSNE(X, y, dirs+['query_set', 'selected_set'], _ax, color_map, [',']*len(dirs)+['.', '*'], marker_sizes)

#    _ax = fig.add_subplot(3,2,3)
#    X = np.concatenate([ground_features], axis=0)
#    y = np.concatenate([ground_features_Y], axis=0)
#    _plot_TSNE(X, y, dirs, _ax, color_map, [',']*len(dirs), marker_sizes)
#
#    _ax = fig.add_subplot(3,2,5)    
#    X = np.concatenate([ground_features, test_features], axis=0)
#    y = np.concatenate([ground_features_Y, np.zeros((len(test_features), 1))+n], axis=0)
#    _plot_TSNE(X, y, dirs+['test_set'], _ax, color_map, [',']*len(dirs)+['1'], marker_sizes)

#    _ax = fig.add_subplot(3,2,6)    
    _ax = fig.add_subplot(1,2,2)    
    X = np.concatenate([ground_features, test_features, query_features, selected_features], axis=0)
    y = np.concatenate([ground_features_Y, np.zeros((len(test_features), 1))+n, np.zeros((len(query_features), 1))+n+1, np.zeros((len(selected_features), 1))+n+2], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30, np.zeros((len(test_features),))+40, np.zeros((len(query_features),))+40, np.zeros((len(selected_features),))+40], axis=0)
    _plot_TSNE(X, y, dirs+['test_set', 'query_set', 'selected_set'], _ax, color_map, [',']*len(dirs)+['.', '1', '*'], marker_sizes)

    fig.suptitle('random', fontsize = 14, fontweight ='bold')
    plt.savefig(run_dir+'/plots/TSNE_visualization.png')

def plot_PCA(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices):
    n = len(dirs)
    selected_features = ground_features[selected_indices]
    color_map = _color_map(dirs)
    fig = plt.figure(figsize=(30,30))
    pca = PCA(n_components=3)

    _ax = fig.add_subplot(3,2,3)
    X = np.concatenate([ground_features], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30], axis=0)
    y = np.concatenate([ground_features_Y], axis=0)

    feat_cols = ['dim'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    label_map = dict(enumerate(dirs))
    df['label'] = df['y'].apply(lambda i: legend_tag(label_map[i]))

    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    _ax = fig.add_subplot(3,2,3)
    g = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=color_map,
        data=df,
        legend="full",
        alpha=0.6,
        s=marker_sizes,
        ax = _ax
    )
    g.legend(loc='center right', bbox_to_anchor=(0.05, 0.9))

    query_select_transform = pca.transform(np.concatenate([query_features, selected_features], axis=0))
    test_transform = pca.transform(test_features)
    query_select_test_transform = pca.transform(np.concatenate([test_features, query_features, selected_features], axis=0))

    X = np.concatenate([ground_features, query_features, selected_features], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30, np.zeros((len(query_features),))+40, np.zeros((len(selected_features),))+40], axis=0)
    y = np.concatenate([ground_features_Y, np.zeros((len(query_features), 1))+n, np.zeros((len(selected_features), 1))+n+1], axis=0)
    _ax = fig.add_subplot(3,2,1)
    _plot_PCA(X, y, dirs+['query_set', 'selected_set'], pca_result, query_select_transform, label_map, _ax, color_map, marker_sizes)

    _ax = fig.add_subplot(3,2,5)
    X = np.concatenate([ground_features, test_features], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30, np.zeros((len(test_features),))+40,], axis=0)
    y = np.concatenate([ground_features_Y, np.zeros((len(test_features), 1))+n], axis=0)    
    _plot_PCA(X, y, dirs+['test_set'], pca_result, test_transform, label_map, _ax, color_map, marker_sizes)

    _ax = fig.add_subplot(3,2,6)
    X = np.concatenate([ground_features, test_features, query_features, selected_features], axis=0)
    marker_sizes = np.concatenate([np.zeros((len(ground_features),))+30, np.zeros((len(test_features),))+40, np.zeros((len(query_features),))+40, np.zeros((len(selected_features),))+40], axis=0)
    y = np.concatenate([ground_features_Y, np.zeros((len(test_features), 1))+n, np.zeros((len(query_features), 1))+n+1, np.zeros((len(selected_features), 1))+n+2], axis=0)
    _plot_PCA(X, y, dirs+['test_set', 'query_set', 'selected_set'], pca_result, query_select_test_transform, label_map, _ax, color_map, marker_sizes)

    fig.suptitle('random', fontsize = 14, fontweight ='bold')
    plt.savefig(run_dir+'/plots/PCA_visualization.png')

def plots(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices):
     plot_TSNE(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)
     plot_PCA(dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)

#    n = len(dirs)
#    #X = np.concatenate([ground_features, query_features, ground_features[selected_indices], test_features], axis=0)
#    #y = np.concatenate([ground_features_Y, np.zeros((len(query_features), 1)) + n, np.zeros((len(selected_indices), 1)) + n+1, np.zeros((len(test_features), 1)) + n+2], axis=0)
#    X = np.concatenate([ground_features, query_features, ground_features[selected_indices]], axis=0)
#    y = np.concatenate([ground_features_Y, np.zeros((len(query_features), 1)) + n, np.zeros((len(selected_indices), 1)) + n+1], axis=0)
#    feat_cols = ['dim'+str(i) for i in range(X.shape[1])]
#    df = pd.DataFrame(X, columns=feat_cols)
#    df['y'] = y
#    
#    #label_map = dict(enumerate(dirs + ['query_set', 'selected_set', 'test_set']))
#    label_map = dict(enumerate(dirs + ['query_set', 'selected_set']))
#    df['label'] = df['y'].apply(lambda i: legend_tag(label_map[i]))
#    
#    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#    tsne_results = tsne.fit_transform(df[feat_cols].values)
#    
#    df['tsne-2d-one'] = tsne_results[:,0]
#    df['tsne-2d-two'] = tsne_results[:,1]
#
#    color_map = _color_map(dirs)
#    fig = plt.figure(figsize=(10,10))
#    
#    ax = fig.add_subplot(1,1,1)
#    g = sns.scatterplot(
#        x="tsne-2d-one", y="tsne-2d-two",
#        hue="label",
#        palette=color_map,
#        data=df,
#        legend="full",
#        alpha=0.6,
        # s=marker_sizes, 
#        ax = ax
#    )
#    g.legend(loc='center right', bbox_to_anchor=(0.2, 0.9))
#    
#    fig.suptitle('random', fontsize = 14, fontweight ='bold')
#    plt.savefig(run_dir+'/plots/visualization.png')


def load_features(file_dir, feature_type):
    features = []
    with open(file_dir.replace('.json', f'_{feature_type}.file'), 'rb') as f:
        while True:
            try:
                features.append(pickle.load(f))
            except EOFError:
                break
    features = np.concatenate(features, axis=0)
    print(features.shape)
    return features

def get_index(list_samples, indices):
    total_duration, index = 0, 0
    while total_duration + list_samples[indices[index]]['duration'] <= 360:
        total_duration += list_samples[indices[index]]['duration']
        index += 1
    return index
    
def preprocess(base_dir, target_size, budget_size, speaker, other_speaker, feature_type):
#    dirs = [f.name for f in os.scandir('.') if f.is_dir()]
#    dirs.remove('.ipynb_checkpoints')
#    dirs.remove('reserved_TSS_output')
#    print('__', dirs)

    accent = accent_map[speaker]

    query_dir = f'{speaker}/manifests/' 
    query_file_path = base_dir + query_dir + 'seed.json'
    query_list = [json.loads(line.strip()) for line in open(query_file_path)]
    query_features = load_features(query_file_path, feature_type)
    query_list, query_features = query_list[:target_size], query_features[:target_size]

    ### test file
    test_file_path = base_dir + query_dir + 'test.json'
    test_list = [json.loads(line.strip()) for line in open(test_file_path)]
    test_features = load_features(test_file_path, feature_type)

    list_total_selection, list_total_count, list_total_duration = [], [], []
    list_accent_sample_count, list_accent_sample_duration = [], []
    output_dir = os.path.join(base_dir, query_dir, f'TSS_output/within/budget_{budget_size}/target_{target_size}/{other_speaker}')
    os.makedirs(output_dir, exist_ok=True)
    for i in [1, 2, 3]:
        run = f'run_{i}'
        run_dir = os.path.join(output_dir, run)
        for folder in ['train', 'output', 'plots']:
            os.makedirs(os.path.join(run_dir, folder), exist_ok=True)
        random.seed(i)
        ground_list, ground_list_Y, ground_features = [], [], []
        selected_indices = []
        temp_dirs = []
        count = 0

        temp_dirs.append(other_speaker)
        selection_file_path = base_dir + '../speaker_with/' + other_speaker + '/manifests/selection.json'
        selection_file_list = [json.loads(line.strip()) for line in open(selection_file_path)]
        indices = list(range(len(selection_file_list)))
        random.shuffle(indices)
        index = get_index(selection_file_list, indices)
        selected_indices = [idx for idx in indices[:index]][:]
        ground_list = selection_file_list[:]
        ground_features = load_features(selection_file_path, feature_type)
        ground_list_Y = [count]*len(selection_file_list)   

        #ground_features = np.concatenate(ground_features, axis=0)
        ground_features_Y = np.asarray(ground_list_Y).reshape(-1, 1)

        print(len(ground_list), ground_features.shape)
        print(len(query_list), query_features.shape)
        print(len(test_list), test_features.shape)

        selected_list = [ground_list[idx] for idx in selected_indices]
        accent_sample_count, accent_sample_duration = 0, 0
        for item in selected_list:
            if accent_map[item['audio_filepath'].split('/')[-3]] == accent_map[speaker]:
                accent_sample_count += 1
                accent_sample_duration += item['duration']
        list_accent_sample_count.append(accent_sample_count)
        list_accent_sample_duration.append(accent_sample_duration)
        list_total_selection.append(Counter([item['audio_filepath'].split('/')[-3] for item in selected_list]))
        list_total_count.append(len(selected_indices))
        list_total_duration.append(sum([item['duration'] for item in selected_list]))
        
#        train_list = selected_list + query_list
        train_list = selected_list

        with open(f'{run_dir}/train/train.json', 'w') as f:
            for line in train_list:
                f.write('{}\n'.format(json.dumps(line)))
                
#        plots(temp_dirs, run_dir, ground_features, ground_features_Y, query_features, test_features, selected_indices)
    
    stats = 'total selection : ' + ' '.join(map(str, list_total_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_count))
    stats += 'total selection duration: ' + ' '.join(map(str, list_total_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_total_duration))
    stats += 'accented selection: ' + ' '.join(map(str, list_accent_sample_count)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_count))
    stats += 'accented duration: ' + ' '.join(map(str, list_accent_sample_duration)) + ' -> {0:.2f}\n'.format(statistics.mean(list_accent_sample_duration))
    stats += '\nall selections: ' + str(list_total_selection)
    
    with open(output_dir + '/stats.txt', 'w') as f:
        f.write(stats)

parser = argparse.ArgumentParser(description ='TSS input')
parser.add_argument('--budget', type = int,  help ='budget')
parser.add_argument('--target', type = int,  help ='target')
parser.add_argument('--speaker', type = str,  help ='query set')
parser.add_argument('--other_speaker', type = str,  help ='other speaker')
parser.add_argument('--feature_type', type = str,  help ='feature space')

args = parser.parse_args()
budget_size = args.budget
target_size = args.target
speaker = args.speaker
other_speaker = args.other_speaker
feature_type = args.feature_type
base_dir = ''

preprocess(base_dir, target_size, budget_size, speaker, other_speaker, feature_type)
