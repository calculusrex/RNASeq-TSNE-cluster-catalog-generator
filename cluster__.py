import pandas as pd
import numpy as np
# from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib
import matplotlib.pyplot as plt
import functools as ft
import os

from colors__ import generate_rgb_colors


def plot_clusters__from_array(arr1):
    x_y = arr1[:, -2:]
    labels = arr1[:, -3]
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = unique_labels.shape[0]
    # color_samples = np.linspace(0, 1, len(unique_labels))
    # np.random.shuffle(color_samples)
    # colors = [plt.cm.gist_rainbow(each)
    #           for each in color_samples]
    colors = generate_rgb_colors(len(unique_labels))
    labels_corelated_w_colors = list(zip(unique_labels, colors))
    for k, col in labels_corelated_w_colors:
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = x_y[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
        plt.text(xy[:, 0].mean(),
                 xy[:, 1].mean(), str(int(k)), fontsize=8)
    plt.title('{} clusters'.format(n_clusters))
    plt.show()


def savefig_clusters__from_array(filename, arr1):
    x_y = arr1[:, -2:]
    labels = arr1[:, -3]
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = unique_labels.shape[0]
    # color_samples = np.linspace(0, 1, len(unique_labels))
    # np.random.shuffle(color_samples)
    # colors = [plt.cm.gist_rainbow(each)
    #           for each in color_samples]
    colors = generate_rgb_colors(len(unique_labels))
    labels_corelated_w_colors = list(zip(unique_labels, colors))
    for k, col in labels_corelated_w_colors:
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = x_y[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
        plt.text(xy[:, 0].mean(),
                 xy[:, 1].mean(), str(int(k)), fontsize=8)
    plt.title('{} clusters'.format(n_clusters))
    plt.savefig('{}.png'.format(filename), dpi=300)
    plt.close()




def remove_file_extension(filename):
    fname = os.path.splitext(filename)[0]
    return fname


def insert_index(array):
    return np.insert(array,
                     0,
                     np.array(range(array.shape[0])),
                     axis=1)


def dataset_radius(x_y):
    return np.max([
        x_y.max(),
        np.abs(x_y.min())
    ])


def insert_clustering_labels(arr, labels):
    return np.insert(arr,
                     -2,
                     labels,
                     axis=1)

def set_clustering_labels(arr, labels):
    arr[:, -3] = labels
    return arr


def unique_clusters(arr1):
    return np.array(np.unique(arr1[:, -3], return_counts=True)).transpose()


def shave_clustering_pass__eps(arr, eps):
    x_y = arr[:, -2:]
    clustering = DBSCAN(
        eps=eps,
        min_samples=1
    ).fit(x_y)
    arr[:, -3] = clustering.labels_
    return arr


def shave_clustering_pass(arr):
    x_y = arr[:, -2:]
    eps = dataset_radius(x_y) / 4
    print('eps: ', eps)
    arr1 = shave_clustering_pass__eps(arr, eps)
    while len(unique_clusters(arr1)) < 2:
        eps = eps / 2
        print('eps: ', eps)
        arr1 = shave_clustering_pass__eps(arr, eps)
    print('=================================================')
    return arr1


def agglomerative_clustering_pass(arr):
    x_y = arr[:, -2:]
    clustering = AgglomerativeClustering(n_clusters=32).fit(x_y)
    arr[:, -3] = clustering.labels_
    return arr


def agglomerative_clustering_pass__cluster_count(arr, n_clusters):
    x_y = arr[:, -2:]
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(x_y)
    arr[:, -3] = clustering.labels_
    return arr




def filter_by_cluster(arr0, clstr_ix):
    return np.array(
        list(
            filter(lambda row: row[-3] == clstr_ix,
                   arr0)))
    

def discriminate_final_lvl_clusters_by_fragmentability(arr0):
    clusters__indices_n_counts = np.array(
        np.unique(arr0[:, -3],
                  return_counts=True)
    ).transpose()
    fragmentable = []
    agglomerative = []
    terminal = []
    for cluster_index__n__count in clusters__indices_n_counts:
        if cluster_index__n__count[1] <= 512:
            terminal.append(cluster_index__n__count[0])
        elif cluster_index__n__count[1] <= 8192:
            agglomerative.append(cluster_index__n__count[0])
        else:
            fragmentable.append(cluster_index__n__count[0])
    fragmentable = list(
        map(lambda clstr_ix: filter_by_cluster(arr0, clstr_ix),
            fragmentable))
    agglomerative = list(
        map(lambda clstr_ix: filter_by_cluster(arr0, clstr_ix),
            agglomerative))
    terminal = list(
        map(lambda clstr_ix: filter_by_cluster(arr0, clstr_ix),
            terminal))
    return fragmentable, agglomerative, terminal

            

# def shave_clustering__tree(arr0):
#     arr1 = shave_clustering_pass(arr0)
#     fragmentable_clusters, terminal_clusters = discriminate_final_lvl_clusters_by_fragmentability(arr1)
#     cluster_dict = {}
#     cluster_dict['terminal'] = np.concatenate(terminal_clusters)
#     if len(fragmentable_clusters) != 0:
#         cluster_dict['fragged'] = list(
#             map(shave_clustering,
#                 fragmentable_clusters))
#     return cluster_dict


def concatenate_cluster_arrays__dual(arr1, arr2):
    arr1_cluster_n = arr1[:, -3].max() + 1
    arr2[:, -3] = arr2[:, -3] + arr1_cluster_n
    return np.concatenate([arr1, arr2])


def concatenate_cluster_arrays(arrays):
    return ft.reduce(concatenate_cluster_arrays__dual,
                     arrays)


def shave_clustering__flat(arr0):
    arr1 = shave_clustering_pass(arr0)
    fragmentable_clusters, agglomerative_clusters, terminal_clusters = discriminate_final_lvl_clusters_by_fragmentability(arr1)
    clusters = []
    clusters.append(
        np.concatenate(terminal_clusters))
    if len(agglomerative_clusters) != 0:
        clusters.append(
            concatenate_cluster_arrays(
                map(agglomerative_clustering_pass,
                    agglomerative_clusters)))
    if len(fragmentable_clusters) != 0:
        clusters.append(
            concatenate_cluster_arrays(
                map(shave_clustering__flat,
                    fragmentable_clusters)))
    return concatenate_cluster_arrays(
        clusters)


def reduce_to_string_label(integer_row):
    return ft.reduce(lambda a, b: '{}.{}'.format(a, b),
                     list(integer_row))
    

def flatten_cluster_hierarchy(tree_arr):
    string_labels = list(map(reduce_to_string_label,
                             tree_arr))
    i = 0
    string_2_integer_label_corr_dict = {}
    for s_lbl in np.unique(np.array(string_labels)):
        string_2_integer_label_corr_dict[s_lbl] = i
        i += 1
    integer_labels = list(map(lambda s_lbl: string_2_integer_label_corr_dict[s_lbl],
                              string_labels))
    return np.array([string_labels, integer_labels])


def clustering_array(df):
    arr0 = insert_index(
        np.array(
            df[['tsne_0', 'tsne_1']]))
    arr0 = insert_clustering_labels(arr0, np.repeat(0, arr0.shape[0]))
    return arr0


def clustered_array(df, cluster_column_label):
    arr0 = insert_index(
        np.array(
            df[[cluster_column_label,
                'tsne_0',
                'tsne_1']]))
    return arr0


def incorporate_cluster_labels_from_array(df, arr, label):
    arr_t = arr.transpose()
    arr_df = pd.DataFrame({'cluster': arr_t[1].astype(np.int32),
                           'tsne_0': arr_t[2],
                           'tsne_1': arr_t[3]},
                          index=arr_t[0]).sort_index()
    df.insert(14, label, arr_df['cluster'], True)


def cluster__csv(input_filename, output_filename):
    df = pd.read_csv(input_filename, header=0)
    arr0 = insert_index(
        np.array(
            df[['tsne_0', 'tsne_1']]))
    arr0 = insert_clustering_labels(arr0, np.repeat(0, arr0.shape[0]))
    arr2 = shave_clustering__flat(arr0)
    incorporate_cluster_labels_from_array(df, arr2, 'cluster_0')
    df.to_csv(output_filename, index=False)


def cluster__csv__dual_layer(input_filename, output_filename):
    df = pd.read_csv(input_filename, header=0)
    arr0 = clustering_array(df)
    arr1 = shave_clustering_pass__eps(
        arr0, 1.5)
    fragmentable_clusters, agglomerative_clusters, terminal_clusters = discriminate_final_lvl_clusters_by_fragmentability(arr1)
    terminal_cluster__lvl1 = np.concatenate(
        terminal_clusters)
    fragmentable_clusters = fragmentable_clusters + agglomerative_clusters
    fragmented_clusters = []
    for clstr0 in fragmentable_clusters:
        clstr1 = shave_clustering__flat(
            clstr0)
        fragmented_clusters.append(
            clstr1)
    fragged = concatenate_cluster_arrays(fragmented_clusters)
    arr2 = concatenate_cluster_arrays([
        terminal_cluster__lvl1,
        fragged
    ])
    incorporate_cluster_labels_from_array(
        df, arr1, 'cluster_lvl0')
    incorporate_cluster_labels_from_array(
        df, arr2, 'cluster_lvl1')
    df.to_csv(output_filename, index=False)
    return df



def agglomerative_clustering__csv(input_filename, output_filename, n_clusters):
    df = pd.read_csv(input_filename, header=0)
    arr0 = clustering_array(df)    
    arr1 = agglomerative_clustering_pass__cluster_count(arr0, n_clusters)
    incorporate_cluster_labels_from_array(
        df, arr1, 'cluster_lvl1')
    df.to_csv(output_filename, index=False)
    savefig_clusters__from_array(remove_file_extension(output_filename), arr1)
    return df



if __name__ == '__main__':
    print('cluster__')

    # df = cluster__csv__dual_layer('transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne.csv',
    #                               'transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__clusters.csv')

    
    # arr1 = clustered_array(df, 'cluster_lvl0')
    # arr2 = clustered_array(df, 'cluster_lvl1')
