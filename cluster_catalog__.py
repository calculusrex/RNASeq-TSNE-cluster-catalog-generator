import io
import PIL
from PIL import Image
import functools as ft

import pandas as pd
import numpy as np
from cluster__ import plot_clusters__from_array, clustered_array
from heatmap__ import heatmap_images
from colors__ import generate_rgb_colors

import matplotlib
import matplotlib.pyplot as plt



def plot_clusters(df, cluster_column):
    arr1 = clustered_array(df, cluster_column)
    plot_clusters__from_array(arr1)

    
def cluster_df_subset(df, cluster_index):
    return df[df['cluster_lvl1'] == cluster_index]


def resize_image_by_factor(im, factor):
    w, h = im.size
    return im.resize(
        (
            int(np.ceil(w * factor)),
            int(np.ceil(h * factor)),
        ),
        resample=PIL.Image.BICUBIC
    )


def concatenate_images_horizontally(im1, im2):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w_sum = w1 + w2
    h_max = max(h1, h2)
    out = Image.new('RGB', (w_sum, h_max))
    out.paste(im1, (0, 0))
    out.paste(im2, (w1, 0))
    return out


def concatenate_images_vertically(im1, im2):
    w1, h1 = im1.size
    w2, h2 = im2.size
    w_max = max(w1, w2)
    h_sum = h1 + h2
    out = Image.new('RGB', (w_max, h_sum))
    out.paste(im1, (0, 0))
    out.paste(im2, (0, h1))
    return out


def cluster_highlight_image(df, cluster_column_label, cluster_index):
    arr1 = clustered_array(df, cluster_column_label)
    x_y = arr1[:, -2:]
    x_y__coi = clustered_array(df[df[cluster_column_label] == cluster_index],
                               cluster_column_label)[:, -2:]
    plt.plot(x_y[:, 0], x_y[:, 1], '.', markerfacecolor=(0.75, 0.75, 0.75, 1), markeredgecolor=(0.75, 0.75, 0.75, 1), markersize=2)
    plt.plot(x_y__coi[:, 0], x_y__coi[:, 1], '.', markerfacecolor=(0.95, 0.2, 0.2, 1), markeredgecolor=(0.95, 0.2, 0.2, 1), markersize=2)
    plt.text(x_y__coi[:, 0].mean(),
             x_y__coi[:, 1].mean(), str(int(cluster_index)), fontsize=6)
    plt.title('Cluster {}'.format(cluster_index))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    return im


def cluster_go_category_image(df, cluster_column_label, cluster_index):
    df_cluster = df[df[cluster_column_label] == cluster_index]
    arr0 = np.array(df_cluster[['go_category', 'tsne_0', 'tsne_1']])
    x_y = arr0[:, -2:]
    labels = arr0[:, 0]
    unique_labels, label_counts = np.unique(
        np.array(df_cluster['go_category']),
        return_counts = True)
    n_clusters = unique_labels.shape[0]
    # color_samples = np.linspace(0, 1, len(unique_labels))
    # colors = [plt.cm.nipy_spectral(each)
    #           for each in color_samples]
    colors = generate_rgb_colors(len(unique_labels))
    labels_corelated_w_colors = list(zip(unique_labels, colors))
    for k, col in labels_corelated_w_colors:
        class_member_mask = (labels == k)
        xy = x_y[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=8)
    plt.title('Cluster {}, (categories)'.format(cluster_index))
    patches = list(
        map(lambda col: matplotlib.patches.Patch(facecolor=col),
            colors)
    )
    legend_keys = list(
        map(lambda label_n_count: '{} ({})'.format(label_n_count[0], label_n_count[1]),
            zip(unique_labels, label_counts)))
    plt.legend(patches, legend_keys, fontsize=8)#, loc=(-0.1, 0))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    return im



def cluster_go_description_image(df, cluster_column_label, cluster_index):
    df_cluster = df[df[cluster_column_label] == cluster_index]
    arr0 = np.array(df_cluster[['go_description', 'tsne_0', 'tsne_1']])
    x_y = arr0[:, -2:]
    labels = arr0[:, 0]
    unique_labels, label_counts = np.unique(
        np.array(df_cluster['go_description']),
        return_counts = True)
    n_clusters = unique_labels.shape[0]
    # color_samples = np.linspace(0, 1, len(unique_labels))
    # colors = [plt.cm.gist_rainbow(each)
    #           for each in color_samples]
    colors = generate_rgb_colors(len(unique_labels))
    labels_corelated_w_colors = list(zip(unique_labels, colors))
    fig = plt.figure()
    ax = plt.subplot(111)
    for k, col in labels_corelated_w_colors:
        class_member_mask = (labels == k)
        xy = x_y[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4)
    plt.title('Cluster {}, (descriptions)'.format(cluster_index))
    patches = list(
        map(lambda col: matplotlib.patches.Patch(facecolor=col),
            colors)
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.4, box.height])
    legend_keys = list(
        map(lambda label_n_count: '{} ({})'.format(label_n_count[0], label_n_count[1]),
            zip(unique_labels, label_counts)))
    ax.legend(patches, legend_keys, fontsize=4, loc='center left', bbox_to_anchor=(1.025, 0.5))# , loc=(-0.1, 0))
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    im = Image.open(buf)
    return im


def cluster_heatmap_images(df, cluster_index):
    return heatmap_images(df[df['cluster_lvl1'] == cluster_index],
                          50, 4,
                          'Cluster {}'.format(cluster_index))


def cluster_breakdown_data(df, cluster_column_label, cluster_index):
    print('Processing cluster {}:'.format(cluster_index))
    data = {}
    print('Creating highlight image....')
    data['highlight'] = cluster_highlight_image(df, cluster_column_label, cluster_index)
    print('Creating gene ontology category image....')
    data['go_category'] = cluster_go_category_image(df, cluster_column_label, cluster_index)
    print('Creating gene ontology description image....')
    data['go_description'] = cluster_go_description_image(df, cluster_column_label, cluster_index)
    print('Creating heatmap images....')
    data['heatmaps'] = cluster_heatmap_images(df, cluster_index)
    print('------------------------------------------------------------------------')
    return data


# def cluster_breakdown_flier_image(df, cluster_column_label, cluster_index):
#     brkdwn = cluster_breakdown_data(df, cluster_column_label, cluster_index)
#     out = concatenate_images_horizontally(
#         resize_image_by_factor(brkdwn['highlight'], 0.5),
#         resize_image_by_factor(brkdwn['go_category'], 0.5)
#     )
#     out = concatenate_images_vertically(
#         out,
#         brkdwn['go_description']
#     )
#     return out


def cluster_breakdown_flier_image(df, cluster_column_label, cluster_index):
    brkdwn = cluster_breakdown_data(df, cluster_column_label, cluster_index)
    cluster_plots = concatenate_images_horizontally(
        resize_image_by_factor(brkdwn['highlight'], 0.5),
        resize_image_by_factor(brkdwn['go_category'], 0.5)
    )
    cluster_plots = concatenate_images_vertically(
        cluster_plots,
        brkdwn['go_description']
    )
    heatmaps = ft.reduce(lambda im1, im2: concatenate_images_vertically(im1, im2),
                         brkdwn['heatmaps'])
    return concatenate_images_vertically(cluster_plots,
                                         heatmaps)


# def cluster_breakdown_flier_image(df, cluster_column_label, cluster_index):
#     brkdwn = cluster_breakdown_data(df, cluster_column_label, cluster_index)
#     cluster_plots = ft.reduce(lambda im1, im2: concatenate_images_horizontally(im1, im2),
#                               map(lambda im: resize_image_by_factor(im, 0.33),
#                                   [
#                                       brkdwn['highlight'],
#                                       brkdwn['go_category'],
#                                       brkdwn['go_description']
#                                   ]))
#     heatmaps = ft.reduce(lambda im1, im2: concatenate_images_vertically(im1, im2),
#                          brkdwn['heatmaps'])
#     return concatenate_images_vertically(cluster_plots,
#                                          heatmaps)



def cluster_regional_breakdown__pdf(df, output_filename, bounding_box, cluster_column):
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = bounding_box
    subset_df = df[(df['tsne_0'] > x_top_left) &
                   (df['tsne_0'] < x_bottom_right) &
                   (df['tsne_1'] > y_bottom_right) &
                   (df['tsne_1'] < y_top_left)]
    clusters = subset_df['cluster_lvl1'].unique()
    fliers = list(map(lambda cluster_index: cluster_breakdown_flier_image(df, cluster_column, cluster_index),
                      clusters))
    fliers[0].save(output_filename, 'PDF', save_all=True, append_images=fliers[1:])



def generate_cluster_catalog(df, output_filename, cluster_column):
    clusters = np.array(df['cluster_lvl1'].unique())
    clusters.sort()
    clusters_per_catalog = 1
    n_clusters = df['cluster_lvl1'].unique().shape[0]
    for i in range(n_clusters // clusters_per_catalog):
        print('Generating cluster catalog {}'.format(i))
        print('-------------------------------------------------------------------')
        clusters__subset = clusters[i*clusters_per_catalog:(i+1)*clusters_per_catalog]
        fliers = list(map(lambda cluster_index: cluster_breakdown_flier_image(df, cluster_column, cluster_index),
                          clusters__subset))
        fliers[0].save('{}____{}.pdf'.format(output_filename, i),
                       'PDF',
                       save_all=True,
                       append_images=fliers[1:])


def generate_cluster_catalog__csv(input_filename, output_filename):
    df = pd.read_csv(input_filename, header=0)
    generate_cluster_catalog(df, output_filename, 'cluster_lvl1')
    return df

 
# def cluster_breakdown_pages(c_h_img, c_go_cat_img, c_go_desc_img, c_heat, c_brkdwn):
#     pass





# def generate_cluster_breakdown_pages(df, cluster_column_label, cluster_index):
#     c_h_img = cluster_highlight_image(df, cluster_column_label, cluster_index)
#     c_go_cat_img = cluster_go_category_image(df, cluster_column_label, cluster_index)
#     c_go_desc_img = cluster_go_description_image(df, cluster_column_label, cluster_index)
#     c_heat_imgs = cluster_heatmap_images(df, cluster_index)

#     c_brkdwn = cluster_breakdown_pages(
#         c_h_img,
#         c_go_cat_img,
#         c_go_desc_img,
#         c_heat_imgs
#     )
#     return c_brkdwn


# def generate_cluster_catalog(df, output_filename, cluster_column_label):
#     pagess = []
#     for clstr in df[cluster_column_label].unique():
#         pagess.append(generate_cluster_breakdown_pages)
#     pages = flatten_pages(pagess)
#     generate_pdf_from_pages(pages, output_filename)


# def generate_cluster_catalog__csv(input_filename, output_filename):
#     df = pd.read_csv(input_filename, header=0)
#     generate_cluster_catalog(df, output_filename, 'cluster_lvl1')        


def plot_clusters_in_range(df, n0, n1):
    cluster_subset = np.array(range(n0, n1))
    df_subset = df[df['cluster_lvl1'].isin(cluster_subset)]
    plot_clusters__from_dataframe(df_subset, 'cluster_lvl1')


if __name__ == '__main__':

    # lvl_0, lvl1 = generate_cluster_catalog(
    #     'transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__clusters.csv')

    input_filename = 'transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__clusters.csv'

    df = pd.read_csv(input_filename, header=0)
    unique_lvl0_clusters = df['cluster_lvl0'].unique()
    unique_lvl1_clusters = df['cluster_lvl1'].unique()

