import io
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns




def single_column_heatmap_image(df, heatmap_title):
    colormap = 'PuOr_r'
    labelsize = 3
    heatmapable_df_keys = [
        '1h_vs_0h', '2h_vs_0h', '4h_vs_0h', '6h_vs_0h', '8h_vs_0h', '10h_vs_0h', '12h_vs_0h', '16h_vs_0h', '24h_vs_0h', '48h_vs_0h']
    fig, ax = plt.subplots()
    offset = 0
    heatmapable_df = df[heatmapable_df_keys]
    sns.heatmap(
        heatmapable_df,
        ax=ax,
        xticklabels=list(map(lambda x: x[:2] if len(x) == 9 else x[0], heatmapable_df_keys)),
        yticklabels=list(df['gene_name']),
        cmap=colormap,
        vmin=-8,
        vmax=8,
        cbar=True,
    )
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.tick_params(which='major', labelsize=5, rotation=0)
    ax.set_aspect(1)
    ax.tick_params(which='major', labelsize=labelsize, rotation=0)
    ax.tick_params(which='minor', labelsize=labelsize, rotation=0)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    fig.suptitle(heatmap_title, fontsize=6)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=300,  figsize=(10, 1), format='png') #, bbox_inches='tight', pad_inches=0.6)
    plt.close()
    im = Image.open(buf)
    return im




def multiple_column_heatmap_image(df, rows_per_column, heatmap_title):
    colormap = 'PuOr_r'
    labelsize = 3
    column_n = int(np.ceil(df.shape[0] / rows_per_column))
    heatmapable_df_keys = [
        '1h_vs_0h', '2h_vs_0h', '4h_vs_0h', '6h_vs_0h', '8h_vs_0h', '10h_vs_0h', '12h_vs_0h', '16h_vs_0h', '24h_vs_0h', '48h_vs_0h']
    fig, axes = plt.subplots(
        ncols=column_n
    )
    fig.subplots_adjust(wspace=0.6)
    offset = 0
    fst_row = offset
    lst_row = offset + rows_per_column
    for ax in axes[:-1]:
        heatmapable_df = df.iloc[fst_row:lst_row][heatmapable_df_keys]
        sns.heatmap(
            heatmapable_df,
            ax=ax,
            xticklabels=list(map(lambda x: x[:2] if len(x) == 9 else x[0], heatmapable_df_keys)),
            yticklabels=list(df.iloc[fst_row:lst_row]['gene_name']),
            cmap=colormap,
            vmin=-8,
            vmax=8,
            cbar=False,
        )
        ax.set_aspect(1)
        ax.tick_params(which='major', labelsize=labelsize, rotation=0)
        ax.tick_params(which='minor', labelsize=labelsize, rotation=0)
        b, t = ax.get_ylim() # discover the values for bottom and top
        b += 0.4 # Add 0.4 to the bottom
        t -= 0.4 # Subtract 0.4 from the top
        ax.set_ylim(b, t) # update the ylim(bottom, top) values
        fst_row += rows_per_column
        lst_row += rows_per_column
    heatmapable_df = df.iloc[fst_row:][heatmapable_df_keys]
    ax = axes[-1]
    sns.heatmap(
        heatmapable_df,
        ax=ax,
        xticklabels=list(map(lambda x: x[:2] if len(x) == 9 else x[0], heatmapable_df_keys)),
        yticklabels=list(df.iloc[fst_row:lst_row]['gene_name']),
        cmap=colormap,
        vmin=-8,
        vmax=8,
        cbar=True,
    )
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.tick_params(which='major', labelsize=5, rotation=0)
    ax.set_aspect(1)
    ax.tick_params(which='major', labelsize=labelsize, rotation=0)
    ax.tick_params(which='minor', labelsize=labelsize, rotation=0)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    fig.suptitle(heatmap_title, fontsize=6)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=300,  figsize=(10, 1), format='png') #, bbox_inches='tight', pad_inches=0.6)
    plt.close()
    im = Image.open(buf)
    return im



def heatmap_images(df, rows_per_column, max_column_number, heatmap_title):
    rows_per_image = rows_per_column * max_column_number
    if df.shape[0] <= rows_per_column:
        return [
            single_column_heatmap_image(df, heatmap_title)
        ]
    elif df.shape[0] <= rows_per_image:
        n_cols = np.ceil(df.shape[0] / rows_per_column)
        return [
            multiple_column_heatmap_image(df, rows_per_column, heatmap_title)
        ]
    else:
        ims = []
        num_heatmaps = int(np.ceil(df.shape[0] / rows_per_image))
        for i in range(int(np.floor(df.shape[0] / rows_per_image))):
            first_index = i * rows_per_image
            last_index = (i+1) * rows_per_image
            df_subset = df[first_index:last_index]
            ims.append(
                multiple_column_heatmap_image(df_subset, rows_per_column, '{}, page {} of {}'.format(heatmap_title, i+1, num_heatmaps))
            )
        i = int(np.floor(df.shape[0] / rows_per_image))
        first_index = i * rows_per_image
        df_subset = df[first_index:]
        if df_subset.shape[0] <= rows_per_column:
            ims.append(
                single_column_heatmap_image(df_subset, '{}, page {} of {}'.format(heatmap_title, i+1, num_heatmaps))
            )
        else:
            ims.append(
                multiple_column_heatmap_image(df_subset, rows_per_column, '{}, page {} of {}'.format(heatmap_title, i+1, num_heatmaps))
            )
        return ims
    


if __name__ == '__main__':

    df = pd.read_csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__clusters.csv', header=0)

