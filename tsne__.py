import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def scatter_plot(xs, ys):
    plt.scatter(xs, ys)
    plt.show()


def compute_tsne__csv(input_filename, output_filename):

    df = pd.read_csv(input_filename, header=0)

    features_df = df[
        df.columns[2:12]
    ]

    X = np.array(features_df)

    X_embedded = TSNE(
        n_components=2,
        early_exaggeration=48, # default is 12
        learning_rate=100, # default is 200
        verbose=0, # default is 0
    ).fit_transform(X)

    X_embedded__transposed = X_embedded.transpose()

    df.insert(14, 'tsne_0', X_embedded__transposed[0], True)
    df.insert(15, 'tsne_1', X_embedded__transposed[1], True)

    df.to_csv(output_filename, index=False)



if __name__ == '__main__':

    df = pd.read_csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled.csv')

    features_df = df[
        df.columns[2:12]
    ]

    X = np.array(features_df)

    X_embedded = TSNE(
        n_components=2,
        early_exaggeration=48, # default is 12
        learning_rate=100, # default is 200
        verbose=2, # default is 0
    ).fit_transform(X)

    X_embedded__transposed = X_embedded.transpose()

    scatter_plot(X_embedded__transposed[0], X_embedded__transposed[1])

    df.insert(14, 'tsne_0', X_embedded__transposed[0], True)
    df.insert(15, 'tsne_1', X_embedded__transposed[1], True)

    df.to_csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne.csv', index=False)
