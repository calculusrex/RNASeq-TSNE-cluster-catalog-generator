import os
import time

from normalize__ import normalize__csv
from gene_ontology__corelation__ import go_label__csv
from tsne__ import compute_tsne__csv
from cluster__ import cluster__csv__dual_layer
from cluster_catalog__ import generate_cluster_catalog
# from cluster_poster__ import generate_cluster_poster


# HOMEWORK: Add columns of derived features from the 10 timepoints.


if __name__ == '__main__':

    intermediate_data__foldername = 'rnaseq_process__{}'.format(time.time())
    os.mkdir(intermediate_data__foldername)

    print('Normalizing data ....')
    normalize__csv('initial_data/timepoint_organized_deg__2.csv',
                   '{}/rnaseq_data__genes_x_timepoints__normalized.csv'.format(intermediate_data__foldername))

    print('Associating Gene Ontology categories and descriptions ....')
    go_label__csv('{}/rnaseq_data__genes_x_timepoints__normalized.csv'.format(intermediate_data__foldername),
                  '{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled.csv'.format(intermediate_data__foldername))

    print('Computing t-distributed Stochastic Neighbor Embedding (TSNE) ....')
    compute_tsne__csv('{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled.csv'.format(intermediate_data__foldername),
                      '{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne.csv'.format(intermediate_data__foldername))

    print('Clustering ....')
    cluster__csv__dual_layer('{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne.csv'.format(intermediate_data__foldername),
                             '{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__clstrd.csv'.format(intermediate_data__foldername))

    generate_cluster_catalog('{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__agglomerative_clstrd_lvl3.csv'.format(intermediate_data__foldername))

    # generate_cluster_poster('{}/rnaseq_data__genes_x_timepoints__normalized__go_labeled__tsne__agglomerative_clstrd_lvl3.csv'.format(intermediate_data__foldername))
