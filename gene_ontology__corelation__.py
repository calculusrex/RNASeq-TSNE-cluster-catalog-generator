import pandas as pd
import numpy as np


def go_label__csv(input_filename, output_filename):

    df__go = pd.read_csv('initial_data/expanded_GO_enrichment_data.csv', header=0)

    df = pd.read_csv(input_filename, header=0)

    go_genes = np.array(df__go['geneID'])

    category_column__list = []
    description_column__list = []

    for barcode in list(df['barcode']):
        if barcode in go_genes:
            row = df__go[df__go['geneID'] == barcode].iloc[0]
            category_column__list.append(
                row['Category'])
            description_column__list.append(
                row['Description'])
        else:
            category_column__list.append('not labeled')
            description_column__list.append('not labeled')

    df.insert(12, 'go_category', category_column__list, True)
    df.insert(13, 'go_description', description_column__list, True)

    df.to_csv(output_filename, index=False)
    return df


if __name__ == '__main__':


    df = pd.read_csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized.csv', header=0)


    df__go = pd.read_csv('initial_data/expanded_GO_enrichment_data.csv', header=0)


    go_genes = np.array(df__go['geneID'])

    category_column__list = []
    description_column__list = []

    for barcode in list(df['barcode']):
        if barcode in go_genes:
            row = df__go[df__go['geneID'] == barcode].iloc[0]
            category_column__list.append(
                row['Category'])
            description_column__list.append(
                row['Description'])
        else:
            category_column__list.append('not labeled')
            description_column__list.append('not labeled')

    df.insert(12, 'go_category', category_column__list, True)
    df.insert(13, 'go_description', description_column__list, True)

    # df.to_csv(output_filename, index=False)



    # df = go_label__csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized.csv', 'transitional_data/rnaseq_data__genes_x_timepoints__normalized__go_labeled.csv')

