import pandas as pd
import numpy as np



# def normalize__csv(input_filename, output_filename):

#     df = pd.read_csv(input_filename, header=0)

#     df.drop('gene_description', axis=1, inplace=True)

#     for col in df.columns[2:]:
#         df[col] = df[col] * (1/16) + 0.5

#     df.to_csv(output_filename, index=False)



def denormalize__csv(input_filename, output_filename):

    df = pd.read_csv(input_filename, header=0)

    for col in df.columns[2:12]:
        df[col] = (df[col] - 0.5) * 16

    df.to_csv(output_filename, index=False)



def normalize__csv(input_filename, output_filename):

    df = pd.read_csv(input_filename, header=0)

    df.drop('gene_description', axis=1, inplace=True)

    for col in df.columns[2:]:
        df[col] = df[col] * (1/16) + 0.5

    df.to_csv(output_filename, index=False)



if __name__ == '__main__':

    df = pd.read_csv('initial_data/timepoint_organized_deg__2.csv', header=0)

    df.drop('gene_description', axis=1, inplace=True)

    for col in df.columns[2:]:
        df[col] = df[col] * (1/16) + 0.5

    df.to_csv('transitional_data/rnaseq_data__genes_x_timepoints__normalized.csv', index=False)
