# --------------------------------------------------------
# Code to compute dataset statistics for class balancing
# Copyright (c) 2021 PIMED@STanford
#
# Written by Arun Seetharaman
# --------------------------------------------------------

import argparse
import numpy as np

from utils.util_functions import prepare_data


def compute_pixel_amounts(args):
    """Compute the number of pixels in each class to be used in the balanced cross entropy loss function.
    :param args: network parameters and configurations
    """
    path_dict = {'t2': args.t2_filepath,
                 'adc': args.adc_filepath,
                 'prostate': args.mask_filepath,
                 'all_cancer': args.all_cancer_filepath,
                 'agg_cancer': args.agg_cancer_filepath,
                 'ind_cancer': args.ind_cancer_filepath}

    _, _, mask_list, label_list, _, _ = prepare_data(path_dict=path_dict, cancer_only=True)

    label_values = [label_np[mask_np > 0] for label_np, mask_np in zip(label_list, mask_list)]
    label_values = np.concatenate(label_values)
    elements, counts = np.unique(label_values, return_counts=True)
    num_normal = counts[0]
    num_agg = counts[1] + .5 * counts[3]
    num_ind = counts[2] + .5 * counts[3]
    print(f'Normal pixels: {num_normal}')
    print(f'Aggressive pixels: {num_agg}')
    print(f'Indolent pixels: {num_ind}')


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--t2_filepath', type=str, required=True)
    parser.add_argument('--adc_filepath', type=str, required=True)
    parser.add_argument('--mask_filepath', type=str, required=True)
    parser.add_argument('--all_cancer_filepath', type=str, required=True)
    parser.add_argument('--agg_cancer_filepath', type=str, required=True)
    parser.add_argument('--ind_cancer_filepath', type=str, required=True)
    args = parser.parse_args()

    compute_pixel_amounts(args)
