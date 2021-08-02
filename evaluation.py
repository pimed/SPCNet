# --------------------------------------------------------
# SPCNNet evaluation
# Copyright (c) 2021 PIMED@Stanford
#
# Written by Arun Seetharaman
# --------------------------------------------------------
import argparse

from evaluation_types.per_pixel import *
from evaluation_types.per_lesion import *
from evaluation_types.per_patient import *


def main(args):
    """
    Run SPCNet on the test data and evaluate the prediction:
    - per pixel
    - per lesion
    - per patient
    - per pixel evaluation also allows saving the model predictions

    :param args: network parameters and configurations
    """

    model = SPCNet()
    path_dict = {'t2': args.t2_filepath,
                 'adc': args.adc_filepath,
                 'prostate': args.mask_filepath,
                 'all_cancer': args.all_cancer_filepath,
                 'agg_cancer': args.agg_cancer_filepath,
                 'ind_cancer': args.ind_cancer_filepath}

    # pixel-level evaluation
    print("Pixel-level evaluation of SPCNet")
    evaluate_per_pixel('model',
                       path_dict,
                       args.output_filepath,
                       args.folds,
                       args.lr)

    # lesion-level evaluation
    print("Lesion-level evaluation of SPCNet")
    evaluate_per_lesion('model',
                        path_dict,
                        args.output_filepath,
                        args.folds,
                        args.lr)

    print("Patient-level evaluation of SPCNet")
    evaluate_per_patient('model',
                         path_dict,
                         args.output_filepath,
                         args.folds,
                         args.lr)


if __name__ == "__main__":
    # parse parameters
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--t2_filepath', type=str, required=True)
    parser.add_argument('--adc_filepath', type=str, required=True)
    parser.add_argument('--mask_filepath', type=str, required=True)
    parser.add_argument('--all_cancer_filepath', type=str, required=True)
    parser.add_argument('--agg_cancer_filepath', type=str, required=True)
    parser.add_argument('--ind_cancer_filepath', type=str, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
