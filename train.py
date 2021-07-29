# --------------------------------------------------------
# SPCNet training
# Copyright (c) 2021 PIMED@STanford
#
# Written by Arun Seetharaman
# --------------------------------------------------------
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from models.SPCNet import SPCNet_all
from utils.util_functions import create_folds, prepare_data, concatenate_data

def train(args):
  """
  Train SPCNet
  :param args: network parameters and configurations
  """
  
  model = SPCNet_all()

  path_dict = {'t2': args.t2_filepath,
               'adc': args.adc_filepath,
               'prostate': args.mask_filepath,
               'all_cancer': args.all_cancer_filepath,
               'agg_cancer': args.agg_cancer_filepath,
               'ind_cancer': args.ind_cancer_filepath}

  t2_list, adc_list, mask_list, label_list, _, _ = prepare_data(path_dict=path_dict, 
                                                                cancer_only=True)

  case_ids = np.arange(len(t2_list))
  splits = create_folds(case_ids, args.folds)

  for fold_idx, (train, test) in enumerate(splits):
      # renormalize each fold on its own
      # 
      t2_np, adc_np, y, stats = concatenate_data(t2_list, 
                                                adc_list, 
                                                label_list, 
                                                mask_list, 
                                                train, 
                                                fold_idx,
                                                args.output_filepath)

      x, y, _ = model.get_x_y(t2_np, adc_np, y)

      t2_val, adc_val, y_val, _ = concatenate_data(
                                                t2_list, 
                                                adc_list, 
                                                label_list, 
                                                mask_list, test, fold_idx,
                                                args.output_filepath, 
                                                stats)

      x_val, y_val, num_channels = model.get_x_y(t2_val, 
                                                adc_val, 
                                                y_val)

      validation_data = (x_val, y_val)

      # create the SPCNET model
      network = model.network(lr=args.lr, num_channels=num_channels)
      # train the model preprocessed data
      history = network.fit(x, 
                            y, 
                            args.batch_size, 
                            args.epochs, 
                            validation_data=validation_data, 
                            verbose=2)
      # Save model weights for each fold  
      network.save(os.path.join(args.output_filepath, f'hed_fold_{fold_idx}.h5'))
      # Plot training and validation loss for each fold.
      plt.figure()
      plt.plot(history.history['ofuse_loss'])
      if validation_data:
          plt.plot(history.history['val_ofuse_loss'])
      plt.title('ofuse_loss')
      if validation_data:
          plt.legend(['train', 'val'])
      plt.savefig(os.path.join(args.output_filepath, f'loss_{fold_idx}.png'))


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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
