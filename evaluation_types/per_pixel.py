# --------------------------------------------------------
# SPCNNet training
# Copyright (c) 2021 PIMED@Stanford
#
# Written by Arun Seetharaman
# --------------------------------------------------------

import argparse
import os
import numpy as np
import pandas as pd

from utils.util_functions import prepare_data, export_volume, evaluate_classifier
from models.SPCNet import SPCNet_all


def evaluate_per_pixel(mode, path_dict, output_filepath, folds, lr, bx_lesions=False, save_preds = True):
    """
    saves model predictions
    evaluate the SPCNet model per pixel

    :param mode: has to be 'model' to evaluate the model, can be changed to other modes like 'radiologist' for evaluating radiologist outlines
    :param path_dict: path to T2, ADC, prostate masks, and cancer labels
    :param output_filepath: path where trained model is saved, and also where the predictions/evaluations are to be saved
    :param folds: Number of folds
    :param lr: learning rate used in training model
    :param save_preds: True if model predictions are to be saved, false otherwise
    :return: None
    """
    # aggregates the results from each model to evaluate test set on a pixel basis
    
    # prepares data for model
    t2_list, adc_list, mask_list, label_list, volume_list, case_ids = prepare_data(path_dict, 
                                                                                   cancer_only=False,
                                                                                   bx_lesions=bx_lesions)
    print ("length of t2_list, label_list, and number of cases:", len(t2_list), len(label_list), len(case_ids))
    
    models = []
    if bx_lesions:
        subdirectory = 'bx_pred' # for biopsy cohort patients 
    else:
        subdirectory = 'rp_pred' # for radical prostatectomy patients
    
    # make subdirectory to save results if they do not exist already
    os.makedirs(os.path.join(output_filepath, subdirectory), exist_ok=True)
 
    if mode == 'model':
        # load model trained for each cross-validation fold
        for fold_idx in range(folds):
            model = SPCNet_all().model_load(os.path.join(output_filepath, f'hed_fold_{fold_idx}.h5'), lr)
            models.append(model)
    else:
        print ("Mode has to be model to save model predictions!")
    
    if save_preds == True:
        # save probability maps and thresholded labels
        # saved probability maps include normal, indolent and aggressive probabilities
        # saved predicted labels include maximum labels and mod labels generated using thresholds decided on training set
        for idx, case in enumerate(case_ids):
            print (case)
            volume = volume_list[idx]
            label_np = label_list[idx]
            mask_np = mask_list[idx]

            all_predictions_normal = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_ind = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_agg = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))

            for i in range(folds):
                stats = np.load(os.path.join(output_filepath, f'stats_{i}.npy'))
                t2_mean, t2_std, adc_mean, adc_std = stats

                t2_np = t2_list[idx]
                adc_np = adc_list[idx]

                t2_np = (t2_np - t2_mean) / t2_std
                adc_np = (adc_np - adc_mean) / adc_std
                x, _, _ = SPCNet_all().get_x_y(t2_np, adc_np)
                prediction = SPCNet_all().model_prediction(models[i], x)
                prediction_normal = prediction[:, :, :, 0]
                prediction_normal[mask_np == 0] = 0
                prediction_ind = prediction[:, :, :, 1]
                prediction_ind[mask_np == 0] = 0
                prediction_agg = prediction[:, :, :, 2]
                prediction_agg[mask_np == 0] = 0

                all_predictions_normal[:, :, :, i] = np.squeeze(prediction_normal)
                all_predictions_ind[:, :, :, i] = np.squeeze(prediction_ind)
                all_predictions_agg[:, :, :, i] = np.squeeze(prediction_agg)

            avg_prediction_normal = np.mean(all_predictions_normal, -1)
            export_volume(output_filepath, volume, avg_prediction_normal, os.path.join(subdirectory, f'{case}_avg_norm_pred.nii'))

            avg_prediction_ind = np.mean(all_predictions_ind, -1)
            export_volume(output_filepath, volume, avg_prediction_ind, os.path.join(subdirectory, f'{case}_avg_ind_pred.nii'))

            avg_prediction_agg = np.mean(all_predictions_agg, -1)
            export_volume(output_filepath, volume, avg_prediction_agg, os.path.join(subdirectory, f'{case}_avg_agg_pred.nii'))

            avg_prediction = np.stack([avg_prediction_normal, avg_prediction_ind, avg_prediction_agg], axis=3)
            max_prediction = np.argmax(avg_prediction, axis=3)
            export_volume(output_filepath, volume, max_prediction, os.path.join(subdirectory, f'{case}_max_label.nii'))

            mod_prediction = SPCNet_all().return_mod_prediction(mask_np, avg_prediction_normal, avg_prediction_agg,
                                                        avg_prediction_ind)
            export_volume(output_filepath, volume, mod_prediction, os.path.join(subdirectory, f'{case}_mod_label.nii'))
    
    total_pred_normal = []
    total_pred_ind = []
    total_pred_agg = []

    total_pred_normal_thresh = []
    total_pred_ind_thresh = []
    total_pred_agg_thresh = []

    total_true_normal = []
    total_true_ind = []
    total_true_agg = []

    # evaluate model on a per-pixel basis
    for idx, case in enumerate(case_ids):
        label_np = label_list[idx]
        #radiologist_label_np = radiologist_label_list[idx]
        mask_np = mask_list[idx]

        if mode == 'model':
            all_predictions_normal = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_ind = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_agg = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))

            for i in range(folds):
                stats = np.load(os.path.join(output_filepath, f'stats_{i}.npy'))
                t2_mean, t2_std, adc_mean, adc_std = stats

                t2_np = t2_list[idx]
                adc_np = adc_list[idx]

                t2_np = (t2_np - t2_mean) / t2_std
                adc_np = (adc_np - adc_mean) / adc_std
                x, _, _ = SPCNet_all().get_x_y(t2_np, adc_np)

                pred = SPCNet_all().model_prediction(models[i], x)
                label_np = label_np.squeeze()

                all_predictions_normal[:, :, :, i] = pred[:, :, :, 0]
                all_predictions_ind[:, :, :, i] = pred[:, :, :, 1]
                all_predictions_agg[:, :, :, i] = pred[:, :, :, 2]

            pred_normal = np.mean(all_predictions_normal, -1)
            pred_ind = np.mean(all_predictions_ind, -1)
            pred_agg = np.mean(all_predictions_agg, -1)
        else:
            print ("Mode has to be model for evaluating model")
      
        label_np = label_np[mask_np > 0]

        true_normal = label_np[label_np > -1]
        true_normal[true_normal > 0] = 1
        true_normal = 1 - true_normal
        true_ind = label_np[np.logical_and(label_np > -1, label_np < 3)]
        true_ind[true_ind == 2] = 0
        true_agg = label_np[np.logical_and(label_np > -1, label_np < 3)]
        true_agg[true_agg == 1] = 0
        true_agg[true_agg == 2] = 1

        mod_prediction = SPCNet_all().return_mod_prediction(mask_np, pred_normal, pred_agg, pred_ind)
        mod_prediction = mod_prediction[mask_np > 0]

        pred_agg = pred_agg[mask_np > 0]
        pred_agg_thresh = np.zeros_like(pred_agg)
        pred_agg_thresh[mod_prediction == 2] = 1
        pred_agg_thresh = pred_agg_thresh[np.logical_and(label_np > -1, label_np < 3)]
        pred_agg = pred_agg[np.logical_and(label_np > -1, label_np < 3)]

        pred_ind = pred_ind[mask_np > 0]
        pred_ind_thresh = np.zeros_like(pred_ind)
        pred_ind_thresh[mod_prediction == 1] = 1
        pred_ind_thresh = pred_ind_thresh[np.logical_and(label_np > -1, label_np < 3)]
        pred_ind = pred_ind[np.logical_and(label_np > -1, label_np < 3)]

        pred_normal = pred_normal[mask_np > 0]
        pred_normal_thresh = np.zeros_like(pred_normal)
        pred_normal_thresh[mod_prediction == 0] = 1
        pred_normal_thresh = pred_normal_thresh[label_np > -1]
        pred_normal = pred_normal[label_np > -1]

        total_true_normal.append(true_normal)
        total_true_ind.append(true_ind)
        total_true_agg.append(true_agg)

        total_pred_normal.append(pred_normal)
        total_pred_ind.append(pred_ind)
        total_pred_agg.append(pred_agg)

        total_pred_normal_thresh.append(pred_normal_thresh)
        total_pred_ind_thresh.append(pred_ind_thresh)
        total_pred_agg_thresh.append(pred_agg_thresh)
        
    print (SPCNet_all().normal_threshold)
    # evaluate each class and get a dictionary of stats to be saved as a csv file
    print (len(total_true_normal), len(total_pred_normal))
    stats_normal = evaluate_classifier(total_true_normal, total_pred_normal, [SPCNet_all().normal_threshold])  # ,
    # pred_thresh=total_pred_normal_thresh)
    total_stats_normal = pd.DataFrame.from_dict(stats_normal)

    stats_ind = evaluate_classifier(total_true_ind, total_pred_ind, [SPCNet_all().ind_threshold])  # ,
    # pred_thresh=total_pred_ind_thresh)
    total_stats_ind = pd.DataFrame.from_dict(stats_ind)

    stats_agg = evaluate_classifier(total_true_agg, total_pred_agg, [SPCNet_all().agg_threshold])  # ,
    # pred_thresh=total_pred_agg_thresh)
    total_stats_agg = pd.DataFrame.from_dict(stats_agg)

    if mode == 'model':
        if bx_lesions:
            title = 'test_model_bx'
        else:
            title = 'test_model_rp'
    else:
        print ("Mode has to be model for evaluating model")
    
    output_filepath = os.path.join(output_filepath, title)
    os.makedirs(output_filepath, exist_ok=True)
    
    # aggregates per-pixel evaluation results for all cases and saves the results in csv files
    total_stats_normal.to_csv(os.path.join(output_filepath, f'{title}_stats_normal.csv'))
    total_stats_ind.to_csv(os.path.join(output_filepath, f'{title}_stats_ind.csv'))
    total_stats_agg.to_csv(os.path.join(output_filepath, f'{title}_stats_agg.csv'))

