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

from utils.util_functions import prepare_data, evaluate_classifier, patient_lesion_classifier_no_sextant, generate_lesions
from models.SPCNet import SPCNet_all


def evaluate_per_patient(mode, path_dict, output_filepath, folds, lr, bx_lesions=False):
    """
    evaluate the SPCNet model per patient

    :param mode: has to be 'model' to evaluate the model, can be changed to other modes like 'radiologist' for evaluating radiologist outlines
    :param path_dict: path to T2, ADC, prostate masks, and cancer labels
    :param output_filepath: path where trained model is saved, and also where the predictions/evaluations are to be saved
    :param folds: Number of folds
    :param lr: learning rate used in training model
 
    :return: None
    """
    # aggregate results from each model and evaluate test set on a per lesion basis

    t2_list, adc_list, mask_list, label_list, volume_list, case_ids = prepare_data(path_dict, cancer_only=False)

    models = []
    
    # load trained models for each fold
    for fold_idx in range(folds):
        model = SPCNet_all().model_load(os.path.join(output_filepath, f'hed_fold_{fold_idx}.h5'), lr)
        models.append(model)

    volume_thresholds = [250]

    num_cases = 0

    # loop through volume thresholds
    for volume_thresh in volume_thresholds:

        total_model_true_grade = []
        total_model_pred_grade = []
        total_model_pred_grade_thresh = []

        # loop through cases
        for idx, case in enumerate(case_ids):
            volume = volume_list[idx]
            label_np = label_list[idx]
            mask_np = mask_list[idx]
           
            all_predictions_normal = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_ind = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))
            all_predictions_agg = np.zeros((label_np.shape[0], label_np.shape[1], label_np.shape[2], folds))

            num_cases += 1

            if mode == 'model':
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
                avg_prediction_ind = np.mean(all_predictions_ind, -1)
                avg_prediction_agg = np.mean(all_predictions_agg, -1)
            else:
                print ("Mode has to be model for model evaluation")

            avg_prediction_normal[mask_np == 0] = 0
            avg_prediction_ind[mask_np == 0] = 0
            avg_prediction_agg[mask_np == 0] = 0

            mod_prediction = SPCNet_all().return_mod_prediction(mask_np, avg_prediction_normal, avg_prediction_agg,
                                                        avg_prediction_ind)

            label_np[mask_np == 0] = 0
            label_np[mask_np == -1] = 3

            altered_pred = np.copy(avg_prediction_normal)
            altered_pred[mask_np > 0] = 1 - altered_pred[mask_np > 0]

            # generate lesions using morphological processing and volume thresholding
            lesions, num_lesions = generate_lesions(volume, label_np, volume_thresh)

            # classifier to determine if clinically significant lesion is detected or not
            normal_case = 'NP' in case
            model_true_grade, model_pred_grade = patient_lesion_classifier_no_sextant(volume, lesions, num_lesions,
                                                                                           avg_prediction_agg, label_np,
                                                                                           mask_np,
                                                                                           normal_case)
            agg_pred_thresh = np.where(mod_prediction == 2, 1, 0)
            _, model_pred_grade_thresh = patient_lesion_classifier_no_sextant(volume, lesions, num_lesions,
                                                                                   agg_pred_thresh,
                                                                                   label_np, mask_np, normal_case)
            if len(model_true_grade) > 0:
                total_model_true_grade.append(model_true_grade)
                total_model_pred_grade.append(model_pred_grade)
                total_model_pred_grade_thresh.append(model_pred_grade_thresh)

        if mode == 'model':
            if bx_lesions:
                title = 'test_model_bx'
            else:
                title = 'test_model_rp'
        else:
            print ("Mode has to be model for model evaluation")
        filepath = os.path.join(output_filepath, title)

        os.makedirs(filepath, exist_ok=True)

        # saves stats in dictionaries and saves in csv files
        stats = evaluate_classifier(total_model_true_grade, total_model_pred_grade,
                                         [SPCNet_all().agg_threshold])  # , pred_thresh=total_model_pred_thresh)
        total_stats = pd.DataFrame.from_dict(stats)
        total_stats.to_csv(os.path.join(filepath, f'{title}_patient_lesion_{volume_thresh}_5_per_no_sextant.csv'))

    return None
