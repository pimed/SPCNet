import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import interp
import SimpleITK as sitk
import tensorflow as tf
from skimage.morphology import disk, closing
from skimage.measure import label

from keras import backend as K
from sklearn.model_selection import KFold
from imutils import rotate

def prepare_data(path_dict, cancer_only=True, bx_lesions=False, rp_lesions=False, rad_lesions=False):
    """
    prepare data for training

    :param path_dict: path dictionary for all images and their correspoding labels
    :param cancer_only: weather to process only cancer label or not
    :param bx_lesions: bounding box for lesions
    :param rp_lesions: if the lesions are from RP cohort or not
    :param rad_lesions: if the lesions are from Rad cohort or not
    :return:
    """
    t2_path = path_dict['t2']
    adc_path = path_dict['adc']
    mask_path = path_dict['prostate']
    all_label_path = path_dict['all_cancer']
    ind_label_path = path_dict['ind_cancer']
    agg_label_path = path_dict['agg_cancer']

    t2_list = []
    adc_list = []
    mask_list = []
    label_list = []
    volume_list = []

    t2_orig_list = []
    adc_orig_list = []
    mask_orig_list = []
    label_orig_list = []
    volume_orig_list = []

    case_ids = [case.replace('_res_T2_hm.nii', '') for case in sorted(os.listdir(t2_path))]
    used_case_ids = []

    for case_id in case_ids:
        t2_file = case_id + '_res_T2_hm.nii'
        adc_file = case_id + '_res_ADC_hm.nii'
        mask_file = case_id + '_res_prostate_label.nii'
        all_label_file = case_id + '_res_cancer_label.nii'
        agg_label_file = case_id + '_res_agg_cancer_label.nii'
        ind_label_file = case_id + '_res_ind_cancer_label.nii'

        t2 = sitk.ReadImage(os.path.join(t2_path, t2_file))
        adc = sitk.ReadImage(os.path.join(adc_path, adc_file))
        # two different possible mask names
        try:
            mask = sitk.ReadImage(os.path.join(mask_path, mask_file))
        except:
            mask_file = case_id + '_res_mask_label.nii'
            mask = sitk.ReadImage(os.path.join(mask_path, mask_file))

        t2_np = sitk.GetArrayFromImage(t2)
        adc_np = sitk.GetArrayFromImage(adc)
        mask_np = sitk.GetArrayFromImage(mask)
        mask_np[mask_np > 0] = 1

        used_case_ids.append(case_id)

        if os.path.exists(os.path.join(all_label_path, all_label_file)):
            all_label = sitk.ReadImage(os.path.join(all_label_path, all_label_file))
            label_np = sitk.GetArrayFromImage(all_label).astype(
                'float32')  # ensures that negative values can be assigned to label_np
            label_np[label_np > 0] = 3  # sets pathologist labels to 3
        else:
            label_np = np.zeros_like(mask_np)  # if no label then normal case and whole prostate is 0

        if os.path.exists(os.path.join(ind_label_path, ind_label_file)):
            ind_label = sitk.ReadImage(os.path.join(ind_label_path, ind_label_file))
            ind_label_np = sitk.GetArrayFromImage(ind_label)
            label_np[ind_label_np > 0] = 1  # sets grade3 pixels to 1

        if os.path.exists(os.path.join(agg_label_path, agg_label_file)):
            agg_label = sitk.ReadImage(os.path.join(agg_label_path, agg_label_file))
            agg_label_np = sitk.GetArrayFromImage(agg_label)
            label_np[agg_label_np > 0] = 2  # sets grade4+ pixels to 2 even if there is overlap with grade3

        # replace label with radiologist lesion if necessary
        if rp_lesions and 'NP' not in case_id:
            label_np = get_rp_rad_lesion(mask, path_dict['rp_lesions'], case_id)

        # replace label with biopsy lesion if necessary
        if bx_lesions and 'NP' not in case_id:
            bx_case, label_np = get_bx_from_rp(path_dict, case_id, mask, keep_grade=True)

        if rad_lesions:
            rad_label_path = path_dict['radiologist_labels']
            label_file = case_id + '_radiologist_label.nii'
            rad_label = sitk.ReadImage(os.path.join(rad_label_path, label_file))
            label_np = sitk.GetArrayFromImage(rad_label).astype('float32')

        label_orig_list.append(label_np)
        t2_orig_list.append(t2_np)
        adc_orig_list.append(adc_np)
        mask_orig_list.append(mask_np)
        volume_orig_list.append(mask)

    for t2_np, adc_np, mask_np, label_np, volume, case_id in zip(t2_orig_list, adc_orig_list, mask_orig_list,
                                                                 label_orig_list, volume_orig_list, used_case_ids):
        if cancer_only and 'NP' not in case_id and label_np is not None:
            coor = np.nonzero(label_np)[0]
        else:
            coor = np.nonzero(mask_np)[0]

        cancer = np.unique(coor)

        mask_np = mask_np[cancer, :, :]

        if label_np is not None:
            label_np = label_np[cancer, :, :]
            label_np[mask_np == 0] = -1  # sets pixels outside the prostate as -1 so model does not train on them

        t2_actual_slice = t2_np[cancer, :, :, np.newaxis]
        t2_before_slice = np.zeros_like(t2_actual_slice)
        t2_after_slice = np.zeros_like(t2_actual_slice)

        adc_actual_slice = adc_np[cancer, :, :, np.newaxis]
        adc_before_slice = np.zeros_like(adc_actual_slice)
        adc_after_slice = np.zeros_like(adc_actual_slice)

        for index, slice_num in enumerate(cancer):
            if slice_num == 0:
                t2_before_slice[index, :, :, :] = t2_actual_slice[index, :, :, :]
                adc_before_slice[index, :, :, :] = adc_actual_slice[index, :, :, :]
            else:
                t2_before_slice[index, :, :, 0] = t2_np[slice_num - 1, :, :]
                adc_before_slice[index, :, :, 0] = adc_np[slice_num - 1, :, :]
            if slice_num == t2_np.shape[0] - 1:
                t2_after_slice[index, :, :, :] = t2_actual_slice[index, :, :, :]
                adc_after_slice[index, :, :, :] = adc_actual_slice[index, :, :, :]
            else:
                t2_after_slice[index, :, :, 0] = t2_np[slice_num + 1, :, :]
                adc_after_slice[index, :, :, 0] = adc_np[slice_num + 1, :, :]

        stacked_t2 = np.concatenate((t2_before_slice, t2_actual_slice, t2_after_slice), -1)
        stacked_adc = np.concatenate((adc_before_slice, adc_actual_slice, adc_after_slice), -1)

        label_list.append(label_np)
        t2_list.append(stacked_t2)
        adc_list.append(stacked_adc)
        mask_list.append(mask_np)
        volume_list.append(volume)

    return t2_list, adc_list, mask_list, label_list, volume_list, used_case_ids


def concatenate_data(t2_list, adc_list, label_list, mask_list, index, fold, filepath, stats=None):
    """
    make data ready for training, including data augmentation

    :param t2_list: list of T2-weighted images
    :param adc_list: list of ADC images
    :param label_list: list of cancer labels
    :param mask_list: list of mask label
    :param index:
    :param fold: which fold to run
    :param filepath: the path to save std and mean of the training data into npy
    :param stats: check if the mean and std of training set is avilable or not.
    :return:
    """
    def myrotate(im, angle):
        try:
            sub_ims = []
            L = im.shape[0]
            for i in range(L):
                sub_ims.append(np.expand_dims(rotate(im[i], angle), axis=0))
            return np.concatenate(sub_ims, axis=0)
        except:
            return im

    angles = [-15, -10, -5, 5, 10, 15]
    num_angles = 1

    aug_list_t2 = []
    aug_list_adc = []
    aug_list_mask = []
    aug_list_label = []

    mask_list = [mask_list[i] for i in index]
    t2_list = [t2_list[i] for i in index]
    adc_list = [adc_list[i] for i in index]
    label_list = [label_list[i] for i in index]

    for t2, adc, mask, label in zip(t2_list, adc_list, mask_list, label_list):
        chosen_angles = random.sample(angles, num_angles)
        for a in chosen_angles:
            aug_list_t2.append(myrotate(t2, a))
            aug_list_adc.append(myrotate(adc, a))
            aug_list_mask.append(myrotate(mask, a))
            aug_list_label.append(myrotate(label, a))

    t2_list += aug_list_t2
    adc_list += aug_list_adc
    mask_list += aug_list_mask
    label_list += aug_list_label

    mask = np.concatenate(mask_list)
    t2 = np.concatenate(t2_list)
    adc = np.concatenate(adc_list)
    y = np.concatenate(label_list)

    if stats:
        t2_mean, t2_std, adc_mean, adc_std = stats
    else:
        t2_mean = np.mean(t2[mask > 0])
        t2_std = np.std(t2[mask > 0])
        adc_mean = np.mean(adc[mask > 0])
        adc_std = np.std(adc[mask > 0])
        stats = (t2_mean, t2_std, adc_mean, adc_std)
        np.save(os.path.join(filepath, f'stats_{fold}.npy'), stats)

    t2 = (t2 - t2_mean) / t2_std
    adc = (adc - adc_mean) / adc_std
    y = y[:, :, :, np.newaxis]

    t2 = np.concatenate((t2, t2[:, :, ::-1, :]))
    adc = np.concatenate((adc, adc[:, :, ::-1, :]))
    y = np.concatenate((y, y[:, :, ::-1, :]))

    return t2, adc, y, stats


def create_folds(case_ids, folds):
    """
    creates the folds for 5-fold cross validation
    :param case_ids: list of cases
    :param folds: number of folds
    :return: train and test splits list
    """
    splits = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    fold_splits = kf.split(case_ids)
    for train, test in fold_splits:
        splits.append((train, test))

    return splits

def export_volume(filepath, ref_volume, np_volume, volume_title, scale=255):
    """
    given a numpy array and reference volume, save volume
    might need original label as well
    assume np_volume corresponds to slices of mask with prostate in it

    :param filepath: the file path to store exported volume
    :param ref_volume: reference volume to use for pixel spacing information
    :param np_volume: predicted segmentation for lesions
    :param volume_title: name of the volume to be stored
    :param scalee: weather to scale intensity values to 255 or not
    :return: None
    """

    ref_vol_np = sitk.GetArrayFromImage(ref_volume)
    export_vol_np = np.zeros(ref_vol_np.shape)

    coor = np.nonzero(ref_vol_np)
    prostate = np.unique(coor[0])

    export_vol_np[prostate, :, :] = scale * np_volume

    export_vol = sitk.GetImageFromArray(export_vol_np)
    export_vol.SetSpacing(ref_volume.GetSpacing())
    export_vol.SetDirection(ref_volume.GetDirection())
    export_vol.SetOrigin(ref_volume.GetOrigin())

    sitk.WriteImage(export_vol, os.path.join(filepath, volume_title))

    return None

def evaluate_classifier(total_true, total_pred, thresholds=None, title=None, pred_thresh=None):
    """
    draws roc_curve, computes auc, sensitivity, specificity, accuracy and number of negative samples

    :param total_true: ground truth
    :param total_pred: total number of prediction
    :param thresholds: thereshold value to clip the values
    :param title: 
    :param pred_thresh: 
    :return: stats for different metrics to evaluate the model performance.
    """

    #-------------------------------------------------------
    # IB: not sure what is the function of pred_thresh and how that is diffrent from thresholds
    # IB: not sure why you choose thresh as thresholds[0]
    #-----------------------------------------------------------
    total_true = np.concatenate(total_true)
    total_pred = np.concatenate(total_pred)
   
    mean_fpr, mean_tpr, per_pixel_auc = draw_roc_curve([total_true], [total_pred], title)
    print (mean_fpr, mean_tpr, per_pixel_auc)
    thresh = thresholds[0]
    print (thresh)
    if pred_thresh is not None:
        pred = np.concatenate(pred_thresh)
    else:
        pred = total_pred

    true = total_true

    true_positives = np.sum(true[pred >= thresh])
    total_positives = np.sum(true)
    per_pixel_sensitivity = true_positives / total_positives

    true_negatives = np.sum(1 - true[pred < thresh])
    total_negatives = np.sum(1 - true)
    per_pixel_specificity = true_negatives / total_negatives

    per_pixel_accuracy = (true_positives + true_negatives) / (total_positives + total_negatives)

    thresh_pred_dice = np.zeros_like(pred)
    thresh_pred_dice[pred >= thresh] = 1

    stats = {'Per Pixel ROC AUC': [[per_pixel_auc]],
             'Per Pixel Accuracy': per_pixel_accuracy,
             'Per Pixel Sensitivity': per_pixel_sensitivity,
             'Per Pixel Specificity': per_pixel_specificity,
             'Num Negatives': np.sum(1 - total_true)}

    return stats

def draw_roc_curve(total_true, total_pred, threshold=False):
    """
    draws AUC-ROC curve
    :param total_true: ground truth
    :param total_pred: total number of prediction

    """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    num_curves = len(total_true)

    plt.figure()

    for curve in range(num_curves):
        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = metrics.roc_curve(total_true[curve].flatten(), total_pred[curve].flatten())

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (curve, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    if threshold:
        return mean_fpr, mean_tpr, mean_auc, thresholds
    else:
        return mean_fpr, mean_tpr, mean_auc

def generate_lesions(ref_vol, label_np_source, volume_thresh):
    """
    computes connected 3D volumes from pixel-level labels
    Uses morphological processing using a 3D structuring element, and thresholding smaller connected components

    :param ref_vol: reference image volume
    :param label_np_source: refined predicted lesions
    
    """
    label_np = np.copy(label_np_source)

    label_np[label_np > 0] = 1
    label_np[label_np <= 0] = 0

    spacing = ref_vol.GetSpacing()

    margin = 2.5 / spacing[0]
    strel = disk(int(margin))

    margin = .5 / spacing[0]
    strel2 = disk(int(margin))
    strel2 = np.pad(strel2, 7, 'constant')

    strel_total = np.stack([strel2, strel, strel2])

    connected = closing(label_np, strel_total)

    lesions, num_lesions = label(connected, return_num=True, connectivity=2)

    for lesion in range(num_lesions):
        if spacing[0] * spacing[1] * spacing[2] * np.sum(lesions == (lesion + 1)) < volume_thresh:
            lesions[lesions == (lesion + 1)] = 0

    lesions_mask = np.copy(lesions)
    lesions_mask[lesions_mask > 0] = 1
    lesions, num_lesions = label(lesions_mask, return_num=True, connectivity=2)

    return lesions, num_lesions

def findTPFN(mask_np, pred_np, label_np):
    """
    finds the true positives and false negatives for sextant based lesion-level evaluation
    """
    list_of_lesions = np.unique(label_np)[1:]
    num_lesions = len(list_of_lesions)
    indiv_lesion_np = np.zeros((num_lesions, np.shape(label_np)[0], np.shape(label_np)[1], np.shape(label_np)[2]),
                               dtype=int)

    thresh = .5

    pred_90percent = np.zeros((num_lesions), dtype=float)
    lesion_classifier = np.zeros((num_lesions), dtype=int)
    for k in range(0, num_lesions):
        lesion = list_of_lesions[k]
        copy_lesion_label = np.copy(label_np)
        copy_lesion_label[label_np != lesion] = 0
        copy_lesion_label[copy_lesion_label > 0] = 1
        indiv_lesion_np[k, :, :, :] = copy_lesion_label
        pred_les_pix = pred_np[indiv_lesion_np[k, :, :, :] > 0]
        pred_90percent[k] = np.percentile(pred_les_pix, 90)

        if pred_90percent[k] >= thresh:
            lesion_classifier[k] = 1
    detected_lesions = np.sum(lesion_classifier)  # true positive
    missed_lesions = num_lesions - detected_lesions  # false negative

    GT_pos = np.ones((num_lesions), dtype=float)
    pred_pos = pred_90percent
    return num_lesions, detected_lesions, missed_lesions, GT_pos, pred_pos

def findTNFP(ref_vol, mask_np, pred_np, label_np, removesa3D):
    """
    finds true negatives and false positives for sextant based lesion level evaluation
    """
    GT_neg = []
    pred_neg = []

    coor = np.nonzero(mask_np)
    prostate = np.unique(coor[0])
    prostate_regions = np.array_split(mask_np[prostate, :, :], 3, axis=0)
    label_regions = np.array_split(label_np[prostate, :, :], 3, axis=0)
    pred_regions = np.array_split(pred_np[prostate, :, :], 3, axis=0)

    for prostate_region, label_region, pred_region in zip(prostate_regions, label_regions, pred_regions):
        if prostate_region.size == 0:
            continue
        prostate_left_right = np.array_split(prostate_region, 2, axis=2)
        label_left_right = np.array_split(label_region, 2, axis=2)
        pred_left_right = np.array_split(pred_region, 2, axis=2)

        for prostate_half, label_half, pred_half in zip(prostate_left_right, label_left_right, pred_left_right):
            if (np.sum(np.logical_and(label_half > 0, prostate_half > 0)) / np.sum(prostate_half > 0)) <= 0:
                GT_neg.append(0)
                normal_tissue = np.logical_and(label_half == 0, prostate_half > 0)
                pred_neg.append(np.percentile(pred_half[normal_tissue], 90))

    return GT_neg, pred_neg


def lesion_classifier(ref_vol, lesion_label_np, pred_np, mask_np):
    """
    finds the GT and predictions for sextant based lesion level evaluation

    :param ref_vol: reference image volume
    :param label_np_source: ground truth labels
    :param label_np_source: model predictions
    :param label_np_source: segmentation mask
    
    """

    removesa3D = False

    num_lesions, len_TP, len_FN, GT_pos, pred_pos = findTPFN(mask_np, pred_np, lesion_label_np)
    GT_neg, pred_neg = findTNFP(ref_vol, mask_np, pred_np, lesion_label_np, removesa3D)
    true = np.concatenate((GT_pos, GT_neg))
    pred = np.concatenate((pred_pos, pred_neg))

    return true, pred

def grade_lesion_classifier(ref_vol, lesions, num_lesions, agg_pred_np, label_np, mask_np, normal_case=False):
    # classifies a lesion into aggressive/indolent based on deep_bio_pixels (digitial pathology labels)
    agg_ratio_thresh = .01

    
    new_lesions = np.copy(lesions)

    deep_bio_pixels = label_np[np.logical_and(np.logical_or(label_np == 2, label_np == 1), lesions > 0)]

    if np.sum(deep_bio_pixels) > 0:
        for lesion in range(1, num_lesions+1):
            lesion_pixels = label_np[lesions == lesion]
            agg_ratio = np.sum(lesion_pixels == 2) / lesion_pixels.size

            if agg_ratio <= agg_ratio_thresh:
                new_lesions[lesions == lesion] = 0

        true, pred = lesion_classifier(ref_vol, new_lesions, agg_pred_np, mask_np)
    elif normal_case:  # will evaluate normal cases without needing deep bio pixels
        true, pred = lesion_classifier(ref_vol, new_lesions, agg_pred_np, mask_np)
    else:
        true = np.array([])
        pred = np.array([])

    return true, pred

def patient_lesion_classifier_no_sextant(ref_vol, lesions, num_lesions, agg_pred_np, label_np, mask_np, normal_case=False):

    agg_ratio_thresh = .05

    # might want to focus only on pixels defined by deep bio
    new_lesions = np.copy(lesions)

    deep_bio_pixels = label_np[np.logical_and(np.logical_or(label_np == 2, label_np == 1), lesions > 0)]

    if np.sum(deep_bio_pixels) > 0:
        for lesion in range(1, num_lesions+1):
            lesion_pixels = label_np[lesions == lesion]
            agg_ratio = np.sum(lesion_pixels == 2) / lesion_pixels.size

            if agg_ratio <= agg_ratio_thresh:
                new_lesions[lesions == lesion] = 0

        true, pred = lesion_classifier(ref_vol, new_lesions, agg_pred_np, mask_np)

        if np.sum(true) > 0:
            # want to determine if model found one of the clinically significant lesions
            pred = pred[true == 1]
            true = np.array([1])
            pred = np.array([np.max(pred)])
        else:
            true = np.array([])
            pred = np.array([])

    elif normal_case:  # will evaluate normal cases without needing deep bio pixels
        pred_thresh = np.where(agg_pred_np > .2, 1, 0)
        pred_lesions, num_pred_lesions = generate_lesions(ref_vol, pred_thresh, 250)
        true = np.array([0])
        if num_pred_lesions > 0:
            pred = np.array([1])
        else:
            pred = np.array([0])
    else:
        true = np.array([])
        pred = np.array([])

    return true, pred
