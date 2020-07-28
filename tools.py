from datetime import datetime
import numpy as np
import os
import cv2
import tensorflow as tf


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)

def makedirs(path):
    """ Try to create the directory
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_max_preds(heatmaps):
    num_joints = heatmaps.shape[2]
    width = heatmaps.shape[1]
    heatmaps = np.transpose(heatmaps, (2, 0, 1))
    heatmaps_reshaped = np.reshape(heatmaps, (1, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = np.reshape(maxvals, (1, num_joints, 1))
    idx = np.reshape(idx, (1, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxvals


def visualize_gt_and_output(background, heatmaps, output_heatmaps, flags, delay=1000, draw_keypoint=False):
    heatmap_width  = heatmaps.shape[1]
    heatmap_height = heatmaps.shape[0]
    
    # H, W, C = [96, 96, 14]
    row1_image = np.zeros((2 * heatmap_height, 7 * heatmap_width, 3))
    row2_image = np.zeros((2 * heatmap_height, 7 * heatmap_width, 3))
    resized_background = cv2.resize(background, (heatmap_width, heatmap_height)).astype(np.float32)
   
    # get keypoint prediction from heatmap
    preds, maxvals = get_max_preds(output_heatmaps)
 
    # row 1
    for i in range(7):
        width_begin = heatmap_width * i
        width_end   = heatmap_width * (i+1)
        
        heatmap = (heatmaps[:, :, i] * 255) * flags[i]
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype(np.float32)

        if draw_keypoint:
            if flags[i] > 0 and maxvals[0][i][0] > 0.3:
                cv2.circle(background, (int(preds[0][i][0] * 2.0), int(preds[0][i][1] * 2.0)), 2, (0, 0, 255), -1)

        masked_image = colored_heatmap * 0.7 + resized_background * 0.3
        row1_image[:heatmap_height, width_begin:width_end, :] = masked_image
    
        output_heatmap = (output_heatmaps[:, :, i] * 255)
        output_heatmap = np.clip(output_heatmap, 0, 255).astype(np.uint8)
        output_colored_heatmap = cv2.applyColorMap(output_heatmap, cv2.COLORMAP_JET).astype(np.float32)
        output_masked_image = output_colored_heatmap * 0.7 + resized_background * 0.3
        row1_image[heatmap_height:2*heatmap_height, width_begin:width_end, :] = output_masked_image
        
        
    # row 2
    for i in range(7):
        width_begin = heatmap_width * i
        width_end   = heatmap_width * (i+1)
        
        heatmap = (heatmaps[:, :, i+7] * 255) * flags[i+7]
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype(np.float32)

        if draw_keypoint:
            if flags[i+7] > 0 and maxvals[0][i+7][0] > 0.3:
                cv2.circle(background, (int(preds[0][i+7][0] * 2.0), int(preds[0][i+7][1] * 2.0)), 2, (0, 0, 255), -1)

        masked_image = colored_heatmap * 0.7 + resized_background * 0.3
        row2_image[:heatmap_height, width_begin:width_end, :] = masked_image
        
        output_heatmap = (output_heatmaps[:, :, i+7] * 255)
        output_heatmap = np.clip(output_heatmap, 0, 255).astype(np.uint8)
        output_colored_heatmap = cv2.applyColorMap(output_heatmap, cv2.COLORMAP_JET).astype(np.float32)
        output_masked_image = output_colored_heatmap * 0.7 + resized_background * 0.3
        row2_image[heatmap_height:2*heatmap_height, width_begin:width_end, :] = output_masked_image
        
    if draw_keypoint:
        total_image = np.zeros((4 * heatmap_height + 20, 9 * heatmap_width, 3), dtype=np.uint8)
        total_image[:2*heatmap_height, :7*heatmap_width, :] = row1_image.astype(np.uint8)
        total_image[2*heatmap_height + 20:4*heatmap_height + 20, :7*heatmap_width, :] = row2_image.astype(np.uint8)
        total_image[heatmap_height+10:3*heatmap_height + 10, 7*heatmap_width:9*heatmap_width, :] = background

    else:     
        total_image = np.zeros((4 * heatmap_height + 20, 7 * heatmap_width, 3), dtype=np.uint8)
        total_image[:2*heatmap_height, :, :] = row1_image.astype(np.uint8)
        total_image[2*heatmap_height + 20:4*heatmap_height + 20, :, :] = row2_image.astype(np.uint8)
    
    resized_total_image = cv2.resize(total_image, (0, 0), fx=2.0, fy=2.0)
    cv2.imshow('test', resized_total_image)

    key = cv2.waitKey(delay) & 0xFF
    if key == 27:
        cv2.destroyAllWindows()
