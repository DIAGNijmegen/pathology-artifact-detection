import numpy as np
import scipy.ndimage
import logging
import os
import datetime
import time
import math


def _array_filtering(binary_array, diagonal_threshold, full_connectivity):
    """
    Classify the regions in the reference binary array into smaller and larger than threshold size diameter classes.

    Args:
        binary_array (np.ndarray): Input binary array.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.

    Returns:
        np.ndarray, list, list: Labeled array, region bounding boxes, large enough region flags for each region.
    """

    # Identify the reference objects.
    #
    connectivity_structure = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]] if full_connectivity else [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_)
    array_labels, _ = scipy.ndimage.measurements.label(input=binary_array, structure=connectivity_structure)
    array_objects = scipy.ndimage.measurements.find_objects(input=array_labels)

    # Collect true positive and false negative regions.
    #
    large_enough_flags = []

    for object_index in range(len(array_objects)):
        object_bounding_box = array_objects[object_index]

        object_height = object_bounding_box[0].stop - object_bounding_box[0].start
        object_width = object_bounding_box[1].stop - object_bounding_box[1].start
        object_diagonal = math.sqrt(object_height * object_height + object_width * object_width)

        large_enough_region = diagonal_threshold < object_diagonal
        large_enough_flags.append(large_enough_region)

    # Return the collected regions and flags.
    #
    return array_labels, array_objects, large_enough_flags

def filter_regions_array(input_array, diagonal_threshold, full_connectivity, foreground_labels=None, background_label=0):
    """
    Filter out regions that have smaller than the configured diagonal. The function modifies the input array.

    Args:
        input_array (np.ndarray): Input array. Modified.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        background_label (int): Label value for overwriting the identified small regions.

    Returns:
        np.ndarray, int, int: Result array, identified region count, filtered region count.
    """

    # Select the foreground labels from the array.
    #
    input_array_binary = np.isin(element=input_array, test_elements=foreground_labels) if foreground_labels is not None else input_array

    # Select the foreground labels from the array.
    #
    # Identify the objects.
    #
    array_labels, array_objects, large_enough_flags = _array_filtering(binary_array=input_array_binary, diagonal_threshold=diagonal_threshold, full_connectivity=full_connectivity)

    # Go through the objects and remove smaller than threshold regions.
    #
    removed_regions = 0
    for object_index in range(len(array_objects)):
        if not large_enough_flags[object_index]:
            object_bounding_box = array_objects[object_index]

            object_patch = array_labels[object_bounding_box]
            object_mask = np.equal(object_patch, object_index + 1)

            content_patch = input_array[object_bounding_box]
            content_patch[object_mask] = background_label

            removed_regions += 1

    # Return the result image, the total number of identified regions and the number of removed regions.
    #
    return input_array, len(array_objects), removed_regions


def fill_holes_array(input_array, diagonal_threshold, full_connectivity, foreground_labels=None, fill_value=1):
    """
    Fill holes that have smaller than the configured diagonal. The function modifies the input array.

    Args:
        input_array (np.ndarray): Input array. Modified.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        fill_value (int): Label value for overwriting the identified region holes.

    Returns:
        np.ndarray, int, int: Result array, identified region count, filtered region count.
    """

    # Select the background labels from the array.
    #
    input_array_binary = np.logical_not(np.isin(element=input_array, test_elements=foreground_labels)) if foreground_labels is not None else np.logical_not(input_array)

    # Identify the objects.
    #
    array_labels, array_objects, large_enough_flags = _array_filtering(binary_array=input_array_binary, diagonal_threshold=diagonal_threshold, full_connectivity=full_connectivity)

    # Go through the objects and fill smaller than threshold holes.
    #
    filled_holes = 0
    for object_index in range(len(array_objects)):
        if not large_enough_flags[object_index]:
            object_bounding_box = array_objects[object_index]

            object_patch = array_labels[object_bounding_box]
            object_mask = np.equal(object_patch, object_index + 1)

            content_patch = input_array[object_bounding_box]
            content_patch[object_mask] = fill_value

            filled_holes += 1

    # Return the result image, the total number of identified holes and the number of filled holes.
    #
    return input_array, len(array_objects), filled_holes