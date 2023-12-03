#!/usr/bin/env python

"""
.. module:: measure_overlap
   :platform: Linux, Windows
   :synopsis: Script for comparing wood classes with metrics output

.. moduleauthor:: JOESH, test version 
"""

import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
import tifffile

import logging
import math
import sys
import os


settings = {
    "labels": {1: "healthy", 2: "infected", 3: "heartwood"},
    "use_label_fix": True,  # Swap labels for healthy wood and heartwood in GT image
    "rgb_to_index_lut": {  # Mapping from CVAT color values to label indices
        (0, 0, 0): 0,       # Background
        (61, 245, 61): 1,   # Heartwood
        (250, 50, 83): 2,   # Infected
        (184, 61, 245): 3,  # Healthy
        (250, 250, 55): 4,  # Bark
    },
}


# Create a logger so that we can suppress output if running in batch mode
logger = logging.getLogger("measure_overlap")
logging.basicConfig(level=logging.INFO)


def convert_rgb_to_label_mask(image, lut): """Convert RGB image to label mask from LUT"""
dim = image.shape
output = np.zeros([dim[0], dim[1]], dtype=np.uint8)
    for y in range(dim[0]):
        for x in range(dim[1]):
            rgb = np.floor(image[y, x, 0:3] * 255.0)
            output[y, x] = lut[tuple(rgb)]
            return output


def swap_image_labels(image, a, b):
    """Swap two image labels (modifies the image in-place)"""
    mask_a = image == a
    mask_b = image == b
    image[mask_a] = b
    image[mask_b] = a


def compute_dice(image, other, label, zero_division=0.0):
    """Compute Dice coefficient for label in image pair"""
    a = image == label
    b = other == label
    if a.sum() + b.sum() == 0:
        return zero_division
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def compute_jaccard(image, other, label, zero_division=0.0):
    """Compute Jaccard index (or IoU) for label in image pair"""
    a = image == label
    b = other == label
    if a.sum() + b.sum() == 0:
        return zero_division
    return (a & b).sum() / (a | b).sum()


def compute_accuracy(image, other, label):
    """Compute accuracy (TP+TN)/(TP+FN+FP+TN) for label in image pair"""
    c = (image > 0) & (other > 0)  # Mask for background removal
    a = image == label
    b = other == label
    return 1.0 - ((a ^ b) & c).sum() / c.sum()


def compute_precision(image, other, label):
    """Compute precision TP/(TP+TN) for label in image pair"""
    c = (image > 0) & (other > 0)  # Mask for background removal
    a = image == label
    b = other == label
    return ((a & b) & c).sum() / max(1, ((a & c).sum()))


def compute_sensitivity(image, other, label):
    """Compute sensitivity TP/(TP+FN) for label in image pair"""
    c = (image > 0) & (other > 0)  # Mask for background removal
    a = image == label
    b = other == label
    return ((a & b) & c).sum() / max(1, ((b & c).sum()))


def compute_hausdorff(image, other, label):
    """Compute 2-sided Hausdorff distance for label in image pair"""
    a = image == label
    b = other == label
    return max(
        sp.spatial.distance.directed_hausdorff(a, b)[0],
        sp.spatial.distance.directed_hausdorff(b, a)[0],
    )


def plot_measurement(measurements, name, labels, prefix_ids, normalized=True):
    """Plot measurement as violin plots"""
    plt.figure(figsize=[6.4 * len(labels), 5.0])
    for item in labels:
        plt.subplot(1, len(labels), item)
        for index, id in enumerate(prefix_ids):
            ys = measurements[id][name[0] + "_" + labels[item]]
            xs = [index + 0.25 * ((math.pi * x) % 1.0 - 0.5) for x in range(0, len(ys))]
            violin = plt.violinplot(
                ys, positions=[index], showextrema=False, showmedians=True, widths=0.75
            )
            violin["bodies"][0].set_edgecolor("#000000")
            violin["cmedians"].set_edgecolor("#000000")
        for index, id in enumerate(prefix_ids):
            ys = measurements[id][name[0] + "_" + labels[item]]
            xs = [index + 0.25 * ((math.pi * x) % 1.0 - 0.5) for x in range(0, len(ys))]
            plt.plot(xs, ys, linestyle="", marker="o", markersize=2)
            plt.gca().lines[-1].set_color("#000000")
        plt.title("Wood class: " + labels[item], fontsize=16)
        plt.xlabel("Log ID", fontsize=14)
        plt.ylabel(name[1], fontsize=14)
        plt.grid(axis="x", linestyle="--")
        plt.xticks(range(0, len(prefix_ids)), prefix_ids, rotation=-45)
        if normalized:
            plt.ylim(0, 1)
        plt.tight_layout()


def main():
    if len(sys.argv) <= 2:
        sys.exit("Usage: ./measure_overlap.py seg_folder gt_folder")
    input_dir_seg = sys.argv[1]
    input_dir_gt = sys.argv[2]

    files_seg = os.listdir(input_dir_seg)
    files_seg = [item for item in files_seg if ".tif" in item]
    files_seg.sort()

    files_gt = os.listdir(input_dir_gt)
    files_gt = [item for item in files_gt if ".png" in item]
    files_gt.sort()

    # Compare filenames to keep only matching image pairs
    files_seg_tmp, files_gt_tmp = [], []
    for item in files_gt:
        common_prefix = item.split(".")[0]
        if common_prefix + "_labels.tif" in files_seg:
            files_seg_tmp.append(common_prefix + "_labels.tif")
            files_gt_tmp.append(common_prefix + ".png")
    files_seg, files_gt = files_seg_tmp, files_gt_tmp
    assert len(files_seg) == len(files_gt)

    labels = settings["labels"]
    measurements = {}

    prefix_ids = list(dict.fromkeys([s.split("_")[1] for s in files_seg]))

    for id in prefix_ids:
        measurements[id] = {}
        for item in labels:
            measurements[id]["dice_" + labels[item]] = []
            measurements[id]["jaccard_" + labels[item]] = []
            measurements[id]["accuracy_" + labels[item]] = []
            measurements[id]["precision_" + labels[item]] = []
            measurements[id]["sensitivity_" + labels[item]] = []
            measurements[id]["hausdorff_" + labels[item]] = []

        print("Computing measurements for image pairs...")
        for pair in zip(files_seg, files_gt):
            if pair[0].split("_")[1] != id:
                continue

            image_seg = tifffile.imread(os.path.join(input_dir_seg, pair[0]))
            image_gt = plt.imread(os.path.join(input_dir_gt, pair[1]))
            if len(image_gt.shape) == 3:
                print("  Converting RGB colors to label mask...")
                image_gt = convert_rgb_to_label_mask(image_gt, settings["rgb_to_index_lut"])
            assert image_seg.shape == image_gt.shape

            # Rescale GT image to obtain label indices
            image_gt = np.floor(image_gt * 255.0 + 0.5).astype(image_seg.dtype)
            if settings["use_label_fix"]:
                #swap_image_labels(image_gt, 1, 3)  # FIXME Hardcoded label indices
                #image_gt[image_gt == 4] = 1  # Assign healthy wood label to bark
                swap_image_labels(image_gt, 2, 3)
                swap_image_labels(image_gt, 1, 3)
                image_gt[image_gt == 5] = 1

            # Compute measurements for this image pair
            for item in labels:
                dice = compute_dice(image_seg, image_gt, item, zero_division=1.0)
                jaccard = compute_jaccard(image_seg, image_gt, item, zero_division=1.0)
                accuracy = compute_accuracy(image_seg, image_gt, item)
                precision = compute_precision(image_seg, image_gt, item)
                sensitivity = compute_sensitivity(image_seg, image_gt, item)
                hausdorff = compute_hausdorff(image_seg, image_gt, item)
                measurements[id]["dice_" + labels[item]].append(dice)
                measurements[id]["jaccard_" + labels[item]].append(jaccard)
                measurements[id]["accuracy_" + labels[item]].append(accuracy)
                measurements[id]["precision_" + labels[item]].append(precision)
                measurements[id]["sensitivity_" + labels[item]].append(sensitivity)
                measurements[id]["hausdorff_" + labels[item]].append(hausdorff)
        print("Done")

        print("Measurements (%s):" % id)
        for item in measurements[id]:
            median = np.median(measurements[id][item])
            mean = np.mean(measurements[id][item])
            std = np.std(measurements[id][item])
            print(
                "  %s: median: %.3f, mean: %.3f, std: %.3f" % (item, median, mean, std)
            )

    plot_measurement(measurements, ("dice", "Dice coefficient"), labels, prefix_ids)
    plt.savefig("output/dice.png")
    plot_measurement(measurements, ("jaccard", "Jaccard index"), labels, prefix_ids)
    plt.savefig("output/jaccard.png")
    plot_measurement(
        measurements, ("accuracy", "Accuracy (TP+TN)/(TP+FN+FP+TN)"), labels, prefix_ids
    )
    plt.savefig("output/accuracy.png")
    plot_measurement(
        measurements, ("precision", "Precision TP/(TP+FP)"), labels, prefix_ids
    )
    plt.savefig("output/precision.png")
    plot_measurement(
        measurements, ("sensitivity", "Sensitivity TP/(TP+FN)"), labels, prefix_ids
    )
    plt.savefig("output/sensitivity.png")
    plot_measurement(
        measurements,
        ("hausdorff", "2-sided Hausdorff distance"),
        labels,
        prefix_ids,
        normalized=False,
    )
    plt.savefig("output/hausdorff.png")

    plt.show()


if __name__ == "__main__":
    main()

print('Plot is completed.')