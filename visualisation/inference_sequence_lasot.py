import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

# In this script we will visualise the results after we pass the sequences through the AiA track framework
# The inferencing from the model will not be done in real time, we will implement that in another script instead
# In this script we will simply specify a sequence, and the visualise the bounding boxes one by one


def get_names(root_path, seq_name=''):

    """
    Grabs the names of of all of the files under a root path
    """

    # Split the sequence names in order to get the sequence class

    root_path = _join_paths(root_path,seq_name)

    # dir_list = os.listdir(root_path)
    names = []
    paths = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            # print(os.path.join(path, name))
            # print(name)
            file_dir = os.path.join(os.path.basename(path),name)
            names.append(file_dir)
            paths.append(os.path.join(path, name))

    return names, paths

def _join_paths(root_path, seq_name=''):

    """
    Joins the paths of the root with the specified sequence
    """

    # Split the sequence name in order to get the sequence class
    class_name = seq_name.split('-')[0]

    return os.path.join(root_path,f'{class_name}',f'{seq_name}','img')

def open_bbox(bbox_path, seq_name='' ):

    """
    Outputs a list full of the ground truth bounding boxes for each of the images of the sequence
    """

    gt_path = os.path.join(bbox_path, f'{seq_name}.txt')
    gt_list = []

    gt_file = open(gt_path)
    for line in gt_file:
        line_list = line.split(',')
        line_list[-1] = line_list[-1].strip()
        gt_list.append(line_list)

    return gt_list

def box_cxcywh_to_xyxy(x_c, y_c, w, h):

    xmin = x_c - 0.5 * w
    ymin = y_c - 0.5 * h
    xmax = x_c + 0.5 * w
    ymax = y_c + 0.5 * h

    return xmin, ymin, xmax, ymax

def loop_images(paths, bbox_path, seq_name=''):

    # Load the ground truth
    box_list = open_bbox(bbox_path, seq_name)

    # Once loaded, we can loop through the lines one by one and then plot the ground truth onto the image

    for index, path in enumerate(paths):
        im = Image.open(path)
        box = box_list[index]


        plot_results(im, box)

def plot_results(pil_img, box):

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    colors = COLORS * 100

    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)

    ax = plt.gca()


    cx = int(box[0])
    cy = int(box[1])
    w = int(box[2])
    h = int(box[3])

    # Convert the bounding boxes into the xminymin, yminymax data style
    xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(cx, cy, w, h)

    ax.add_patch(plt.Rectangle((cx, cy), w, h, fill=False, color = colors[0], linewidth=3))

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    bbox_path = '/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/AiATrack/AiATrack/tracking/PATH/AiATrack/test/tracking_results/aiatrack/baseline'
    root_path = '/media/atr17/HDD Storage/Datasets_Download/LaSOT/LaSOT'

    seq_name = 'robot-5'

    names, paths = get_names(root_path, seq_name)

    names = sorted(names)
    paths = sorted(paths)
    #
    print(paths)

    gt_list = open_bbox(bbox_path, seq_name)

    print(gt_list)

    loop_images(paths, bbox_path, seq_name)

    # TODO: Should also implement the ground truth visualisation when got time.
