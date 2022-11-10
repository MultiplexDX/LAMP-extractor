import yaml
import cv2
import numpy as np

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def draw_keypoints(image, keypoints, diameter=1):
    # BGR colors
    KEYPOINT_COLORS = [
        (255, 0, 0), #BLUE
        (0, 255, 0), #GREEN
        (0, 0, 255), #RED
        (255, 255, 255) #WHITE
    ]
    for index, (x, y) in enumerate(keypoints):
        if x < 0 or y < 0:
            continue
        cv2.circle(img=image, center=(int(x), int(y)), radius=diameter, color=KEYPOINT_COLORS[index], thickness=3)
    return image

def colorize_matrix(matrix):
    colors = [
        [224, 224, 224],
        [255, 0, 255],
        [255, 128, 0],
        [255, 255, 0],
    ]

    colormask = np.zeros([*matrix.shape, 3])
    colormask[matrix == 0] = colors[0]
    colormask[matrix == 1] = colors[1]
    colormask[matrix == 2] = colors[2]
    colormask[matrix == 3] = colors[3]
    return colormask.astype(np.uint8)