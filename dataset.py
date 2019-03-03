from torchvision.datasets.voc import VOCDetection
import numpy as np


# TODO: overrap VOCDataset by YOLODataset

# ref: https://github.com/makora9143/yolo-pytorch/blob/master/yolov1/data.py
def create_label(obj_list, class_list, image_size, B=2, S=14):
    obj_list_norm = [normalize_coords(obj, image_size, B, S) for obj in obj_list]

    confs = np.zeros([S * S, B]) # for 2 bounding box per each cell
    coord = np.zeros([S * S, B, 4]) # for 4 coordinates per bounding box per cell
    proid = np.zeros([S * S, C]) # for class_probs weight \mathbb{1}^{obj}
    prear = np.zeros([S * S, 4]) # for bounding box coordinates

def normalize_coords(obj, img_size, B, S):
    w_img, h_img = img_size
    x_obj, y_obj, w_obj, h_obj = obj

    cell_x = 1. * w_img / S # width per cell
    cell_y = 1. * h_img / S # height per cell

    # rescale the center x to cell size
    cx = (x_obj + w_obj / 2) / cell_x
    cy = (y_obj + h_obj / 2) / cell_y

    assert(cx < S and cy < S)

    x_obj = cx - np.floor(cx) # center x in each cell
    y_obj = cy - np.floor(cy) # center x in each cell

    w_obj = np.sqrt(w_obj / w_img)
    h_obj = np.sqrt(h_obj / h_img)

    index = int(np.floor(cy) * S + np.floor(cx))
    obj = [x_obj, y_obj, w_obj, h_obj, index]

    return obj


if __name__ == "__main__":
    obj_list = [[128, 28, 55, 100]]
    class_list = [[0]]
    create_label(obj_list, class_list, image_size=(600, 400))
    VOCDetection(root="/Users/yudaikato/dataset/voc/",  year="2012")
