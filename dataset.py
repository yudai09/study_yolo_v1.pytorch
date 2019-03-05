from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import numpy as np


class YOLODataset(Dataset):
    def __init__(self):
        super(YOLODataset, self).__init__()
        for img, anno in VOCDetection(root="/Users/yudaikato/dataset/voc/",  year="2012"):
            bndbox_list = []
            objs = anno["annotation"]["object"]
            if isinstance(objs, list):
                for obj in objs:
                    bndbox_list.append(self._trans_coord(obj["bndbox"]))
                    class_list = [obj["name"] for obj in anno["annotation"]["object"]]
            else:
                print(objs)
                bndbox_list.append(self._trans_coord(objs["bndbox"]))
                class_list = [objs["name"]]


            label = create_label(bndbox_list, class_list, image_size=img.size)

    def _trans_coord(self, bb):
        return [
            float(bb["xmin"]),
            float(bb["ymin"]),
            float(bb["xmax"]) - float(bb["xmin"]),
            float(bb["ymax"]) - float(bb["ymin"])
        ]


# ref: https://github.com/makora9143/yolo-pytorch/blob/master/yolov1/data.py
def create_label(bndbox_list, class_list, image_size, S=14, B=2, C=20):
    labels = {
        "person", "car", "bicycle", "bus",
        "motorbike", "train", "aeroplane",
        "chair", "bottle", "diningtable",
        "pottedplant", "tvmonitor", "sofa",
        "bird", "cat", "cow", "dog",
        "horse", "sheep", "boat",
    }
    label2num = {l:n for l,n in zip(labels, range(len(labels)))}

    bndbox_list_norm = [normalize_coords(bndbox, image_size, B, S) for bndbox in bndbox_list]

    class_probs = np.zeros([S * S, C]) # for one_hot vector per each cell
    confs = np.zeros([S * S, B]) # for 2 bounding box per each cell
    coord = np.zeros([S * S, B, 4]) # for 4 coordinates per bounding box per cell
    proid = np.zeros([S * S, C]) # for class_probs weight \mathbb{1}^{bndbox}
    prear = np.zeros([S * S, 4]) # for bounding box coordinates

    for (bndbox, idx), cls in zip(bndbox_list_norm, class_list):
        # assert(cls in labels)
        class_probs[idx, label2num[cls]] = 1.
        confs[idx, :] = [1.] * B
        coord[idx, :, :] = [bndbox] * B
        proid[idx, :] = [1] * C

        # transform width and height to the scale of coordinates
        prear[idx, 0] = bndbox[0] - bndbox[2] ** 2 * 0.5 * S # x_left
        prear[idx, 1] = bndbox[1] - bndbox[3] ** 2 * 0.5 * S # y_top
        prear[idx, 2] = bndbox[0] + bndbox[2] ** 2 * 0.5 * S # x_right
        prear[idx, 3] = bndbox[1] + bndbox[3] ** 2 * 0.5 * S # y_bottom

    # for calculate upleft, bottomright and areas for 2 bounding box(not for 1 bounding box)
    upleft = np.expand_dims(prear[:, 0:2], 1)
    bottomright = np.expand_dims(prear[:, 2:4], 1)
    wh = bottomright - upleft
    area = wh[:, :, 0] * wh[:, :, 1]
    upleft = np.concatenate([upleft] * B, 1)
    bottomright = np.concatenate([bottomright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    y_true = {
            'class_probs': class_probs,
            'confs': confs,
            'coord': coord,
            'proid': proid,
            'areas': areas,
            'upleft': upleft,
            'bottomright': bottomright
            }

    return y_true


def normalize_coords(bndbox, img_size, B, S):
    w_img, h_img = img_size
    x_bndbox, y_bndbox, w_bndbox, h_bndbox = bndbox

    w_cell = 1. * w_img / S # width per cell
    h_cell = 1. * h_img / S # height per cell

    # rescale the center x to cell size
    cx = (x_bndbox + w_bndbox / 2) / w_cell
    cy = (y_bndbox + h_bndbox / 2) / h_cell

    assert(cx < S and cy < S)

    x_bndbox = cx - np.floor(cx) # center x in each cell
    y_bndbox = cy - np.floor(cy) # center x in each cell

    w_bndbox = np.sqrt(w_bndbox / w_img)
    h_bndbox = np.sqrt(h_bndbox / h_img)

    idx = int(np.floor(cy) * S + np.floor(cx))
    bndbox = [x_bndbox, y_bndbox, w_bndbox, h_bndbox]

    return bndbox, idx


if __name__ == "__main__":
    YOLODataset()
