import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, S=14, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.S = S # number of grids
        self.B = B # number of bboxes in each cell
        self.C = C # number of classes

    # ref: https://github.com/makora9143/yolo-pytorch/blob/master/yolov1/loss.py
    def forward(self, y_pred, y_true):
        ''' Calculate the loss of YOLO model.
        args:
            y_pred: (Batch, 7 * 7 * 30)
            y_true: dict object that contains:
                class_probs,
                confs,
                coord,
                proid,
                areas,
                upleft,
                bottomright
        '''
        
