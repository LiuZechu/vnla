import sys

import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ObjectDetectionModel(nn.Module):

    def __init__(self, hparams, device):

        super(ObjectDetectionModel, self).__init__()        

        # load cooccurrence matrix
        self.cooccurrence_matrix = torch.load('matrix.pt').to(device=device)
        
        # neural network for object detection
        output_size = 320 # TODO: remove hardcoding
        input_size = hparams.img_feature_size
        detection_layers = []
        current_layer_size = input_size
        next_layer_size = hparams.img_feature_size // 2

        if not hasattr(hparams, 'num_detection_layers'):
            hparams.num_detection_layers = 2

        for i in range(hparams.num_detection_layers):
            detection_layers.append(nn.Linear(current_layer_size, next_layer_size))
            detection_layers.append(nn.ReLU())
            detection_layers.append(nn.Dropout(p=hparams.dropout_ratio))
            current_layer_size = next_layer_size
            next_layer_size //= 2
        detection_layers.append(nn.Linear(current_layer_size, output_size))
        detection_layers.append(nn.Sigmoid())

        self.detection = nn.Sequential(*tuple(detection_layers))

        self.device = device

    def forward(self, image_feature):
        objects_occurrence = self.detection(image_feature)
        objects_nearby = torch.matmul(objects_occurrence, self.cooccurrence_matrix)       
 
        return objects_nearby

if __name__ == "__main__":
  print(torch.__version__)
