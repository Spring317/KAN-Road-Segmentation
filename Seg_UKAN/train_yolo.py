from ultralytics import YOLO as yolo
import torch.nn as nn


class model:
"""Load YOLO in pytorch format and train it on the given data path."""
   
    def __init__(self, model_path):
        self.model = yolo(model_path)

    def load_model(self) -> nn.Module:
        return self.model

    def train(self, data_path, epochs=100, batch_size=16):
        self.model.train(data=data_path, epochs=epochs, batch_size=batch_size)


