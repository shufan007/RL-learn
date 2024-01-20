
from datetime import datetime
import numpy as np
import torch


def time2str(timestamp):
    d = datetime.fromtimestamp(timestamp)
    timestamp_str = d.strftime('%Y-%m-%d %H:%M:%S')
    return timestamp_str


def save_pth_model(model, save_model_path):
    torch.save(model.policy, save_model_path + '.pth')


def load_pth_model(pth_model_path):
    pth_model = torch.load(pth_model_path)
    return pth_model