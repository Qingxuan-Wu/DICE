import os
from typing import Any
import pickle
import numpy as np
import torch


def pickle_loader(path: str) -> Any:
    with open(path, 'rb') as pfile:
        data = pickle.load(pfile)
    return data


def pickle_saver(path: str, obj: Any) -> None:
    with open(path, 'wb') as pfile:
        pickle.dump(obj, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    return


def tor2np(data: torch.Tensor) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if data is None:
        return None
    if data.is_cuda:
        return data.detach().cpu().numpy()
    else:
        return data.detach().numpy()


def np2tor(data: np.ndarray, device=None) -> torch.Tensor:
    if type(device) == type(None):
        return torch.FloatTensor(data)
    else: 
        return torch.FloatTensor(data).to(device)


def make_dirs(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("new dir created :", save_path)
    return

def save_args_as_text(args,save_path):
    arg_list=str(args)[len("Namespace("):-1].split(",")
    with open(save_path+"/args.txt", 'w') as fp:
        for item in arg_list: 
            fp.write("%s\n" % item)
        print('Done')
    return 