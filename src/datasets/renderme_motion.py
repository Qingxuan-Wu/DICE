import os
import joblib
import torch
import numpy as np

class RenderMeMotionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/code/datasets/renderme"):
        self.data_list = self.load_data(root_dir)

    def load_data(self, root_dir):
        print(f"loading RenderMe-360 data from {root_dir}")
        data_list = []
        for person_dir in sorted(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                for file_name in sorted(os.listdir(person_path)):
                    if file_name.endswith(".pkl"):
                        file_path = os.path.join(person_path, file_name)
                        with open(file_path, 'rb') as f:
                            data = joblib.load(f)
                    for key in data.keys():
                        d = {}
                        d["face_shape"] = data[key]["shape"]
                        d["face_exp"] = data[key]["exp"] 
                        d["face_pose"] = np.concatenate([data[key]["global_pose"], data[key]["jaw_pose"]])
                        data_list.append(d)
        print(len(data_list))

        for d in data_list:
            # check
            assert(sorted(list(d.keys())) == sorted(["face_shape", "face_exp", "face_pose"]))
            assert(d["face_shape"].shape == (100,))
            assert(d["face_exp"].shape == (50,))
            assert(d["face_pose"].shape == (6,))

        for idx in range(len(data_list)):
            for key in ["face_shape", "face_exp", "face_pose"]:
                data_list[idx][key] = torch.from_numpy(data_list[idx][key])

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
