import os

import zipfile
import numpy as np
import cv2

import copy

import torch
from torch.utils.data import Dataset
from PIL import Image

test = False
if test:
    ZIP = 'smallytb.zip'
    META = 'smeta.txt'
    PREFIX = 'smallytb'
else:
    ZIP = 'ytb.zip'
    META = 'meta.txt'
    PREFIX = 'newytb'

class ZipLoader(object):
    """Defines a class to load zip file.
    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image

class YTBDataset(Dataset):
    def __init__(self, interval, data_path, phase, transform):
        super().__init__()

        self.zip_path = os.path.join(data_path, ZIP)
        self.zip_loader = ZipLoader()
        self.meta_path = os.path.join(data_path, META)
        self.meta_info = open(self.meta_path).readlines()   
        self.transform = transform
        self.phase = phase
        self.interval = interval
        total_len = len(self.meta_info) // self.interval
        self.eval_offset = int(total_len * 0.7)
        assert phase in ['train', 'eval', 'all'], 'Not supported phase'

    def __len__(self):
        total_len = len(self.meta_info) // self.interval
        if self.phase == 'train':
            return int(total_len * 0.7) 
        elif self.phase == 'eval':
            return total_len - self.eval_offset 
        else:
            return total_len

    def __getitem__(self, index):
        if self.phase == 'eval':
            index += self.eval_offset
        index = index * self.interval
        info = self.meta_info[index]
        path, throttle, steering, speed = info.split()

        if index < 6510787:
            rindex = index
        else:
            rindex = index - 6510787

        path = path[:path.find('/')] + '/' + str(rindex) + path[-4:]

        throttle = float(throttle)
        steering = float(steering)
        speed = float(speed[:-3])
        img = self.zip_loader.get_image(self.zip_path, os.path.join(PREFIX, path))
        img = self.transform(Image.fromarray(img))
        return img, steering, throttle, speed

class LabelYTBDataset(YTBDataset):
    @staticmethod
    def _get_label(s, t):
        if s<0:
            s=0
        if s>1:
            s=1

        return s

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img, steering, throttle, speed = item
        label = self._get_label(steering, throttle)
        return img, label

def read_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("%s not found!" % filepath)
    image = cv2.imread(filepath)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if is_grayimage(image):
        image = image[..., :1]
    return image

def is_grayimage(image):
    if len(image.shape) == 2:
        return True
    x = abs(image[:, :, 0] - image[:, :, 1])
    y = np.sum(x)
    if y == 0:
        return True
    else:
        return False
          
class CarlaDataset(Dataset):

    def __init__(
        self, 
        root_dir: str,
        transform: bool = True, 
        preloads: str = None, 
        use_throttle: bool = False,
        interval: int = 5,
        action_seq_length: int = 5,
        aug = None,
    ) -> None:
        self._root_dir = root_dir
        self._transform = transform
        self.use_throttle = use_throttle
        self.interval = interval
        self.action_seq_length = action_seq_length
        self.aug = aug

        preload_file = preloads
        print(preloads)
        if preload_file is not None:
            print('[DATASET] Loading from NPY')
            self._sensor_data_names, self._measurements = np.load(preload_file, allow_pickle=True)
            
        self.calculate_real_length()
            
            
    # this function eliminate the last few frames of each episode
    # because we need a sequence of actions(after each frame) as label
    def calculate_real_length(self):
        interval = max(self.interval, self.action_seq_length)
        # first get two shifted measurements 
        self._sensor_data_names_left = self._sensor_data_names[:-interval].copy()
        self._sensor_data_names_right = self._sensor_data_names[interval:].copy()
        
        # map each meansurements to 'epi_xxxxx'
        get_episode = np.vectorize(lambda s : s.split('/')[0])
        self._sensor_data_names_left = get_episode(self._sensor_data_names_left)
        self._sensor_data_names_right = get_episode(self._sensor_data_names_right)
        
        # print(len(self._sensor_data_names_left))
        # print(len(self._sensor_data_names_right))
        # print(len(self._sensor_data_names))

        # get available indices
        self.available_indices = np.arange(len(self._sensor_data_names) - interval)[
            self._sensor_data_names_left == self._sensor_data_names_right
        ]

    # to eliminate the correlation between frames, we choose 
    # frames every {self.interval}
    def __len__(self) -> int:
        # return len(self._sensor_data_names) // self.interval
        return len(self.available_indices) // self.interval

    # first convert index to the scale it should be
    # then choose from the available indices list
    def __getitem__(self, index: int):
        index = index * self.interval
        index = self.available_indices[index]
        img_path = os.path.join(self._root_dir, self._sensor_data_names[index])
        img = read_image(img_path)
        if self._transform:
            img = img.transpose(2, 0, 1)
            img = img / 255.
        img = img.astype(np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        measurements = self._measurements[index:index+self.action_seq_length].copy()
        data = dict()
        data['rgb'] = img

        # concatenate all actions in the short future
        for i in range(self.action_seq_length):
            for k, v in measurements[i].items():
                v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
                if k not in data:
                    data[k] = v
                else:
                    data[k] = torch.cat([data[k], v], dim=0)
        
        img = data['rgb']
        if self.use_throttle:
            label = torch.cat([data['steer'], data['throttle']], dim=0)
        else:
            label = data['steer']
        
        if self.aug is not None:
            img = self.aug(img)
        
        return img, label


if __name__ == '__main__':
    transform = None
    root_dir='/home/ywang3/workplace/dataset_carla/town1_small/datasets_train/cilrs_datasets_train'
    preloads='/home/ywang3/workplace/dataset_carla/town1_small/_preloads/datasets_train.npy'
    
    dataset = CarlaDataset(root_dir=root_dir, preloads=preloads, use_throttle=False, interval=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None, drop_last=True)
    
    count = 0
    mean = 0
    std = 0
    for _, (img, label) in train_loader:
        
        
    
    data, label = dataset[10]
    print(data.shape)
    
