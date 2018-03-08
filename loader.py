import json
import os
from functools import lru_cache

import numpy as np
import skimage
import skimage.io
import torch.utils.data as data
from vectormath import Vector2


@lru_cache(maxsize=32)
def gaussian(size, sigma=0.25, mean=0.5):
    width = size
    heigth = size
    amplitude = 1.0
    sigma_u = sigma
    sigma_v = sigma
    mean_u = mean * width + 0.5
    mean_v = mean * heigth + 0.5

    over_sigma_u = 1.0 / (sigma_u * width)
    over_sigma_v = 1.0 / (sigma_v * heigth)

    x = np.arange(0, width, 1, np.float32)
    y = x[:, np.newaxis]

    du = (x + 1 - mean_u) * over_sigma_u
    dv = (y + 1 - mean_v) * over_sigma_v

    return amplitude * np.exp(-0.5 * (du * du + dv * dv))


def generate_heatmap(size, y0, x0, pad=3):
    y0, x0 = int(y0), int(x0)
    dst = [max(0, y0 - pad), max(0, min(size, y0 + pad + 1)), max(0, x0 - pad), max(0, min(size, x0 + pad + 1))]
    src = [-min(0, y0 - pad), pad + min(pad, size - y0 - 1) + 1, -min(0, x0 - pad), pad + min(pad, size - x0 - 1) + 1]

    heatmap = np.zeros([size, size])
    g = gaussian(7)
    heatmap[dst[0]:dst[1], dst[2]:dst[3]] = g[src[0]:src[1], src[2]:src[3]]

    return heatmap


class UnityChan(data.Dataset):
    # json config file names
    CONFIG = 'configure.json'
    CONFIG_ANGLE = 'configure_angle.json'
    CONFIG_JOINT = 'joint.json'

    # json entity names
    NUMBER_OF_INDEX = 'numberOfIndex'
    ROTATION_ANGLE = 'rotationAngle'
    NUMBER_OF_JOINT = 'numberOfJoint'
    JOINTS = 'joints'

    # dict entity names
    PATH = 'path'
    IMAGE = 'image'
    DEPTH = 'depth'
    POSITION = 'position'

    # file name
    DEPTH_FILE = 'depth.png'
    IMAGE_FILE = 'image.png'

    WIDTH = HEIGHT = 256
    DEPTH_WIDTH = DEPTH_HEIGHT = 64

    def __init__(self, root, task='train', is_augmentation=True):
        self.root = root
        self.task = task
        self.is_augmentation = is_augmentation

        self._all_config, self._data = self._load_all_config()
        self._data_pairs = self._create_training_pairs()

    def _load_all_config(self):
        _config = dict()
        _data = list()

        config = os.path.join(self.root, self.CONFIG)
        if os.path.isfile(config) is None:
            msg = "File doesn't exist: '{}'"
            raise FileNotFoundError(msg.format(config))

        jc = json.load(open(config))
        _config[self.NUMBER_OF_INDEX] = jc[self.NUMBER_OF_INDEX]

        for action_id in range(_config[self.NUMBER_OF_INDEX]):
            config_angle = os.path.join(self.root, str(action_id), self.CONFIG_ANGLE)
            if os.path.isfile(config_angle) is None:
                msg = "File doesn't exist: '{}'"
                raise FileNotFoundError(msg.format(config))

            with open(config_angle) as fa:
                ja = json.load(fa)
            action_config = {
                self.PATH: os.path.join(self.root, str(action_id)),
                self.ROTATION_ANGLE: ja[self.ROTATION_ANGLE],
                self.NUMBER_OF_INDEX: ja[self.NUMBER_OF_INDEX]}

            poses_list = list()
            for camera_id in range(action_config[self.NUMBER_OF_INDEX]):
                config_joint = os.path.join(action_config[self.PATH], str(camera_id), self.CONFIG_JOINT)
                if os.path.isfile(config_angle) is None:
                    msg = "File doesn't exist: '{}'"
                    raise FileNotFoundError(msg.format(config))

                with open(config_joint) as fj:
                    jj = json.load(fj)

                poses_dict = {
                    self.PATH: os.path.join(action_config[self.PATH], str(camera_id)),
                    self.IMAGE: os.path.join(action_config[self.PATH], str(camera_id), self.IMAGE_FILE),
                    self.DEPTH: os.path.join(action_config[self.PATH], str(camera_id), self.DEPTH_FILE),
                    self.NUMBER_OF_JOINT: jj[self.NUMBER_OF_JOINT],
                    self.POSITION: [(pos['x'], self.HEIGHT - pos['y']) for pos in jj[self.JOINTS]]}
                poses_list.append(poses_dict)

            _config[action_id] = action_config
            _data.append(poses_list)

        return _config, _data

    def _create_training_pairs(self):
        total_pre_pose = list()
        total_aft_pose = list()
        for action in self._data:
            aft_pose = action[:]
            aft_pose.append(action[0])

            pre_pose = action[:]
            pre_pose.insert(0, action[-1])

            total_pre_pose.extend(pre_pose)
            total_aft_pose.extend(aft_pose)

        return list(zip(total_pre_pose, total_aft_pose))

    def __getitem__(self, index):
        pair_pose = self._data_pairs[index]

        return self._get_data(pair_pose[0]), self._get_data(pair_pose[1])

    def _get_data(self, pose):
        image = skimage.io.imread(pose[self.IMAGE])
        image = skimage.img_as_float(image)
        image = image.astype(np.float32)

        depth = skimage.io.imread(pose[self.DEPTH])
        if depth.shape[-1] == 4:  # has alpha channel
            depth = depth[:, :, 0]
        depth = skimage.img_as_float(depth)
        depth = depth.astype(np.float32)

        box_size = self.WIDTH
        resize_ratio = box_size / float(self.DEPTH_WIDTH)

        heatmaps = np.zeros(shape=(self.DEPTH_HEIGHT, self.DEPTH_WIDTH, pose[self.NUMBER_OF_JOINT]), dtype=np.float32)

        for idx in range(pose[self.NUMBER_OF_JOINT]):  # num of joint
            keypoint = Vector2(pose[self.POSITION][idx])
            # keypoint -= (center - box_size / 2)
            keypoint /= resize_ratio  # space change: original image >> crop image

            if min(keypoint) < 0 or max(keypoint) >= 64:
                continue

            heatmaps[:, :, idx] = generate_heatmap(64, keypoint.y, keypoint.x)

        return image, depth, heatmaps

    def __len__(self):
        return len(self._data_pairs)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from visdom import Visdom
    import utils
    from tqdm import tqdm

    viz = Visdom()

    u = UnityChan('./unity-chan', task='train')
    train_reader = DataLoader(u, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    with tqdm(total=len(train_reader), unit=' iters', unit_scale=False) as pbar:
        for prev, aftr in train_reader:
            viz.images(prev[0].permute(0, 3, 1, 2), win='pre')
            viz.images(aftr[0].permute(0, 3, 1, 2), win='aft')

            viz.images(aftr[1].view(-1, 1, 64, 64), win='pre_dpt')
            viz.images(aftr[1].view(-1, 1, 64, 64), win='aft_dpt')
            utils.draw_merged_image(prev[2].permute(0, 3, 1, 2), prev[0], window='merged')

            pbar.update(1)
