import numpy as np
import skimage
import skimage.transform
import torch
from torch.autograd import Variable
from visdom import Visdom

viz = Visdom()

Color = torch.FloatTensor(
    [[0, 0, 0.5],
     [0, 0, 1],
     [0, 1, 0],
     [1, 1, 0],
     [1, 0, 0]]
)

Color_cuda = Color.cuda()


def to_var(x, is_volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=is_volatile)


def batch_to_grid(batch_image, num_rows, num_cols):
    # batch_image shape [ B, H, W, C ]
    batch, height, width, channel = batch_image.shape

    grid_size = num_rows * num_cols
    if grid_size < batch:
        batch_image = batch_image[:grid_size]
        batch = grid_size
    elif grid_size > batch:
        dummy_img = np.zeros((grid_size - batch, height, width, channel))
        batch_image = np.concatenate((batch_image, dummy_img), axis=0)

    grid_image = batch_image.reshape(num_rows, num_cols, height, width, channel)
    grid_image = grid_image.swapaxes(1, 2).reshape(num_rows * height, num_cols * width, channel)

    return grid_image


def merge_to_color_heatmap(batch_heatmaps, h_format='NCHW'):
    color_ = Color_cuda if batch_heatmaps.is_cuda else Color

    if h_format == 'NHWC':
        batch_heatmaps = batch_heatmaps.permute(0, 3, 1, 2).contiguous()

    batch, joints, height, width = batch_heatmaps.size()

    heatmaps = batch_heatmaps.clamp(0, 1.).view(-1)

    frac = torch.div(heatmaps, 0.25)
    lower_indices, upper_indices = torch.floor(frac).long(), torch.ceil(frac).long()

    t = frac - torch.floor(frac)
    t = t.view(-1, 1)

    k = color_.index_select(0, lower_indices)
    k_1 = color_.index_select(0, upper_indices)

    color_heatmap = (1.0 - t) * k + t * k_1
    color_heatmap = color_heatmap.view(batch, joints, height, width, 3)
    color_heatmap = color_heatmap.permute(0, 4, 2, 3, 1)
    color_heatmap, _ = torch.max(color_heatmap, 4)

    return color_heatmap


def draw_line(x, y, window):
    assert viz.check_connection()

    return viz.line(X=x,
                    Y=y,
                    win=window,
                    update='append' if window is not None else None)


def draw_merged_image(heatmaps, images, window):
    assert viz.check_connection()

    if isinstance(heatmaps, Variable):
        heatmaps = heatmaps.data

    if isinstance(images, Variable):
        images = images.data.numpy()
    elif isinstance(images, torch.FloatTensor):
        images = images.numpy()

    heatmaps = merge_to_color_heatmap(heatmaps)
    heatmaps = heatmaps.permute(0, 2, 3, 1).cpu()  # NHWC

    resized_heatmaps = list()
    for idx, ht in enumerate(heatmaps):
        color_ht = skimage.transform.resize(ht.numpy(), (256, 256), mode='constant')
        resized_heatmaps.append(color_ht.transpose(2, 0, 1))

    resized_heatmaps = np.stack(resized_heatmaps, axis=0)

    images = images.transpose(0, 3, 1, 2) * 0.6
    overlayed_image = np.clip(images + resized_heatmaps * 0.4, 0, 1.)

    return viz.images(tensor=overlayed_image, nrow=4, win=window)
