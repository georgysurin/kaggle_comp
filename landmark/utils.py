import numpy as np
import sys
import datetime


def preprocess_image(image):
    image = image.transpose(2, 0, 1)
    image = (image - 127.5) / 128.0
    image = image.astype(np.float32)

    return image


def print_one_line(s):
    time_string = datetime.datetime.now().strftime('%H:%M:%S')
    sys.stdout.write('\r' + time_string + ' ' + s)
    sys.stdout.flush()


def set_lr(optimizer, new_lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = new_lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
