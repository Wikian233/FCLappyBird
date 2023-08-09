import cv2
import time
import numpy as np
from mss import mss
import subprocess
import re

import torch
import torch.nn as nn


conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
bn1 = nn.BatchNorm2d(16)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) 
conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
bn2 = nn.BatchNorm2d(32)
conv_layer3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
bn3 = nn.BatchNorm2d(16)
conv_layer4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
bn4 = nn.BatchNorm2d(4)

if torch.cuda.is_available():
    conv_layer = conv_layer.cuda()
    pool_layer = pool_layer.cuda()
    conv_layer2 = conv_layer2.cuda()
    conv_layer3 = conv_layer3.cuda()
    conv_layer4 = conv_layer4.cuda()
    bn1 = bn1.cuda()
    bn2 = bn2.cuda()
    bn3 = bn3.cuda()
    bn4 = bn4.cuda()


def get_window_geometry(window_name):
    """
    Get the geometry of a window with the given name by parsing the output of 'xwininfo'.
    Return a dictionary containing the left, top, width, and height of the window.
    """
    output = subprocess.check_output(['xwininfo', '-name', window_name]).decode()

    geometry = re.search('geometry (\d+x\d+\+\d+\+\d+)', output)
    if not geometry:
        raise ValueError(f'Could not find window with name: {window_name}')

    geometry = geometry.group(1)
    width, height, left, top = map(int, re.findall('\d+', geometry))

    return {'left': left, 'top': top + 50, 'width': width, 'height': height}

def process_img(image):
    image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor_frame = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # 为batch和channel增加一个维度

    if torch.cuda.is_available():
        tensor_frame = tensor_frame.cuda()

    # the first conv-bn-relu-pooling layer
    tensor_frame = conv_layer(tensor_frame)
    tensor_frame = bn1(tensor_frame)
    tensor_frame = nn.functional.relu(tensor_frame)
    tensor_frame = pool_layer(tensor_frame) 

    # the second conv-bn-relu layer
    tensor_frame = conv_layer2(tensor_frame)
    tensor_frame = bn2(tensor_frame)
    tensor_frame = nn.functional.relu(tensor_frame)
    tensor_frame = pool_layer(tensor_frame)
    # the third conv-bn-relu layer
    tensor_frame = conv_layer3(tensor_frame)
    tensor_frame = bn3(tensor_frame)
    tensor_frame = nn.functional.relu(tensor_frame)
    tensor_frame = pool_layer(tensor_frame)
    # the fourth conv-bn-relu layer
    tensor_frame = conv_layer4(tensor_frame)
    tensor_frame = bn4(tensor_frame)
    tensor_frame = nn.functional.relu(tensor_frame)
    tensor_frame = pool_layer(tensor_frame)

    # if necessary, convert tensor back to numpy
    tensor_frame = tensor_frame.cpu().detach().numpy()

    return tensor_frame

def screen_record(window_name):
    sct = mss()
    last_time = time.time()

    while True:
        geometry = get_window_geometry(window_name)
        screenshot = sct.grab(geometry)
        img = np.array(screenshot)

        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time

        print('fps: {}'.format(fps))


        img = process_img(img)
        image = np.ndarray.flatten(img[0, 1, :, :])
        print(image.shape)


        img_to_show = img[0, 2, :, :]
        img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min()) * 255
        img_to_show = img_to_show.astype(np.uint8)
        cv2.imshow('Screen Capture', img_to_show)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        

if __name__ == "__main__":
    screen_record('FCLappyBird')

