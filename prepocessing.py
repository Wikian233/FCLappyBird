import cv2
import time
import numpy as np
from mss import mss
import subprocess
import re


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
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img

def convolution2D(img, kernel):
    # 输入图像和核的尺寸
    iH, iW = img.shape
    kH, kW = kernel.shape
    
    # 输出的尺寸
    oH, oW = iH - kH + 1, iW - kW + 1
    output = np.zeros((oH, oW))
    
    # 卷积操作
    for i in range(oH):
        for j in range(oW):
            output[i, j] = np.sum(img[i:i+kH, j:j+kW] * kernel)
            
    return output

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

        img = cv2.resize(img, (geometry['width'] // 4, geometry['height'] // 2))  # decrease resolution
        img = cv2.putText(img, 'FPS: {:.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        img = process_img(img)

        sobel_kernel = np.array([[1, 1],
                                 [ -1, -1]])
        
        img = convolution2D(img, sobel_kernel)

        cv2.imshow('Screen Capture', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    screen_record('FCLappyBird')

