import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import floor, ceil, sqrt

import warnings
warnings.filterwarnings("ignore")

def imshow(*args, figsize=10, to_rgb=True, title=None, fontsize=12):
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    images = args[0] if type(args[0]) is list else list(args)
    if to_rgb:
        images = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), images))
    if title is not None:
        assert len(title) == len(images), "Please provide a title for each image."
    plt.figure(figsize=figsize)
    for i in range(1, len(images)+1):
        plt.subplot(1, len(images), i)
        if title is not None:
            plt.title(title[i-1], fontsize=fontsize)
        plt.imshow(images[i-1])
        plt.axis('off')
    plt.show()
     
def fourier_transform_shift(image):
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude, phase = np.abs(fft), np.angle(fft)
    return magnitude, phase

def fourier_transform(image):
    fft = np.fft.fft2(image)
    magnitude, phase = np.abs(fft), np.angle(fft)
    return magnitude, phase


def normalize(image, rmin=0, rmax=255, to_uint=True):
    norm = ((image - image.min()) / (image.max() - image.min())) * (rmax - rmin) + rmin
    if to_uint:
        norm = norm.astype('uint8')
    return norm

def visualize_magnitude(magnitude):
    return normalize(20 * np.log(magnitude))
4.1.1
a = [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]
fft = np.fft.rfft2(a)
mag = np.abs(fft)
print(mag)
[[1.     0.25  ]
 [0.25   0.0625]
 [0.25   0.0625]]

a = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
fft = np.fft.rfft2(a)
mag = np.abs(fft)
print(mag)
[[0. 9.]
 [9. 9.]
 [9. 9.]]

a = [[0,-1,0],[-1,5,-1],[0,-1,0]]
fft = np.fft.rfft2(a)
mag = np.abs(fft)
print(mag)
[[1. 4.]
 [4. 7.]
 [4. 7.]]
image = Image.open('Lena.bmp')
image_arr = np.array(image)/255.0
filter = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]], dtype=np.float32)
convolved_arr = convolve(image_arr, filter, padding=(1, 1))
Image.fromarray(np.uint8(255 * convolved_arr), 'RGB') # Convolved Image
 
filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
convolved_arr = convolve(image_arr, filter, padding=(1, 1))
Image.fromarray(np.uint8(255 * convolved_arr), 'RGB') # Convolved Image
 
filter = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
convolved_arr = convolve(image_arr, filter, padding=(1, 1))
Image.fromarray(np.uint8(255 * convolved_arr), 'RGB') # Convolved Image
 


4.1.2
#with shift

img1 = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
#with shift
mag1, phase1 = fourier_transform_shift(img1)
#without log
mag_normal = normalize(mag1)
#with log
mag_visual = visualize_magnitude(mag1)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img1, mag_normal, mag_visual, title=titles, figsize=16)

#without shift
mag1, phase1 = fourier_transform_shift(img1)
#without log
mag_normal = normalize(mag1)
#with log
mag_visual = visualize_magnitude(mag1)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img1, mag_normal, mag_visual, title=titles, figsize=16)

img2 = cv2.imread('F16.bmp', cv2.IMREAD_GRAYSCALE)
#with shift
mag2, phase2 = fourier_transform(img2)
#without log
mag_normal = normalize(mag2)
#with log
mag_visual = visualize_magnitude(mag2)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img2, mag_normal, mag_visual, title=titles, figsize=16)

#without shift
mag2, phase2 = fourier_transform_shift(img2)
#without log
mag_normal = normalize(mag2)
#with log
mag_visual = visualize_magnitude(mag2)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img2, mag_normal, mag_visual, title=titles, figsize=16)

img3 = cv2.imread('Barbara.bmp', cv2.IMREAD_GRAYSCALE)
#with shift
mag3, phase3 = fourier_transform(img3)
#without log
mag_normal = normalize(mag3)
#with log
mag_visual = visualize_magnitude(mag3)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img3, mag_normal, mag_visual, title=titles, figsize=16)

#without shift
mag3, phase3 = fourier_transform_shift(img3)
#without log
mag_normal = normalize(mag3)
#with log
mag_visual = visualize_magnitude(mag3)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img3, mag_normal, mag_visual, title=titles, figsize=16)

img4 = cv2.imread('Baboon.bmp', cv2.IMREAD_GRAYSCALE)
#without shift
mag4, phase4 = fourier_transform(img4)
#without log
mag_normal = normalize(mag4)
#with log
mag_visual = visualize_magnitude(mag4)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img4, mag_normal, mag_visual, title=titles, figsize=16)

#with shift
mag14, phase4 = fourier_transform_shift(img4)
#without log
mag_normal = normalize(mag4)
#with log
mag_visual = visualize_magnitude(mag4)
titles = ['Image', 'magnitude', 'log(magnitude)',]
imshow(img4, mag_normal, mag_visual, title=titles, figsize=16)
