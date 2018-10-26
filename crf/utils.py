import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def read_image(imgname):
    img = Image.open(imgname).convert('RGB')
    img = np.array(img)
    img = img.astype(float)
    img = img / 255.
    return img

def grayscale(img):
    return rgb2gray(img)

def pad_image(img, t=0, b=0, l=0, r=0, padding='fill'):
    """
    Args:
        [img]       Shape HxW   Grayscale image.
        [t]         Int         Top padding.
        [b]         Int         Bottom padding.
        [l]         Int         Left padding.
        [r]         Int         Right padding.
        [padding]   String      Type of padding. Either "fill" or "replicate".
    Rets:
        Grayscale image with size (H+t+b, W+l+r) with margins padded.
    """
    if (t==0) and (b==0) and (l==0) and (r==0):
        return img
    h = img.shape[0]
    w = img.shape[1]
    newimg = np.zeros((h+t+b,w+l+r), dtype=img.dtype)
    newimg[t:h+t,l:w+l]=img
    if padding=='replicate':
        newimg[:t,l:w+l] = img[[0],:]
        newimg[h+t:,l:w+l] = img[[-1],:]
        newimg[t:h+t,:l] = img[:,[0]]
        newimg[t:h+t,w+l:] = img[:,[-1]]
        newimg[:t,:l] = img[0,0]
        newimg[:t,w+l:] = img[0,-1]
        newimg[h+t:,w+l:] = img[-1,-1]
        newimg[h+t:,:l] = img[-1,0]

    assert padding in ['fill', 'replicate'], "Only two options supported now"
    return newimg


def crop_patch(img, xmin, ymin, xmax, ymax):
    """
    Args:
        [img]   Shape:HxW   Grayscale image.
        [xmin]  Int         Minimum index on x-axis (i.e., along the width)
        [xmax]  Inta        Maximum index on x-axis (i.e., alone the width)
        [ymin]  Int         Minimum index on y-axis (i.e., along the height)
        [ymax]  Int         Minimum index on y-axis (i.e., along the height)
    Rets:
        Image of shape up to (ymax-ymin, xmax-xmin).
        If the index range goes outside the image, the return patch will be cropped.
    """
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    newimg = np.zeros((ymax-ymin, xmax-xmin))
    lby = np.maximum(ymin, 0)
    uby = np.minimum(ymax, img.shape[0])
    lbx = np.maximum(xmin, 0)
    ubx = np.minimum(xmax, img.shape[1])
    newimg[lby-ymin:uby-ymin, lbx-xmin:ubx-xmin] = img[lby:uby, lbx:ubx]
    return newimg


def compute_histogram(histogram_shape, indices, vals):
    """
    Args:
        [histogram_shape]   Shape of the return histograms.
        [indices]           A tiple of array [x_1, x_2, ..., x_d], where
                            [x_1, ..., x_d] are the indices to np.array of shape
                            [histogram_shape].
        [vals]              The weights to be added to the histogram.
    Rets:
        Numpy array of shape [histogram_shape].
    Example:
        >> compute_histogram((2,2), ([0,1,1,0], [0,1,0,1]), [0.1, 0.3, 0.3, 0.3])
        >> array([[0.1, 0.3],
                  [0.3, 0.3]])
    """
    histogram = np.zeros(histogram_shape)
    assert len(indices)==len(histogram_shape), 'Shape of histogram must match number of indices: {:d},{:d}'.format(len(indices),len(histogram_shape))
    np.add.at(histogram, indices, vals)
    return histogram

import sys, re
from struct import unpack
def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
    return np.flip(np.array(img).reshape((height, width)),axis=0)