import os
import numpy
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

import jlinops

def read_image(fname: str, normalize: bool = True, grayscale: bool = True) -> np.ndarray:
    """
    Reads

    :param fname: name of the image file located in ``\jlinops\data\images``.
    :param normalize: whether to normalize the image so that all channels lie in [0,1].
    :param grayscale: whether to convert the image to grayscale.
    """    

    path = os.path.join(os.path.dirname(jlinops.__file__), "data/images")
    path = os.path.join(path, fname)
    img = Image.open(path)

    if grayscale:
        img = ImageOps.grayscale(img)

    img = np.asarray(img)

    if normalize:
        img = img/255.0
    
    return img


def ge(**kwargs) -> np.ndarray:
    """
    Returns the GE test image.
    """
    return read_image("GEimage.png", **kwargs)


def cameraman(**kwargs) -> np.ndarray:
    """
    Returns the cameraman test image.
    """
    return read_image("cameraman.png", **kwargs)


def grandcanyon(**kwargs) -> np.ndarray:
    """
    Returns an image of the Grand Canyon.
    """
    return read_image("grand_canyon.jpg", **kwargs)


def shepplogan(**kwargs) -> np.ndarray:
    """
    Returns the Shepp-Logan phantom test image.
    """
    return read_image("SheppLogan_Phantom.png", **kwargs)


def fingerprint(**kwargs) -> np.ndarray:
    """
    Returns the fingerprint test image.
    """
    return read_image("fingerprint.jpg", **kwargs)


def cortex(**kwargs) -> np.ndarray:
    """
    Returns the cortex test image.
    """
    return read_image("cortex.bmp", **kwargs)


def satellite(**kwargs) -> np.ndarray:
    """
    Returns the satellite test image.
    """
    return read_image("satellite.png", **kwargs)


def seaice(**kwargs) -> np.ndarray:
    """
    Returns the sea ice test image.
    """
    return read_image("detailed_arctic_sea_ice.jpg", **kwargs)


def mri(**kwargs) -> np.ndarray:
    """
    Returns the mri test image.
    """
    return read_image("mri.png", **kwargs)


def graecolatinsquare(which=3, **kwargs) -> np.ndarray:
    """
    Returns the graeco-latin square test image.
    """
    valid_sizes = [3, 4, 5, 10]
    assert which in valid_sizes, f"Invalid number for square, must be one of {valid_sizes}"
    return read_image(f"graecolatinsquare{which}.png", **kwargs)


def sar1(**kwargs) -> np.ndarray:
    """
    Returns the sar1 square test image.
    """
    return read_image("sar1.jpg", **kwargs)


def sar2(**kwargs) -> np.ndarray:
    """
    Returns the sar2 square test image.
    """
    return read_image("sar2.png", **kwargs)

def meme(**kwargs) -> np.ndarray:
    """
    Returns the meme test image.
    """
    return read_image("meme.jpg", **kwargs)
















