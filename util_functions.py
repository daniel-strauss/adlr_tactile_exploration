import numpy
import numpy as np


# converts imgarray to list of points
def img_array_to_p_list(a):
    return np.argwhere(a > 0)


def from_torch(a):
    return a[0].cpu().detach().numpy()


# converts a n,m array to a n,m,1 array
def add_color_dim(a):
    return a.reshape((1, a.shape[0], a.shape[1]))


# converts array of shape n,m,c to shape c,n,m
def convert_for_imshow(a, cs=None, bin=True):
    if a.dtype == np.uint8:
        a = a.astype(np.float32)
        a = a / 255

    if not cs is None:
        if bin:
            eps = 0.01
            df = 0.4
            img = numpy.zeros(a.shape)
            img += cs[0][:, None, None] * a[0]
            img += cs[1][:, None, None] * a[1]
            img += cs[2][:, None, None] * a[2]
            img += cs['background'][:, None, None] * (np.sum(a, axis=0) <= 0.01)
        else:
            eps = 0.01
            bg = cs['background'][:, None, None]
            img = numpy.zeros(a.shape)
            for i in range(3):
                img += (bg * (1 - a[i]) + cs[i][:, None, None] * a[i]) * (a[i] > eps)

            img += bg * (np.sum(a, axis=0) <= eps)

        a = np.clip(img, 0, 1)

    return a.transpose(2, 1, 0)


def two_img_to_one(a, b):
    if a.shape[1:-1] != b.shape[1:-1]:
        print("WRONG SHAPES:")
        print(a.shape)
        print(b.shape)
        raise ValueError("images are not compatible, need same amount of rows and cols")
    return np.concatenate((a, b), axis=0)


def add_zero_channel(a):
    zeros = add_color_dim(np.zeros(a.shape[1:], dtype=a.dtype))
    return two_img_to_one(a, zeros)
