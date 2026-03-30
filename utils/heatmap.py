import numpy as np


def get_gaussian_radius(det_size, min_overlap):
    width, height = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    radius = min(r1, r2, r3)
    radius = max(1, int(radius))
    return radius

def set_gaussian_heat(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def set_standard_heat(heatmap, center, radius=1):
    height, width = heatmap.shape[0:2]
    x, y = int(center[0]), int(center[1])
    start_x = max(0, x-radius)
    end_x = min(width, x+radius+1)
    start_y = max(0, y-radius)
    end_y = min(height, y+radius+1)
    region = heatmap[start_y:end_y, start_x:end_x]
    region[region != 1] = -1                # 只修改不等于1的位置
    heatmap[y, x] = 1
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
