import numpy as np
import matplotlib.pyplot as plt

from img_file import img_denormalize

def plot_histogram(img: np.ndarray):
    if len(img.shape) == 3:
        plot_histogram_rgb(img)
    else:
        plot_histogram_gs(img)

def plot_histogram_gs(img: np.ndarray):
    try:
        bins = np.zeros(256, dtype=np.float32)
        data = img_denormalize(img)
        def count(px: np.uint8):
            bins[px] += 1
        np.vectorize(count)(data)

        bins /= bins.max()
        plt.plot(bins, color='black')
        # TODO: img write
        plt.show()
        
    finally:
        plt.clf()

def plot_histogram_rgb(img: np.ndarray):
    try:
        # fig, (ax1, ax2) = plt.subplots(1, 2, plt.figure)
        b_bins = np.zeros(256, dtype=np.float32)
        g_bins = np.zeros(256, dtype=np.float32)
        r_bins = np.zeros(256, dtype=np.float32)

        data = img_denormalize(img)
        b, g, r = np.split(data, 3)

        def b_count(px: np.ndarray): b_bins[px] += 1
        def g_count(px: np.ndarray): g_bins[px] += 1
        def r_count(px: np.ndarray): r_bins[px] += 1

        np.vectorize(b_count)(b)
        np.vectorize(g_count)(g)
        np.vectorize(r_count)(r)

        b_bins /= b_bins.max()
        g_bins /= g_bins.max()
        r_bins /= r_bins.max()

        plt.plot(b_bins, color='blue', alpha=0.7)
        plt.plot(g_bins, color='green', alpha=0.7)
        plt.plot(r_bins, color='red', alpha=0.7)

        # TODO: img write
        plt.show()
        
    finally:
        plt.clf()

