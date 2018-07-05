import numpy as np
import skimage.io as sio
from scipy.io import loadmat
from plotting import *
import math
from scipy.signal import convolve2d

'''
COMPUTE_GRADIENT Given an image, computes the pixel gradients

Arguments:
    im - a grayscale image, represented as an ndarray of size (H, W) containing
        the pixel values

Returns:
    angles - ndarray of size (H-2, W-2) containing gradient angles in degrees

    magnitudes - ndarray of size (H-2, W-2) containing gradient magnitudes

The way that the angles and magnitude per pixel are computed as follows:
Given the following pixel grid

    P1 P2 P3
    P4 P5 P6
    P7 P8 P9

We compute the angle on P5 as arctan(dy/dx) = arctan(P2-P8 / P4-P6).
Note that we should be using np.arctan2, which is more numerically stable.
However, this puts us in the range [-180, 180] degrees. To be in the range
[0,180], we need to simply add 180 degrees to the negative angles.

The magnitude is simply sqrt((P4-P6)^2 + (P2-P8)^2)
'''
def compute_gradient(im):
    # TODO: Implement this method!
    dx_filter = np.zeros((3,3))
    dx_filter[1, 0] = -1
    dx_filter[1, 2] = 1
    dx = convolve2d(im, dx_filter, mode='valid')

    dy_filter = np.zeros((3,3))
    dy_filter[0, 1] = -1
    dy_filter[2, 1] = 1
    dy = convolve2d(im, dy_filter, mode='valid')

    all = np.concatenate((np.expand_dims(dx, axis = -1), np.expand_dims(dy, axis = -1)), axis=-1)
    mag = np.linalg.norm(all, axis = -1)
    angles = np.arctan2(dy, dx) * 180 / np.pi
    angles = np.where(angles >= 0, angles, angles + 180)
    return angles, mag

'''
GENERATE_HISTOGRAM Given matrices of angles and magnitudes of the image
gradient, generate the histogram of angles

Arguments:
    angles - ndarray of size (M, N) containing gradient angles in degrees

    magnitudes - ndarray of size (M, N) containing gradient magnitudes

    nbins - the number of bins that you want to bin the histogram into

Returns:
    histogram - an ndarray of size (nbins,) containing the distribution
        of gradient angles.

This method should be implemented as follows:

1)Each histogram will bin the angle from 0 to 180 degrees. The number of bins
will dictate what angles fall into what bin (i.e. if nbins=9, then first bin
will contain the votes of angles close to 10, the second bin will contain those
close to 30, etc).

2) To create these histogram, iterate over the gradient grids, putting each
gradient into its respective bins. To do this properly, we interpolate and
weight the voting by both its magnitude and how close it is to the average
angle of the two bins closest to the angle of the gradient. For example, if we
have nbins = 9 and receive angle of 20 degrees with magnitude 1, then we the
vote contribution to the histogram weights equally with the first and second bins
(since its closest to both 10 and 30 degrees). If instead, we recieve angle of
25 degrees with magnitude 2, then it is weighted 25% in the first bin and 75%
in second bin, but with twice the voting power.

Mathematically, if you have an angle, magnitude, the center_angle1 of the lower
bin center_angle2 of the higher bin, then:

histogram[bin1] += magnitude * |angle - center_angle2| / (180 / nbins)
histogram[bin2] += magnitude * |angle - center_angle1| / (180 / nbins)

Notice how that we're weighting by the distance to the opposite center. The
further the angle is from one center means it is closer to the opposite center
(and should be weighted more).

One special case you will have to take care of is when the angle is near
180 degrees or 0 degrees. It may be that the two nearest bins are the first and
last bins respectively.OA
'''
def generate_histogram(angles, magnitudes, nbins = 9):
    # TODO: Implement this method!
    bin_width = 180.0/nbins
    bin = np.maximum(0, np.floor((angles-1e-6) / bin_width))
    other_bin = np.floor(angles / bin_width + 0.5)
    other_bin = np.where(other_bin != bin, other_bin, bin-1)

    v1 = np.multiply(magnitudes, np.abs(angles - (other_bin + 0.5)*bin_width) / bin_width)
    v2 = np.multiply(magnitudes, np.abs(angles - (bin + 0.5)*bin_width) / bin_width)

    other_bin = np.where(other_bin == nbins, 0, np.where(other_bin < 0, nbins-1, other_bin))
    histo = np.bincount(bin.astype(np.int).reshape(-1), v1.reshape(-1))
    if histo.shape[0] < nbins:
      histo = np.hstack((histo, np.zeros((nbins-histo.shape[0]))))
    histo2 = np.bincount(other_bin.astype(np.int).reshape(-1), v2.reshape(-1))
    if histo2.shape[0] < nbins:
      histo2 = np.hstack((histo2, np.zeros((nbins-histo2.shape[0]))))

    return histo + histo2

'''
COMPUTE_HOG_FEATURES Computes the histogram of gradients features
Arguments:
    im - the image matrix

    pixels_in_cell - each cell will be of size (pixels_in_cell, pixels_in_cell)
        pixels

    cells_in_block - each block will be of size (cells_in_block, cells_in_block)
        cells

    nbins - number of histogram bins

Returns:
    features - the hog features of the image represented as an ndarray of size
        (H_blocks, W_blocks, cells_in_block * cells_in_block * nbins), where
            H_blocks is the number of blocks that fit height-wise
            W_blocks is the number of blocks that fit width-wise

Generating the HoG features can be done as follows:

1) Compute the gradient for the image, generating angles and magnitudes

2) Define a cell, which is a grid of (pixels_in_cell, pixels_in_cell) pixels.
Also, define a block, which is a grid of (cells_in_block, cells_in_block) cells.
This means each block is a grid of side length pixels_in_cell * cell_in_block
pixels.

3) Pass a sliding window over the image, with the window size being the size of
a block. The stride of the sliding window should be half the block size, (50%
overlap). Each cell in each block will store a histogram of the gradients in
that cell. Consequently, there will be cells_in_block * cells_in_block
histograms in each block. This means that each block feature will initially
represented as a (cells_in_block, cells_in_block, nbins) ndarray, that can
reshaped into a (cells_in_block * cells_in_block *nbins,) ndarray. Make sure to
normalize such that the norm of this flattened block feature is 1.

4) The overall hog feature that you return will be a grid of all these flattened
block features.

Note: The final overall feature ndarray can be flattened if you want to use to
train a classifier or use it as a feature vector.
'''
def compute_hog_features(im, pixels_in_cell, cells_in_block, nbins):
    # TODO: Implement this method!
    angles, magnitude = compute_gradient(im)

    h, w = angles.shape[:2]
    cell_size = pixels_in_cell
    block_size = cells_in_block * pixels_in_cell
    stride = block_size // 2
    assert(stride % pixels_in_cell == 0)
    stride_cell = stride // pixels_in_cell

    h_cells = int(np.ceil(h / pixels_in_cell))
    w_cells = int(np.ceil(w / pixels_in_cell))
    histo = np.zeros((h_cells, w_cells, nbins))
    for i in range(h_cells):
      i_start = i*pixels_in_cell
      for j in range(w_cells):
        j_start = j * pixels_in_cell
        histo[i, j] = generate_histogram(angles[i_start : (i_start+pixels_in_cell), j_start:(j_start+pixels_in_cell)], magnitude[i_start : (i_start+pixels_in_cell), j_start:(j_start+pixels_in_cell)], nbins)


    # if not enough cells, we will discard it otherwise the hog feature length will differ
    h_blocks = ((h_cells - cells_in_block) // stride_cell) + 1
    w_blocks = ((w_cells - cells_in_block) // stride_cell) + 1
    hog = np.zeros((h_blocks, w_blocks, cells_in_block*cells_in_block*nbins))
    for i in range(h_blocks):
      i_start = i * stride_cell
      for j in range(w_blocks):
        j_start = j * stride_cell
        f = histo[i_start : (i_start+cells_in_block), j_start : (j_start+cells_in_block)].reshape(-1)
        hog[i, j] = f / np.linalg.norm(f)
    return hog


if __name__ == '__main__':
    # Part A: Checking the image gradient
    print '-' * 80
    print 'Part A: Image gradient'
    print '-' * 80
    im = sio.imread('simple.jpg', True)
    grad_angle, grad_magnitude = compute_gradient(im)
    print "Expected angle: 126.339396329"
    print "Expected magnitude: 0.423547566786"
    print "Checking gradient test case 1:", \
        np.abs(grad_angle[0][0] - 126.339396329) < 1e-3 and \
        np.abs(grad_magnitude[0][0] - 0.423547566786) < 1e-3
    im = np.array([[1, 2, 2, 4, 8],
                    [3, 0, 1, 5, 10],
                    [10, 13, 12, 2, 7],
                    [10, 5, 1, 0, 3],
                    [1, 1, 1.5, 2, 2.5]])
    grad_angle, grad_magnitude = compute_gradient(im)
    correct_angle = np.array([[ 100.30484647,   63.43494882,  167.47119229],
                              [  68.19859051,    0.        ,   45.        ],
                              [  53.13010235,   64.53665494,  180.        ]])
    correct_magnitude = np.array([[ 11.18033989,  11.18033989,   9.21954446],
                                  [  5.38516481,  11.        ,   7.07106781],
                                  [ 15.        ,  11.62970335,   2.        ]])
    print "Expected angles: \n", correct_angle
    print "Expected magnitudes: \n", correct_magnitude
    print "Checking gradient test case 2:", \
        np.allclose(grad_angle, correct_angle) and \
        np.allclose(grad_magnitude, correct_magnitude)

    # Part B: Checking the histogram generation
    print '-' * 80
    print 'Part B: Histogram generation'
    print '-' * 80
    angles = np.array([[10, 30, 50], [70, 90, 110], [130, 150, 170]])
    magnitudes = np.arange(1,10).reshape((3,3))
    print "Checking histogram test case 1:", \
        np.all(generate_histogram(angles, magnitudes, nbins = 9) == np.arange(1,10))

    angles = np.array([[20, 40, 60], [80, 100, 120], [140, 160, 180]])
    magnitudes = np.arange(1,19,2).reshape((3,3))
    histogram = generate_histogram(angles, magnitudes, nbins = 9)
    print "Checking histogram test case 2:", \
        np.all(histogram  == np.array([9, 2, 4, 6, 8, 10, 12, 14, 16]))

    angles = np.array([[13, 23, 14.3], [53, 108, 1], [77, 8, 32]])
    magnitudes = np.ones((3,3))
    histogram = generate_histogram(angles, magnitudes, nbins = 9)
    print "Submit these results:", histogram

    # Part C: Computing and displaying the final HoG features
    # vary cell size to change the output feature vector. These parameters are common parameters
    pixels_in_cell = 8
    cells_in_block = 2
    nbins = 9
    im = sio.imread('car.jpg', True)
    car_hog_feat = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
    show_hog(im, car_hog_feat, figsize = (18,6))
