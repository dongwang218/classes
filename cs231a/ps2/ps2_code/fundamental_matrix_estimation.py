import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    A = np.zeros((points1.shape[0], 9))
    for i in range(points1.shape[0]):
        x, y = points1[i][0], points1[i][1]
        u, v = points2[i][0], points2[i][1]
        A[i, :] = [u*x, u*y, u, v*x, v*y, v, x, y, 1]
    U, s, Vh = np.linalg.svd(A)
    F = np.reshape(Vh[-1, :], (3, 3))
    U, s, Vh = np.linalg.svd(F)
    S = np.diag(s)
    S[2,2] = 0
    return np.dot(U, np.dot(S, Vh))

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    t1 = np.mean(points1, axis = 0)
    c1 = points1 - t1
    s1 = np.sqrt(2) / np.mean(np.linalg.norm(c1[:, :2], axis = -1))
    T1 = np.array([[s1, 0, -s1 * t1[0]],
                   [0, s1, -s1 * t1[1]],
                   [0, 0, 1]])
    p1 = np.dot(T1, points1.T).T

    t2 = np.mean(points2, axis = 0)
    c2 = points2 - t2
    s2 = np.sqrt(2) / np.mean(np.linalg.norm(c2[:, :2], axis = -1))
    T2 = np.array([[s2, 0, -s2 * t2[0]],
                   [0, s2, -s2 * t2[1]],
                   [0, 0, 1]])
    p2 = np.dot(T2, points2.T).T

    F = lls_eight_point_alg(p1, p2)

    return np.dot(T2.T, np.dot(F, T1))

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image
    im2 - a HxW(xC) matrix that contains pixel values from the second image
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
    def get_y(x, line):
        return -(line[0] * x + line[2]) / line[1]

    plt.subplot(1, 2, 1)
    l_prime = np.dot(F, points1.T).T
    for i in range(points1.shape[0]):
        max_x = im2.shape[1]
        plt.plot([0, max_x], [get_y(0, l_prime[i]), get_y(max_x, l_prime[i])], 'r-')
    plt.imshow(im2, cmap='gray')
    plt.scatter(points2[:, 0],  points2[:, 1])
    plt.ylim((im2.shape[0], 0))

    plt.subplot(1, 2, 2)
    l = np.dot(F.T, points2.T).T
    for i in range(points1.shape[0]):
        max_x = im1.shape[1]
        plt.plot([0, max_x], [get_y(0, l[i]), get_y(max_x, l[i])], 'r-')
    plt.imshow(im1, cmap='gray')
    plt.scatter(points1[:, 0],  points1[:, 1])
    plt.ylim((im1.shape[0], 0))
    plt.show()

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    l_prime = np.dot(F, points1.T).T
    distance = np.divide(np.sum(np.multiply(points2, l_prime), axis = -1), np.linalg.norm(l_prime[:, :2], axis = -1))
    return np.mean(np.abs(distance))

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
