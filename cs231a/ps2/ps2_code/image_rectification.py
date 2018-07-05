import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
    U, s, Vh = np.linalg.svd(F)
    e_prime = Vh[-1, :]
    return e_prime / e_prime[2]
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    h, w = im2.shape[:2]
    T = np.eye(3)
    T[:2, -1] = -0.5 * np.array([w, h])
    ep = T.dot(e2)
    n = np.linalg.norm(ep[:2])
    a = 1 if ep[0] >= 0 else -1
    R = np.array([[a * ep[0]/n, a *ep[1]/n, 0],
                  [-a * ep[1]/n, a * ep[0]/n, 0],
                  [0, 0, 1] ])
    f = R.dot(ep)[0]
    G = np.eye(3)
    G[2, 0] = -1.0/f
    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    ep = np.array([[0, -e2[2], e2[1]],
                   [e2[2], 0, -e2[0]],
                   [-e2[1], e2[0], 0]])
    M = ep.dot(F) + np.outer(e2, np.ones((1, 3)))

    p1h = H2.dot(M.dot(points1.T)).T
    p2h = H2.dot(points2.T).T
    b = p2h / p2h[:, -1].reshape((-1, 1))
    W = p1h / p1h[:, -1].reshape((-1, 1))
    a = np.linalg.lstsq(W, b[:,0])[0]
    HA = np.eye(3)
    HA[0] = a
    H1 = HA.dot(H2.dot(M))

    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
