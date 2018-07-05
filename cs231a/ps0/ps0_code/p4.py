# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = None
    img1 = misc.imread('./image1.jpg', 'L')

    # Reconstruction based on reduced SVD
    U, s, V = np.linalg.svd(img1, full_matrices=False)
    S = np.diag(s)
    imgSVD = np.dot(U, np.dot(S, V))

    plt.subplot(1, 2, 1)
    plt.title("original image1")
    plt.imshow(img1, cmap=plt.cm.Greys_r)

    plt.subplot(1, 2, 2)
    plt.title("reconstruction based on SVD")
    plt.imshow(imgSVD, cmap=plt.cm.Greys_r)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    def low_rank_approx(SVD, r=1):
        u, s, v = SVD
        Ar = np.zeros((u.shape[0], v.shape[1]))
        for i in xrange(r):
            Ar += s[i] * np.outer(u[:,i], v[i])
        return Ar
    rank1approx = low_rank_approx((U, s, V))
    plt.title("rank1approx")
    plt.imshow(rank1approx, cmap=plt.cm.Greys_r)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank20approx = low_rank_approx((U, s, V), 20)
    plt.title("rank20approx")
    plt.imshow(rank20approx, cmap=plt.cm.Greys_r)
    plt.show()

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
