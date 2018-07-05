# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('./image1.jpg')
    img2 = misc.imread('./image2.jpg')

    plt.subplot(1, 2, 1)
    plt.title("original image1")
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title("original image2")
    plt.imshow(img2)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    def imagescaling(img):
        pixmax = np.max(img)
        pixmin = np.min(img)
        return (img - pixmin) / (pixmax - pixmin)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # scale images
    img1 = imagescaling(img1)
    img2 = imagescaling(img2)

    plt.subplot(1, 2, 1)
    plt.title("image1 w/ double precision")
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title("image2 w/ double precision")
    plt.imshow(img2)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.

    # BEGIN YOUR CODE HERE
    img_sum = imagescaling(img1+img2)

    plt.title("sum of img1 and img2")
    plt.imshow(img_sum)
    plt.show()

    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    newImage1 = np.zeros(img1.shape, dtype=np.float64)
    newImage1[:, 0:img1.shape[1]//2] = img1[:, 0:img1.shape[1]//2]
    newImage1[:, img1.shape[1]//2:] = img2[:, img1.shape[1]//2:]
    plt.title("half/half img1 and img2")
    plt.imshow(newImage1)
    plt.show()


    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.zeros(img1.shape, dtype=np.float64)
    for i in range(img1.shape[0]):
      if i % 2 == 0:
        newImage2[i, :] = img1[i, :]
      else:
        newImage2[i, :] = img2[i, :]
    plt.title("mix img1 and img2")
    plt.imshow(newImage2)
    plt.show()

    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    mask = np.tile(np.reshape(np.arange(img1.shape[0]), (img1.shape[0], 1, 1)), (1, img1.shape[1], 3)) % 2 == 0

    newImage3 = np.where(mask, img1, img2)
    plt.title("mix img1 and img2")
    plt.imshow(newImage3)
    plt.show()

    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    newImage4 = np.sum(newImage3, axis=-1)/3
    plt.title("gray")
    plt.imshow(newImage4, cmap=plt.cm.Greys_r)
    plt.show()

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
