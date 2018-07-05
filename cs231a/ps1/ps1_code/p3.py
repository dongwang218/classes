# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    #TODO: Fill in this code
    m1 = float(points[1][1] - points[0][1]) / (points[1][0] - points[0][0])
    m2 = float(points[3][1] - points[2][1]) / (points[3][0] - points[2][0])
    b1 = points[1][1] - m1 * points[1][0]
    b2 = points[3][1] - m2 * points[3][0]
    x = (b2 -b1) / (m1-m2)
    y = m1 * x + b1
    return (x, y)
'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    #TODO: Fill in this code
    """compute K from 3 vanishing points, from three orthogonal planes"""
    # point = K
    v1, v2, v3 = vanishing_points

    A = np.array ([[v1[0]*v2[0] + v1[1]*v2[1], v1[0]+v2[0], v1[1]+v2[1], 1],
                  [v1[0]*v3[0] + v1[1]*v3[1], v1[0]+v3[0], v1[1]+v3[1], 1],
                   [v2[0]*v3[0] + v2[1]*v3[1], v2[0]+v3[0], v2[1]+v3[1], 1]
                  ])
    U, s, Vh = np.linalg.svd(A, full_matrices = True)
    print ('U', U.shape, 'Vh', Vh.shape)
    w = Vh[-1, :]
    W = np.array([[w[0], 0, w[1]],
                  [0, w[0], w[2]],
                  [w[1], w[2], w[3]]])
    kinv = np.linalg.cholesky(W)
    k = np.linalg.pinv(kinv)
    k = k / k[2, 2]
    return k

'''
compute_angle_between_planes
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    #TODO: Fill in this code
    w = np.dot(K, K.T)
    p1 = np.hstack((np.array(vanishing_pair1), [[1], [1]]))
    p2 = np.hstack((np.array(vanishing_pair2), [[1], [1]]))

    L1 = np.cross(p1[0], p1[1])
    L2 = np.cross(p2[0], p2[1])
    cos_theta = np.dot(L1.T, np.dot(w, L2)) / (np.sqrt(np.dot(L1.T, np.dot(w, L1))) * np.sqrt(np.dot(L2.T, np.dot(w, L2))))
    return np.arccos(cos_theta) * 180 / math.pi

'''
compute_rotation_matrix_between_cameras
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    #TODO: Fill in this code
    p1 = np.hstack((np.array(vanishing_points1), [[1]]*len(vanishing_points1)))
    p2 = np.hstack((np.array(vanishing_points2), [[1]]*len(vanishing_points2)))
    kp1 = np.dot(np.linalg.inv(K), p1.T)
    kp2 = np.dot(np.linalg.inv(K), p2.T)

    kp1 = kp1 / np.linalg.norm(kp1, axis=0)
    kp2 = kp2 / np.linalg.norm(kp2, axis=0)

    return np.dot(kp2, np.linalg.inv(kp1))

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
