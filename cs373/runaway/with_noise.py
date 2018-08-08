# ----------
# Part Two
#
# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position.
#
# ----------
# GRADING
#
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import * # Check the matrix.py tab to see how this works.
import random

# This is the function you have to write. Note that measurement is a
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.
def estimate_next_pos_avg(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""

    # You must return xy_estimate (x, y), and OTHER (even if it is None)
    # in this order for grading purposes.
    if OTHER is None:
        OTHER = []
    OTHER.append(measurement)
    if len(OTHER) > 2:
        distance = []
        heading = []
        for i in range(len(OTHER)-1):
            dx = OTHER[i+1][0] - OTHER[i][0]
            dy = OTHER[i+1][1] - OTHER[i][1]
            distance.append(sqrt(dx**2+dy**2))
            heading.append(atan2(dy, dx))
        avg_distance = sum(distance) / len(distance)
        turn = [(heading[i+1] - heading[i]) % (2*pi) for i in range(len(heading)-1)]
        avg_turn = sum(turn) / len(turn)

        heading = heading[-1] + avg_turn
        xy_estimate = [measurement[0] + cos(heading) * avg_distance, measurement[1] + sin(heading) * avg_distance]
    else:
        xy_estimate = measurement
    print(measurement, xy_estimate)
    return xy_estimate, OTHER

def estimate_next_pos_ekf(measurement, OTHER = None):
  #print('measurement', measurement)
  if OTHER is None:
    return measurement, {'cache': [measurement]}
  elif 'mu' not in OTHER:
    if len(OTHER['cache']) < 3:
      OTHER['cache'].append(measurement)
      return measurement, OTHER
    # we have three points now
    distance = []
    heading = []
    cache = OTHER['cache']
    for i in range(len(cache)-1):
      dx = cache[i+1][0] - cache[i][0]
      dy = cache[i+1][1] - cache[i][1]
      distance.append(sqrt(dx**2+dy**2))
      heading.append(atan2(dy, dx))
    avg_distance = sum(distance) / len(distance)
    turn = [(heading[i+1] - heading[i]) % (2*pi) for i in range(len(heading)-1)]
    avg_turn = sum(turn) / len(turn)

    #print('avg_distance', avg_distance, 'avg_turn', avg_turn, 'heading', heading)
    # x, y, heading, distance, turn
    #mu = matrix([[measurement[0]], [measurement[1]], [heading[-1]], [avg_distance], [avg_turn]])
    mu = matrix([[measurement[0]], [measurement[1]], [heading[-1]], [avg_distance], [avg_turn]])

    sigma =  matrix([[10, 0, 0, 0, 0],
                 [0, 10, 0, 0, 0],
                 [0, 0, 10, 0, 0],
                 [0, 0, 0, 10, 0],
                 [0, 0, 0, 0, 10]])# initial uncertainty: 0 for positions x and y, 1000 for the two velocities
    OTHER = {'mu': mu, 'sigma': sigma}

    return measurement, OTHER
  else:
    u = matrix([[0.], [0.], [0.], [0.], [0.]]) # external motion
    mu = OTHER['mu']
    sigma = OTHER['sigma']
    x, y, heading, distance, turn = mu.value[0][0], mu.value[1][0], mu.value[2][0], mu.value[3][0], mu.value[4][0]
    # prediction
    mu_hat = matrix([[x + distance * cos(heading + turn)],
                     [y + distance * sin(heading + turn)],
                     [heading + turn],
                     [distance], [turn]])

    G = matrix([[1, 0, -distance*sin(heading+turn), cos(heading+turn), -distance*sin(heading+turn)],
                [0, 1, distance*cos(heading+turn), sin(heading+turn), distance*cos(heading+turn)],
                [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    sigma_hat = G * sigma * G.transpose() # no motion noise yet


    # measurement update
    H =  matrix([[1, 0, 0, 0, 0], [0,1,0,0, 0]])# measurement function: reflect the fact that we observe x and y but not the two velocities
    R =  matrix([[measurement_noise, 0], [0, measurement_noise]])# measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
    S = H * sigma_hat * H.transpose() + R
    #print('S', S, 'y', y, 'G', G)
    K = sigma_hat * H.transpose() * S.inverse_np()

    Z = matrix([measurement])
    Y = Z.transpose() - (H * mu_hat)
    mu = mu_hat + (K * Y)
    I =  matrix([[]])
    I.identity(5)
    sigma = (I - (K * H)) * sigma_hat

  OTHER = {'mu': mu, 'sigma': sigma}

  # make a prediction
  x, y, heading, distance, turn = mu.value[0][0], mu.value[1][0], mu.value[2][0], mu.value[3][0], mu.value[4][0]

  xy_estimate = [x + distance * cos(heading + turn),
                 y + distance * sin(heading + turn)]

  #print('xy_estimate', xy_estimate, 'mu', mu, 'mu_hat', mu_hat)
  return xy_estimate, OTHER

def estimate_next_pos(measurement, OTHER = None):
  return estimate_next_pos_ekf(measurement, OTHER)

# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        if ctr == 1000:
            print "Sorry, it took you too many steps to localize the target."
    return localized

# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER = None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER: # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER

# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

demo_grading(estimate_next_pos, test_target)
