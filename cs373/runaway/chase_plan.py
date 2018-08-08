# ----------
# Part Four
#
# Again, you'll track down and recover the runaway Traxbot.
# But this time, your speed will be about the same as the runaway bot.
# This may require more careful planning than you used last time.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time.
#
# ----------
# GRADING
#
# Same as part 3. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrix import *
import random

def estimate_next_pos(measurement, steps=1, OTHER = None):
  #print('measurement', measurement)
  if OTHER is None:
    return [measurement for i in range(steps)], {'cache': [measurement]}
  elif 'mu' not in OTHER:
    if len(OTHER['cache']) < 3:
      OTHER['cache'].append(measurement)
      return [measurement for i in range(steps)], OTHER
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

    return [measurement for i in range(steps)], OTHER
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

  xy_estimate = []
  for i in range(steps):
      next_heading = heading + turn
      next_x = x + distance * cos(next_heading)
      next_y = y + distance * sin(next_heading)
      xy_estimate.append([next_x, next_y])
      x = next_x
      y = next_y
      heading = next_heading

  #print('xy_estimate', xy_estimate, 'mu', mu, 'mu_hat', mu_hat)
  return xy_estimate, OTHER

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves.

    targets_next, OTHER = estimate_next_pos(target_measurement, 3, OTHER)
    goal = targets_next[-1]
    for i in range(len(targets_next)):
        if distance_between(targets_next[i], hunter_position) < max_distance:
            goal = targets_next[i]
    heading_to_target = get_heading(hunter_position, goal)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = distance_between(goal, hunter_position)

    return turning, distance, OTHER

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught



def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def demo_grading_vis(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.97 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization
    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()
        #Visualize it
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught


def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all
    the target measurements, hunter positions, and hunter headings over time, but it doesn't
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables

    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER

target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = .05*target.distance
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

print demo_grading_vis(hunter, target, next_move)
