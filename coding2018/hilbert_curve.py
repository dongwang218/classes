import numpy as np

def translate(p, dxy):
  return [[point[0]+dxy[0], point[1]+dxy[1]] for point in p]

def rotate(p, degree):
  assert(degree in [90, -90])
  assert(p[0] == [0, 0])
  if degree == -90:
    matrix = np.array([[0, 1],
                       [-1, 0]])
  else:
    matrix = np.array([[0, -1],
                       [1, 0]])

  r = np.dot(matrix, np.array(p).T)
  r = r.T
  result = []
  for row in r:
    result.append(row.tolist())

  return result

def hilbert(n):
  """
  n is a power of 2, print a traversal of hilbert curve from (0,0) end at (n-1, 0)
  """

  if n == 2:
    return [[0, 0], [0, 1], [1, 1], [1, 0]]
  piece = hilbert(n/2)
  bottom_left = translate(rotate(piece, -90), (0, n/2-1))
  bottom_left.reverse()
  assert(bottom_left[0] == [0, 0])
  top_left = translate(piece, (0, n/2))
  top_right = translate(top_left, (n/2, 0))
  bottom_right = translate(rotate(piece, 90), (n-1, 0))
  bottom_right.reverse()
  assert(bottom_right[-1] == [n-1, 0])
  return bottom_left + top_left + top_right + bottom_right

if __name__ == '__main__':
  for n in [64]:
    p = hilbert(n)
    print(p)

    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    window.screensize(n+2, n+2)
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    chaser_robot.penup()
    chaser_robot.goto(0, 0)
    chaser_robot.pendown()

    for point in p:
        chaser_robot.goto(point[0]*5, point[1]*5)

    turtle.exitonclick()
