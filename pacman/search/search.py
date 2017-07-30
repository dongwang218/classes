# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    def dfs(pos, visited):
      if pos in visited:
        return None

      if problem.isGoalState(pos):
        return []
      else:
        visited[pos] = True
        for successor, action, _ in problem.getSuccessors(pos):
          path = dfs(successor, visited)
          if path is not None:
            return [action] + path
        del visited[pos]
        return None

    # problem is position search
    start = problem.getStartState()
    return dfs(start, {}) # avoid loops

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    parent = {}
    start = problem.getStartState()
    parent[start] = [start, None]
    fringe = [start]
    while fringe:
      pos = fringe.pop(0)
      if problem.isGoalState(pos):
        actions = []
        prev = pos
        while True:
          prev, action = parent[prev]
          if action is None:
            assert prev == start
            return actions
          actions.insert(0, action)
      else:
        for successor, action, _ in problem.getSuccessors(pos):
          if not successor in parent:
            fringe.append(successor)
            parent[successor] = [pos, action]
    raise Exception('no solution')

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # implment tree search, but try to avoid loops and unnecessary by look at cost of path
    parent = {} # node ever in queue, map node to cost, so can avoid insert multiple times into queue unless cost is lower
    expanded = {}
    import Queue
    queue = Queue.PriorityQueue()
    start = problem.getStartState()
    queue.put((0, start))
    parent[start] = (0, start, None)
    while queue:
      cost, pos = queue.get()
      if problem.isGoalState(pos):
        actions = []
        prev = pos
        while True:
          _, prev, action = parent[prev]
          if action is None:
            assert prev == start
            return actions
          actions.insert(0, action)
      elif not pos in expanded:
        expanded[pos] = True
        for successor, action, cost_delta in problem.getSuccessors(pos):
          new_cost = cost + cost_delta
          if not successor in parent or parent[successor][0] > new_cost:
            parent[successor] = (new_cost, pos, action)
            queue.put((new_cost, successor))
    raise Exception('no solution')

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # may need to expand more than once
    parent = {} # node ever in queue, map node to cost, so can avoid insert multiple times into queue unless cost is lower
    import Queue
    queue = Queue.PriorityQueue()
    start = problem.getStartState()
    queue.put((0 + heuristic(start, problem), (0, start)))
    parent[start] = (0, start, None)
    while queue:
      cost, (actual_cost, pos) = queue.get()
      if problem.isGoalState(pos):
        actions = []
        prev = pos
        while True:
          _, prev, action = parent[prev]
          if action is None:
            assert prev == start
            return actions
          actions.insert(0, action)
      else:
        for successor, action, cost_delta in problem.getSuccessors(pos):
          new_cost = actual_cost + cost_delta
          if not successor in parent or parent[successor][0] > new_cost:
            parent[successor] = (new_cost, pos, action)
            queue.put((new_cost + heuristic(successor, problem), (new_cost, successor)))
    raise Exception('no solution')


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
