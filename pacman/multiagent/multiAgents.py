# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # issue with this, pacman stuck, because it can't go around walls, or the function prefers stuck
        # never get to below 2 distance to any ghost that is not scared
        # if can eat food or capsule just do it
        # if not, then choose action that is closes to food and stay away from walls, reason to stay way from walls is to avoid getting stuck, the avg distance to food is also trying to avoid stuck
        x, y = newPos
        distances_bad = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in successorGameState.getGhostStates() if ghostState.scaredTimer <= 0]
        if distances_bad and min(distances_bad) <= 2:
          return 1000 * min(distances_bad) - 10000 # [-8000, -10000]
        elif currentGameState.getFood()[x][y] or newPos in currentGameState.getCapsules():
          return 10000
        elif newFood.asList() == []:
          return 10000
        else:
          distances = [manhattanDistance(newPos, (x1, y1)) for x1, y1 in newFood.asList()]
          walls = successorGameState.getWalls()
          num_walls = walls[x-1][y]+walls[x+1][y]+walls[x][y-1]+walls[x][y+1]
          return 10000 - min(distances) - float(sum(distances)) / len(distances)- num_walls

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # issue with this sol, pacman stuck because after depth the score does not make a difference whereever it goes, eg when all dots are far away
        # also very slow
        # also did not win the small classic
        def minMax(state, depth, agentIndex):
          if depth == 0:
            return (self.evaluationFunction(state), None)
          else:
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth if next_agent > 0 else  depth -1
            values = []
            actions = state.getLegalActions(agentIndex)
            if actions:
              for action in actions:
                next_state = state.generateSuccessor(agentIndex, action)
                values.append((minMax(next_state, next_depth, next_agent)[0], action))
              values = sorted(values, key = lambda x: x[0])
              if agentIndex == 0:
                return values[-1]
              else:
                return values[0]
            else:
              return minMax(state, next_depth, next_agent)
        value = minMax(gameState, self.depth, 0)
        return value[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    MAX_VALUE = 1e19
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minMax(state, depth, agentIndex, alpha, beta):
          if depth == 0:
            return (self.evaluationFunction(state), None)
          else:
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth if next_agent > 0 else  depth -1
            value = [-self.MAX_VALUE if agentIndex == 0 else self.MAX_VALUE, None]
            actions = state.getLegalActions(agentIndex)
            if actions:
              for action in actions:
                next_state = state.generateSuccessor(agentIndex, action)
                v = minMax(next_state, next_depth, next_agent, alpha, beta)
                if agentIndex == 0:
                  if v[0] > value[0]:
                    value = [v[0], action]
                  if value[0] > beta:
                    return value
                  if value[0] > alpha:
                    alpha = value[0]
                else:
                  if v[0] < value[0]:
                    value = [v[0], action]
                  if value[0] < alpha:
                    return value
                  if value[0] < beta:
                    beta = value[0]
              return value
            else:
              return minMax(state, next_depth, next_agent, alpha, beta)
        value = minMax(gameState, self.depth, 0, -self.MAX_VALUE, self.MAX_VALUE)
        return value[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(state, depth, agentIndex):
          if depth == 0:
            return (self.evaluationFunction(state), None)
          else:
            next_agent = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth if next_agent > 0 else  depth -1
            values = []
            actions = state.getLegalActions(agentIndex)
            if actions:
              for action in actions:
                next_state = state.generateSuccessor(agentIndex, action)
                values.append((expectiMax(next_state, next_depth, next_agent)[0], action))
              values = sorted(values, key = lambda x: x[0])
              if agentIndex == 0:
                return values[-1]
              else:
                vs = [x for x, a in values]
                return [float(sum(vs)) / len(vs), values[0][1]]
            else:
              return expectiMax(state, next_depth, next_agent)
        value = expectiMax(gameState, self.depth, 0)
        return value[1]

def bfs(start_state, goal):
  parent = {}
  start = start_state.getPacmanPosition()
  parent[start] = True
  fringe = [(start_state, 0)]
  while fringe:
    state, steps = fringe.pop(0)
    pos = state.getPacmanPosition()
    if pos == goal:
      return steps
    else:
      actions = state.getLegalActions(0)
      for action in actions:
        next_state = state.generatePacmanSuccessor(action)
        next_pos = next_state.getPacmanPosition()
        if not next_pos in parent:
          fringe.append((next_state, steps+1))
          parent[next_pos] = True
  return 1000

# python pacman.py -p ExpectimaxAgent -l smallClassic -a depth=3,evalFn=betterEvaluationFunction # -q -n 10
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    import numpy as np
    score = currentGameState.getScore() + (1000 - len(currentGameState.getCapsules()))

    # make score between 0 and 0.5
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    x, y = newPos
    distances_bad = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in currentGameState.getGhostStates() if ghostState.scaredTimer <= 0]
    if distances_bad and min(distances_bad) <= 2:
      score -= 0.25 * (3-min(distances_bad))
    elif newFood.asList() == []:
      pass
    else:
      distances = [manhattanDistance(newPos, (x1, y1)) for x1, y1 in newFood.asList()]
      closest = newFood.asList()[np.argmin(np.array(distances))]
      distance = bfs(currentGameState, closest)
      x = distance
      #walls = currentGameState.getWalls()
      #num_walls = walls[x-1][y]+walls[x+1][y]+walls[x][y-1]+walls[x][y+1]
      #x = min(distances) + float(sum(distances)) / len(distances) + num_walls + len(currentGameState.getCapsules())
      score -= (1/(1+math.exp(- x)) - 0.5)
    return score
# Abbreviation
better = betterEvaluationFunction
