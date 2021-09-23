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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood().asList()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        
        # keep ghost distance high
        ghosts = 0
        for ghost in newGhostStates:
            timer = (ghost.scaredTimer)
            ghost = ghost.getPosition()
            ghosts = ghosts + util.manhattanDistance(newPos,ghost)
            if ghosts > 8 and timer == 0:
                ghosts = 4
        
        # want food distance low
        foods = 0
        for food in newFood:
            foods = foods + util.manhattanDistance(newPos, food)
        
        if foods == 0:
            foods = 1
            
        # flip ghost distance when on timer so that pacman will go towards ghosts rather than away
        if timer > 0:
            eval_func = foods / ghosts + childGameState.getScore()
        else:
            eval_func = ghosts / foods + childGameState.getScore()
        return eval_func

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
    
    def getMaxValue(self,state, depth):
        #print('max')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = float('-inf')
        agent = 0
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            score = self.getMinValue(next_state, depth, 1)
            if score > v:
                v = score
        return v
    
    
    def getMinValue(self,state, depth, agent):
        #print('min')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = float('inf')
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            if agent == state.getNumAgents()-1:
                score = self.getMaxValue(next_state, depth+1)
            else:
                score = self.getMinValue(next_state, depth, agent+1)
            if score < v:
                v = score
        return v


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        depth = 1
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            score = self.getMinValue(next_state, depth, 1)
            if score > v:
                v = score
                true_action = action
                
        return true_action
                        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getMaxValue(self,state, depth, alpha = float('-inf'), beta = float('inf')):
        #print('max')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = float('-inf')
        agent = 0
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            score = self.getMinValue(next_state, depth, 1, alpha, beta)
            if score > v:
                v = score
                if v > beta:
                    break
            alpha = max(alpha, v)
        return v
    
    
    def getMinValue(self,state, depth, agent, alpha = float('-inf'), beta = float('inf')):
        #print('min')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = float('inf')
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            if agent == state.getNumAgents()-1:
                score = self.getMaxValue(next_state, depth+1, alpha, beta)
            else:
                score = self.getMinValue(next_state, depth, agent+1, alpha, beta)
            if score < v:
                v = score
                if v < alpha:
                    break
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = 1
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            score = self.getMinValue(next_state, depth, 1, alpha, beta)
            if score > v:
                v = score
                true_action = action
                if v > beta:
                    break
            alpha = max(alpha, v)
                
        return true_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getMaxValue(self,state, depth):
        #print('max')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = float('-inf')
        agent = 0
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            score = self.getExpValue(next_state, depth, 1)
            if score > v:
                v = score
        return v
    
    
    def getExpValue(self,state, depth, agent):
        #print('min')
        if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
        v = 0
        for action in state.getLegalActions(agent):
            next_state = state.getNextState(agent, action)
            if agent == state.getNumAgents()-1:
                score = self.getMaxValue(next_state, depth+1)
            else:
                score = self.getExpValue(next_state, depth, agent+1)
            v += score/len(state.getLegalActions(agent))
        return v
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        depth = 1
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            next_state = gameState.getNextState(0, action)
            score = self.getExpValue(next_state, depth, 1)
            if score > v:
                v = score
                true_action = action
                
        return true_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I just used the same evaluation fundtion from problem 1 
                    with the currentGameState instead of childGameState
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    
    # keep ghost distance high
    ghosts = 0
    for ghost in newGhostStates:
        timer = (ghost.scaredTimer)
        ghost = ghost.getPosition()
        ghosts = ghosts + util.manhattanDistance(newPos,ghost)
        if ghosts > 8 and timer == 0:
            ghosts = 4
    
    # want food distance low
    foods = 0
    for food in newFood:
        foods = foods + util.manhattanDistance(newPos, food)
    
    if foods == 0:
        foods = 1
        
    # flip ghost distance when on timer so that pacman will go towards ghosts rather than away
    if timer > 0:
        eval_func = foods / ghosts + currentGameState.getScore()
    else:
        eval_func = ghosts / foods + currentGameState.getScore()
    return eval_func
# Abbreviation
better = betterEvaluationFunction
