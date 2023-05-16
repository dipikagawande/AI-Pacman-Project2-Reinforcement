# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        ## Initialize a "dictionary" (Counter object) of Q(s,a) values
        self.dict_Q_sa = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        ## Can simply return the value from the dictionary Counter object
        return self.dict_Q_sa[(state, action)]
    

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        ## If s is a Terminal state, return V(s)=0
        if not self.getLegalActions(state):
          return 0.0
        ## Else if s is not a Terminal state:
        else:
          ## Initiate empty list of Q(s,a) values at this state
          list_Q_sa = []
          ## For each possible action at this state s:
          for action in self.getLegalActions(state):
              ## Calculate Qi(s,a) and add to Q-val list
              list_Q_sa.append(self.getQValue(state, action))
              ## V(s) is the max Q(s,a)
          return max(list_Q_sa)


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        ## If s is a Terminal state, return pi(s)=None
        if not self.getLegalActions(state):
          return None
        ## Else if s is not a Terminal state:
        else:
          ## Initialize a dict of actions and their Q(s,a) at this state
          dict_actionsQs = {}
          ## For each possible action a at this state s:
          for action in self.getLegalActions(state):
              ## Calculate Q(s,a)...
              Q_sa = self.getQValue(state, action)
              ## ...and store it in the dictionary
              dict_actionsQs[action] = Q_sa

          maxValue = self.getValue(state)
          ## All actions (dict keys) whose Q = max(Q) are tied for best action.
          ## Below line of code from StackOverflow post: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
          tiedForBest = [key for key, value in dict_actionsQs.items() if value == maxValue]
          #return max(dict_actionsQs, key = dict_actionsQs.get) 
          ## Pick pi(s) at random out of the tied-for-best actions
          return random.choice(tiedForBest)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        ## If we get 1 with p = epsilon (epsilon = exploration probability):
        if util.flipCoin(self.epsilon):
            ## Choose a random action from the legal actions at this state
            action = random.choice(legalActions)
        ## If we get 0 (with p = 1-epsilon)
        else:
          ## Take the optimal action pi(s), the argmax of Q(s, a) values
          action = self.getPolicy(state)

        return action
    

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

          This update function follows the formula:
          Q(s,a) = (1-alpha) * Q(s,a) + alpha*Qhat(s,a),
          where:
          Qhat(s,a) = R(s,a) + gamma * MAX_{a'} Q(s',a')
        """
        "*** YOUR CODE HERE ***"
        ## Name the parameters suggestively
        gamma = self.discount
        alpha = self.alpha
        
        ## The sample
        Qhat_sa = reward + gamma * self.getValue(nextState)
        ## Update the old Q-value with Q-hat from the sample
        update = (1-alpha)*self.getQValue(state, action) + alpha*Qhat_sa
        ## Store the new Q(s,a) in the Q-value dictionary (Counter object)
        self.dict_Q_sa[(state, action)] = update


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)
    

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
