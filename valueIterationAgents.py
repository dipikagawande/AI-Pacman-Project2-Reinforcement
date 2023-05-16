# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        '''
        This method will calculate updated values for state s at each 
        iteration. Run value iteration for the number of iters specified in 
        self.iterations. Use self.values to store Q-values for (s, a) in each 
        iteration. self.values is a Counter object (a dictionary).

        As we learned in Problem Set 2, terminal states have value = 0. 

        To find the new V(s) at every non-terminal state, this method calls 
        the method getQValue, defined later in this code. So we don't need to 
        specify the nitty gritty of Q-value calculation in this method.
        
        We use the formula: V_{i+1}(s) = MAX_{a} Q_i(s,a)
        '''
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        ## Run this value iteration algo for as many iters as specified:
        for i in range(self.iterations):
            ## Copy Vi(s) and call it Vi+1(s); we only have to update  
            ## the Values that actually change in each iteration 
            dict_Viplus1 = self.values.copy()
            ## For each state s in the MDP:
            for s in self.mdp.getStates():
                ## Initiate empty list of Q(s,a) values at this state
                list_Qi_sa = []
                ## If s is not a Terminal state, calculate its V_i+1(s)
                if not self.mdp.isTerminal(s):
                    ## For each possible action at this state s:
                    for a in self.mdp.getPossibleActions(s):
                        ## Calculate Qi(s,a) and add to Q-val list
                        list_Qi_sa.append(self.getQValue(s,a))
                    ## The new V_i+1(s) at this iteration is the max Qi(s,a)
                    dict_Viplus1[s] = max(list_Qi_sa) 
                ## Else if s is a Terminal state):
                else:
                    ## The new V_i+1(s) is zero
                    dict_Viplus1[s] = 0

            ## Update Vi(s) to Vi+1(s) and start next iteration
            self.values = dict_Viplus1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          
          Use the formula: 
          Q_i(s,a) = SUM_{s'} T(s,a,s') [R(s,a) + gamma V_i(s')],
          where s is the current state and s' is a successor state.
        """
        "*** YOUR CODE HERE ***"
        ## Initialize
        RHSi = []
        gamma = self.discount

        ## Method getTransitionStatesAndProbs returns list of (nextState, prob) 
        ## pairs with all states reachable from 'state' by taking 'action', and 
        ## their transition probability.
        for sPrimePair in self.mdp.getTransitionStatesAndProbs(state, action):
            ## Store reward R(s,a)
            R_sa = self.mdp.getReward(state, action, sPrimePair[0])
            ## Store Vi(s')
            Vi_sPrime = self.values[sPrimePair[0]]
            ## Append right-hand-side of the equation to the RHS list
            RHSi.append(sPrimePair[1] * (R_sa + (gamma * Vi_sPrime)))
        ## We want the sum of the RHS
        return sum(RHSi)
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          We use the formula: pi(s) = ARGMAX_{a} Q(s,a)
        """
        "*** YOUR CODE HERE ***"
        ## If current state s is not Terminal:
        if not self.mdp.isTerminal(state):
            ## Initialize a dict of actions and their Q(s,a) at this state
            dict_actionsQs = {}
            ## For each possible action a at this state s:
            for a in self.mdp.getPossibleActions(state):
                ## Calculate Q(s,a)...
                Q_sa = self.computeQValueFromValues(state, a)
                ## ...and store it in the dictionary
                dict_actionsQs[a] = Q_sa
            ## pi(s) is the action (dict key) of the max Q(s,a) value
            return max(dict_actionsQs, key = dict_actionsQs.get)
        ## Else (current state is Terminal, then no actions are possible)
        else:
            return None
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
