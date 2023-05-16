# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    """
    Want agent to prefer the close exit and the cliff, so prefers shorter 
    path. We can do this by strongly discounting future rewards, and putting
    a high negative reward for being alive so it doesn't prefer long routes. 
    We need to make these strong, but not so strong that it jumps off the 
    cliff instead of surviving 3 steps to the close exit. Since we are being
    risky and staying near the cliff, want zero or little noise (so agent 
    doesn't fall off cliff instead of reaching close exit.)
    """
    answerDiscount = 0.6
    answerNoise = 0.1
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
    Want agent to prefer the close exit but NOT the cliff, so we want to still
    discount the faraway rewards but keep a slightly higher reward for staying
    alive for longer steps. This way the agent will prefer a longer route
    avoiding the cliff, but not have so much stamina that it prioritizes
    the faraway exit. We can have a little more noise to allow the agent
    to explore a roundabout path away from the cliff.
    """
    answerDiscount = 0.6
    answerNoise = 0.3
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
    Now we want agent to prioritize the faraway exit but prefer the shortest
    route to get there, risking the cliff. So we will have almost no discount
    for future rewards, but still keep a high penalty for being alive (so it
    prefers the shortest route to the faraway exit). Like in 3a, since we are 
    being risky near the cliff, want zero or little noise (so agent doesn't 
    fall off cliff en route to the farway exit.)
    """
    answerDiscount = 0.99
    answerNoise = 0.0
    answerLivingReward = -4
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
    Now we want agent to prioritize the faraway exit and take the long route
    away from the cliff. So we will have almost no discount for future rewards, 
    and also we will make it free to stay alive for as many steps as possible
    (no negative reward for surviving at every state). We can add noise to 
    allow the agent to explore a roundabout path away from the cliff.
    """
    answerDiscount = 0.99
    answerNoise = 0.4
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
    An episode should never terminate. We also want to avoid the cliff, which 
    means we need to prioritize longer routes towards the top of the graph. 
    Therefore cannot have any penalty for staying alive. We should discount all
    future rewards, since we don't want to take either the close or far exit.
    We can add a LOT of noise to allow the agent to explore a roundabout and
    fruitless path away from the cliff.
    """
    answerDiscount = 0
    answerNoise = 0.8
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

    
def question6():
    answerEpsilon = None
    answerLearningRate = None
    #return answerEpsilon, answerLearningRatE
    # If not possible, return 'NOT POSSIBLE'
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
