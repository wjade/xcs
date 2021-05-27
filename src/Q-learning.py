import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBClassifier
import math
import collections
from collections import namedtuple

#executing data cleaning code
exec(open("/Users/yejiang/Desktop/Stanford ML class/project/code/data cleaning.py").read())

###parameter tuning using Q-learning

#define boundaries for the parameters, need to include in the code below that once boundary is reached, the MDP terminates 
min_col_sample = 0.5
max_col_sample = 1
min_row_sample = 0.5
max_row_sample = 1
min_g = 0
max_g = 10
min_eta = 0.1
max_eta = 0.5
min_d = 3
max_d = 7
min_c = 1
max_c = 3

num_parameter = 6

num_actions = 2*num_parameter #each parameter has the option of increase or decrease 0.1(or 1 or 5)

num_states = (len(np.arange(0.5, 1.1, 0.1)) #number of options for col_sample
* len(np.arange(0.5, 1.1, 0.1)) #number of options for row_sample
* len((0, 5, 10)) #number of options for gamma
* len(np.arange(0.1, 0.6, 0.1)) #number of options for eta
* len(np.arange(3,8,1)) #number of options for max_depth
* len(np.arange(1,4,1))) #number of options for min_childweighht

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
	"""
	Creates an epsilon-greedy policy based
	on a given Q-function and epsilon.
	
	Returns a function that takes the state
	as an input and returns the probabilities
	for each action in the form of a numpy array
	of length of the action space(set of possible actions).
	"""
	def policyFunction(state):

                                        #each action has the same probability of epsilon/num_actions
		Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / num_actions
				
		best_action = np.argmax(Q[state]) #state is the key of the dictionary
		Action_probabilities[best_action] += (1.0 - epsilon)
		return Action_probabilities

	return policyFunction


initial_state = (1, 0.9, 0, 0.3, 6, 1)
		
def qLearning(initial_state, num_episodes, discount_factor, alpha, epsilon):

            EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

            # Keeps track of useful statistics
            stats = EpisodeStats(episode_lengths = np.zeros(num_episodes), episode_rewards = np.zeros(num_episodes))

            # Action value function
            # A nested dictionary that maps
            # state -> (action -> action-value).
            Q = collections.defaultdict(lambda: np.zeros(num_actions))

            policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)

            # For every episode -- eposide is all states that come in between an initial state and a terminal state
            for ith_episode in range(num_episodes):

                if ith_episode > 0:
                    state_tuple = next_state
                else:
                    state_tuple = initial_state

                #for t in itertools.count():
                for t in range(0, 100):

                    print(t)

                    col_sample, row_sample, g, eta, d, c = state_tuple

                    # get probabilities of all actions from current state
                    action_probabilities = policy(state_tuple)

                    # choose action according to the probability distribution
                    action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)

                    if action == 1:
                        col_sample = (col_sample - 0.1)
                    elif action == 2:
                        col_sample = (col_sample + 0.1)
                    elif action == 3:
                        row_sample = (row_sample - 0.1)
                    elif action == 4:
                        row_sample = (row_sample + 0.1)
                    elif action == 5:
                        g = (g-5)
                    elif action == 6 :
                        g = (g + 5)
                    elif action == 7:
                        eta = (eta - 0.1)
                    elif action == 8:
                        eta = (eta + 0.1)
                    elif action == 9 :
                        d = (d-1)
                    elif action == 10:
                        d = (d + 1)
                    elif action == 11:
                        c = (c-1)
                    elif action == 12:
                        c = (c + 1)

                    if (col_sample > max_col_sample or col_sample < min_col_sample
                            or row_sample > max_row_sample or row_sample < min_row_sample
                            or g > max_g or g < min_g
                            or eta > max_eta or eta < min_eta
                            or d > max_d or d < min_d
                            or c > max_c or c < min_c):

                        break

                    else:
                            
                            # take action
                            model_RL = XGBClassifier(base_score=0.5
                                                                , colsample_by_tree = col_sample
                                                                , subsample = row_sample
                                                                , gamma = g
                                                                , learning_rate = eta
                                                                , max_depth = d
                                                                , min_child_weight = c
                                                                , missing=None
                                                                , n_estimators=100
                                                                , nthread=-1
                                                                , max_delta_step=0
                                                                , objective='binary:logistic'
                                                                , reg_alpha=0
                                                                , reg_lambda=1
                                                                , scale_pos_weight=1
                                                                , seed=42
                                                                , silent=True)

                    
                            model_RL.fit(X_train, Y_train)
            
                            dev_pred = model_RL.predict(X_dev)
                   
                            dev_accuracy = accuracy_score(Y_dev, dev_pred)

                            #calculate reward corresponding to the action
                            reward = dev_accuracy - 0.9578 #dev accuracy of tuned model - dev accuracy of baseline model

                            #experiment new_reward - old_reward

                            #record new state
                            next_state = (col_sample, row_sample, g, eta, d, c)

                            # Update statistics
                            stats.episode_rewards[ith_episode] += reward
                            stats.episode_lengths[ith_episode] = t
                    
                            # TD Update
                            best_next_action = np.argmax(Q[next_state])	
                            td_target = reward + discount_factor * Q[next_state][best_next_action]
                    
                            td_delta = td_target - Q[state_tuple][action]
                            Q[state_tuple][action] += alpha * td_delta

                            state_tuple = next_state
                            
                return Q, stats

#Q, stats = qLearning((1, 0.9, 0, 0.3, 6, 1), 10, discount_factor = 0.9, alpha = 0.6, epsilon = 0.1)
Q, stats = qLearning((1, 0.9, 0, 0.3, 6, 1), 10, discount_factor = 0.95, alpha = 0.9, epsilon = 0.2)
