#build GBM model and manual tuning functions
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
#exec(open("/Users/yejiang/Desktop/Stanford ML class/project/code/data cleaning.py").read())

# split data into X and y
X_train = Train.drop(['Claim_Number', 'Target'], axis=1)
Y_train = Train['Target']

X_dev = Dev.drop(['Claim_Number', 'Target'], axis=1)
Y_dev = Dev['Target']

X_test = Test.drop(['Claim_Number', 'Target'], axis=1)
Y_test = Test['Target']

#using default parameters as baseline model
model_baseline = XGBClassifier(base_score=0.5

                              , colsample_by_tree = 1
                              , subsample = 0.9
                              , gamma = 0
                              , learning_rate = 0.3
                              , max_depth = 6
                              , min_child_weight = 1
                              
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

model_baseline.fit(X_train, Y_train)

train_pred = model_baseline.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_pred)#0.9649
print(train_accuracy)

dev_pred = model_baseline.predict(X_dev)
dev_accuracy = accuracy_score(Y_dev, dev_pred)#0.9578
print(dev_accuracy)

test_pred = model_baseline.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_pred)#0.9584
print(test_accuracy)


#grid search for parameter tuning

grid_search = []

i = 0

for col_sample in (0.7, 1):#baseline is 1
    for row_sample in (0.5, 0.9): #baseline is 0.9
        for g in (0, 5): #baseline is 0
            for eta in (0.05, 0.3): #baseline is 0.3
                for d in (4, 6): #baseline is 6
                    for c in (1, 5): #baseline is 1

                        i = i + 1
                        
                        model_grid = XGBClassifier(base_score=0.5

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


                        model_grid.fit(X_train, Y_train)

                        train_pred = model_grid.predict(X_train)
                        train_accuracy = accuracy_score(Y_train, train_pred)

                        dev_pred = model_grid.predict(X_dev)
                        dev_accuracy = accuracy_score(Y_dev, dev_pred)

                        grid_search.append([col_sample, row_sample, g, eta, d, c, train_accuracy, dev_accuracy])

                        print(i)


grid_search = np.array(grid_search)
print(grid_search)

#returns grid_search with the max
max_value = grid_search[grid_search[:,7]==grid_search[:,7].max()]

#max value1
#[0.7, 0.9, 5, 0.3, 6, 5, 0.9622, 0.9585]
#[1,    0.9, 5, 0.3, 6, 5, 0.9622, 0.9585]

#grid search's result is slightly better than baseline, imporved from 0.9578 to 0.9585
#comparing the parameters, the baseline model appears to be overfitting(parameters are telling the same story)





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

#create dictionary for state space

def simulate(state_tuple):

    """purpose of this function: for a given state, find the best action, calculate the reward of the given state and action combination, and output the outcome state as a tuple """
    
    col_sample, row_sample, g, eta, d, c = state_tuple

    policy = createEpsilonGreedyPolicy(Q, epsilon, num_actions)

    # get probabilities of all actions from current state
    action_probabilities = policy(state)

    # choose action according to the probability distribution
    action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)

    if action == 1:
        col_sample = max((col_sample - 0.1), min_col_sample)
        elif action == 2:
           col_sample = min((col_sample + 0.1), max_col_sample)
          elif action == 3:
             row_sample = max((row_cample - 0.1), min_col_sample)
             elif action == 4:
                row_sample = min((row_sample + 0.1), max_row_sample)
               elif action == 5:
                    g = max((g-5), min_g)
                   elif action == 6 :
                       g = min((g + 5), max_g)
                           elif action == 7:
                             eta = max((eta - 0.1), min_eta)
                              elif action == 8:
                                 eta = min((eta + 0.1), max_eta)
                                 elif action == 9 :
                                     d = max((d-1), min_d)
                                       elif action == 10:
                                          d = min((d + 1), max_d)
                                         elif action == 11:
                                              c = max((c-1), min_c)
                                              elif action == 12:
                                                   c = min((c + 1), max_c)

    updated_state = (col_sample, row_sample, g, eta, d, c)

    return updated_state

			
def qLearning(state_tuple, num_episodes, discount_factor = 0.9, alpha = 0.6, epsilon = 0.1):

                    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
                   
                    # Keeps track of useful statistics
                    stats = EpisodeStats(episode_lengths = np.zeros(num_episodes), episode_rewards = np.zeros(num_episodes))

                    # Action value function
                    # A nested dictionary that maps
                    # state -> (action -> action-value).
                    Q = collections.defaultdict(lambda: np.zeros(num_actions))

	# For every episode -- eposide is all states that come in between an initial state and a terminal state
	for ith_episode in range(num_episodes):

                                        state_tuple = #initialize

		for t in itertools.count():

            #run earlier functions to find actions to take and updated state
            updated_state = simulate(state_tuple)
            
            col_sample, row_sample, g, eta, d, c = updated_state

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
			#next_state = [col_sample, row_sample, g, eta, d, c]

			# Update statistics
			stats.episode_rewards[ith_episode] += reward
			stats.episode_lengths[ith_episode] = t
			
			# TD Update
			best_next_action = np.argmax(Q[next_state])	
			td_target = reward + discount_factor * Q[next_state][best_next_action]
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta

			# done is True if episode terminated
			if done:
				break
				
			state = next_state
	
	return Q, stats


Q, stats = qLearning(env, 1000)
