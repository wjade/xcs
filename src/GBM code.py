#build GBM model and manual tuning functions
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBClassifier

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

#parameter tuning using Q-learning
#consider the model to have 64 states, each state is a combination of the parameters.

num_actions = 2**num_parameter #each parameter has the option of increase 10% or decrease 10%(or increase 1 or descrease 1)

##num_states = ??? how to define num of states?

#parameters that stricten regulation if increased
#gamma
#min_child_weight

#parameters that ease regulation if decreased
#colsample_by_tree = col_sample
#subsample = row_sample
#learning_rate = eta
#max_depth = d


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
				
		best_action = np.argmax(Q[state])
		Action_probabilities[best_action] += (1.0 - epsilon)
		return Action_probabilities

	return policyFunction


#define boundaries for the parameters, need to include in the code below that once boundary is reached, the MDP terminates 
min_col_sample = 0.5
max_col_sample = 1
min_row_sample = 0.5
max_row_sample = 1
min_g = 0
max_g = 15
min_eta = 0.1
max_eta = 0.5
min_d = 3
max_d = 7
min_c = 1
max_c = 3



def qLearning(env, num_episodes, discount_factor = 1.0, alpha = 0.6, epsilon = 0.1):
	"""
	Q-Learning algorithm: Off-policy TD control.
	Finds the optimal greedy policy while improving
	following an epsilon-greedy policy"""
	
	# Action value function
	# A nested dictionary that maps
	# state -> (action -> action-value).
	Q = defaultdict(lambda: np.zeros(env.action_space.n))

	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths = np.zeros(num_episodes),
		episode_rewards = np.zeros(num_episodes))	
	
	# Create an epsilon greedy policy function
	# appropriately for environment action space
	policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
	
	# For every episode
	for ith_episode in range(num_episodes):
		
		# Reset the environment and pick the first action
		state = env.reset()
		
		for t in itertools.count():
			
			# get probabilities of all actions from current state
			action_probabilities = policy(state)

			# choose action according to
			# the probability distribution
			action = np.random.choice(np.arange(
					len(action_probabilities)),
					p = action_probabilities)

			# take action and get reward, transit to next state
			next_state, reward, done, _ = env.step(action)

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
