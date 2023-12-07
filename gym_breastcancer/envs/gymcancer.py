import numpy as np
import random
import gym
import pandas as pd
import glob

class BreastCancerDCIS(gym.Env):
    
    def __init__(self):
        
        # Six features
        tumor_site = ['Local','Spread']
        tumor_margin = ['Negative','Positive','Close', 'Null']
        tumor_classification = ['Non-Invasive','Invasive']
        age = ['<50','>=50']
        breast_mass = ['Yes','No']
    


        self.observation_space = gym.spaces.Box(low = np.array([0,0,0,0,0]), high = np.array([1,3,1,1,1]), dtype = np.int64)

        # 5 discrete actions. BCS, MAS, APBI, WBRT, Endocrine Therapy

        self.action_names = ['BCS', 'MAS', 'APBI', 'WBRT', 'Endocrine Therapy', 'Reconstruction']

        self.action_space = gym.spaces.Discrete(6)

        # Generate state space with patient having cancer

        sample = self.observation_space.sample() # Generate sample state space

        feature_index = [1,4] # Choose the index of the list where patients should have initial value larger than zero

        # If the sample generated a healthy patient it will resample. 
        while [sample[i] for i in feature_index] != [3,0]:
            sample = self.observation_space.sample()
        
        self.state = sample 

        # Set treatment length

        self.treatment_length = 10

        # Render Text

        self.feature_names =[tumor_site, tumor_margin, tumor_classification, age, breast_mass]
        self.log = ''
        
        
        # Read Transition Matrix
        files = sorted(glob.glob('transition_matrix/*.csv'))
        self.transition_matrix = {}
        for i,j in zip(files,range(len(files))):
            self.transition_matrix[j] = pd.read_csv(i, index_col= 0)
        
        
        
        
    def step(self, action):

        

        self.feature_space_text = [j[i] for i,j in zip(self.state,self.feature_names)]

        # Log the state

        self.log += f'state: {self.feature_space_text}\n'

        # Log the action

        self.log += f'action: {self.action_names[action]}\n'

        # Parse the transition matrix and find new state
        
        probs = self.transition_matrix[action].loc[str(tuple(self.feature_space_text))]

        transition_state_str = np.random.choice(probs.index, 1 , p = probs.values, replace = True)

        trstate_strp = transition_state_str[0].split("'") # Modify the previous line 

        transition_state_text = [i for i in trstate_strp if i not in ['(', ')', ', ']]# Remove the unwanted strings from above line
 
        self.state = np.array([j.index(i) for i,j in zip(transition_state_text,self.feature_names)]) # New state in numpy array

        self.treatment_length -= 1 # Reduce episode length
        

        # Log the transition state

        self.log += f'new_state: {transition_state_text}\n'

        # Done condition

        if self.treatment_length == 0:
            done = True
        else:
            done = False


        # Reward  condition


        if set([self.state[i] for i in [0,1,2,4]]) == {0}:
            reward = 10
            done = True
        elif self.feature_space_text == transition_state_text:
            reward = -10
        else:
            reward = -1


        

        self.log += f'Reward: {reward}\n'

        self.log += f'Done: {done}\n'

        info = {}

        return self.state, reward, done, info

        


    
    def render(self, mode=None):
        print(self.log)
        self.log = ''
    
    def reset(self):
        # Generate state space with patient having cancer

        sample = self.observation_space.sample() # Generate sample state space

        feature_index = [1,4] # Choose the index of the list where patients should have initial value larger than zero

        # If the sample generated a healthy patient it will resample. 
        while [sample[i] for i in feature_index][0:4] != [3,0]:
            sample = self.observation_space.sample()
        
        self.state = sample

        self.treatment_length = 10

        return self.state
    


