import gym
import gym_breastcancer
import numpy as np
import glob
import shutil
import train_agents
import pandas as pd
from scipy.special import softmax
from ray.tune.registry import register_env
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


env = gym.make('breastcancer-v1')
register_env("myEnv", lambda config: env)



def create_dataset(start, stop, agent_type):


    dataset = {'action' : [], 'all_action_probabilities' : [], 
           'episode_id' : [], 'episode_name' : [], 'reward' : [],
           'transition_number' : [],'features' : []}
    

    
    for episode in range(start,stop):
        state = env.reset()
        done = False
        transition_number = 0
        dataset['features'].append(state)
        while not done:
            dataset['transition_number'].append(transition_number)
            transition_number += 1
            action, _, extra = agent_type.compute_single_action(env.state, full_fetch = True)
            dataset['action'].append(action)
            n_state, reward, done, info = env.step(action)
            dataset['features'].append(n_state)
            dataset['episode_id'].append(episode)
            dataset['episode_name'].append('cancer')
            dataset['reward'].append(reward)
            # probs = [.16]*6
            # dataset['all_action_probabilities'].append(probs)
            probs = softmax(extra['q_values'])
            dataset['all_action_probabilities'].append(list(probs))
            
            if done:
                dataset['action'].append(None)
                dataset['all_action_probabilities'].append(None)
                dataset['reward'].append(None)
                dataset['transition_number'].append(None)
                dataset['episode_id'].append(episode)
                dataset['episode_name'].append('cancer')

    data = pd.DataFrame.from_dict(dataset)
    #data = data.replace('NaN', np.nan)
    data['action'] = data['action'].astype('Int64')
    data['episode_id'] = data['episode_id'].astype('Int64')
    data['reward'] = data['reward'].astype('Int64')
    data['transition_number'] = data['transition_number'].astype('Int64')

    cols = pd.DataFrame(data['features'].to_list(), columns=['state_feature_0','state_feature_1','state_feature_2',
                                                                'state_feature_3','state_feature_4'])
    
    data.drop(['features'], axis = 1, inplace=True)

    data_ = pd.concat([data,cols], axis=1, join='inner')
    
    return data_ , data_.shape


# def create_mixed_data(data1,data2, ratio):
#     dataset = pd.concat([data1,data2], axis = 0)
#     dataset.to_csv(f'offline_data/Augmented Data/{ratio}.csv', index = False)
#     return dataset


def checkpoint_directory(timesteps):

    links = glob.glob(f'offline_data/Agents/{timesteps}/[offline_data]*/[checkpoint]*/checkpoint-{timesteps}')
    x = str(timesteps)
    filter = [i for i in links if i.endswith(x)]

    return filter[0]

def augmented_data(timestep):

    df = pd.read_csv('offline_data/Exploration Data/exploration.csv', index_col= None)
    agent_generated = pd.read_csv(f'offline_data/Agent Generated/offline_data_{timestep}.csv', index_col=None)


    for i in range(10,100,10):
        ids = df.episode_id.unique()
        ids_selected = np.random.choice(ids, size = int(len(ids)*i/100), replace = False)
        df_subset = df.groupby('episode_id').filter(lambda x: x['episode_id'].iloc[0] in ids_selected)
        combined = pd.concat([df_subset, agent_generated], axis = 0)
        combined.to_csv(f'offline_data/Augmented Data {timestep}/buffer_{i}_agent.csv', index = False)



if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--timestep", default=5, type=int, help="Timestep of the Agent(5/10/15/20)")
    args = vars(parser.parse_args())


    agent = train_agents.restore(args["timestep"])
    path = checkpoint_directory(args["timestep"])
    agent.restore(checkpoint_path= path)

    data, _ = create_dataset(0, 1000, agent)
    data.to_csv(f'offline_data/Agent Generated/offline_data_{args["timestep"]}.csv', index = False)

    augmented_data(args["timestep"])
    shutil.rmtree(agent.logdir) 

