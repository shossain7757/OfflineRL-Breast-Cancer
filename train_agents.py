import gym
import gym_breastcancer
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
import os
from datetime import datetime
import tempfile
from ray.tune.logger import UnifiedLogger
import pandas as pd
import glob
import numpy as np
from scipy.special import softmax
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter




def train_ray(timesteps, offline, should_train, parallel):

    ## Instantitiate The Environment


    env = gym.make('breastcancer-v1')
    register_env("myEnv", lambda config: env)



    ## Setting the configuration of the network
    
    config = dqn.DEFAULT_CONFIG.copy()
    config["double_q"] = False
    config["dueling"] = False
    

    if parallel:
        config["num_workers"] = 4
        config["num_gpus"] = 1
        config["num_gpus_per_worker"] = 0.25
        config['tf_session_args']['device_count']['GPU'] = 1

    if offline:
        config["output"] = f'offline_data/Agents/{timesteps}'
        config["output_max_file_size"] = 1024*1024*1024
        config["output_compress_columns"] = []
        config["batch_mode"] = "complete_episodes"


    ## A custom logger function to log the training results in the 
    ## desired directory

    def custom_log_creator(custom_path, custom_str):

        timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        logdir_prefix = "{}_{}".format(custom_str, timestr)

        def logger_creator(config):

            if not os.path.exists(custom_path):
                os.makedirs(custom_path)
            logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
            return UnifiedLogger(config, logdir, loggers=None)

        return logger_creator


    agent = dqn.DQNTrainer(config=config, 
                           env="myEnv", 
                           logger_creator=
                           custom_log_creator(os.path.expanduser(f'./offline_data/Agents/{timesteps}'), 'offline_data'))


    if should_train:
        for j in range(timesteps):
            agent.train()

        agent.save()


    return agent


    

    



# Convert JSON into readable CSV for COACH
def convert_JSON_to_CSV(timesteps):

    file = glob.glob(f'offline_data/{timesteps}/[output]*.json')
    dfs = [pd.read_json(fp,lines=True) for fp in file]
    df = pd.concat(dfs, axis=0)


    df.drop(['type','prev_actions','prev_rewards',
                'dones','unroll_id','agent_index', 'q_values',
                'action_logp','action_prob', 'weights'], axis = 1, inplace = True)

    df = df.explode(['obs', 'new_obs', 'actions', 'rewards', 'eps_id', 't', 'action_dist_inputs'])

    df = df.groupby('eps_id').apply(lambda x: x.append({'eps_id': x.name, 'obs' : x['new_obs'].iat[-1], 't' : x['t'].iat[-1]+1}, 
                                                    ignore_index=True)).astype({'eps_id': int,  't': int}).reset_index(drop=True)


    df.drop(['new_obs'], axis = 1, inplace=True)


    df.rename(columns = {'actions' : 'action',
                                'obs' : 'state_feature_0',
                                'rewards' : 'reward',
                                'eps_id' : 'episode_id',
                                't' : 'transition_number',
                                'action_dist_inputs' : 'all_action_probabilities'}, inplace = True)


    cols = pd.DataFrame(df['state_feature_0'].to_list(), columns=['state_feature_0','state_feature_1','state_feature_2',
                                                                'state_feature_3','state_feature_4'])


    df.drop(['state_feature_0'], axis = 1, inplace=True)

    df = pd.concat([df, cols], axis=1, join='inner')

    df['all_action_probabilities'] = df['all_action_probabilities'].apply(softmax)
    df['episode_name'] = 'cancer'

    df = df[['action','all_action_probabilities','episode_id',
                            'episode_name','reward','transition_number',
                            'state_feature_0','state_feature_1','state_feature_2','state_feature_3', 'state_feature_4']]

    all_action_probabilities = []    
    for i in df['all_action_probabilities']:
        x = i.tolist()
        all_action_probabilities.append(x)
        
    df['all_action_probabilities'] = all_action_probabilities

    df.to_csv(f'offline_data/Agents/{timesteps}/offline_data_{timesteps}.csv', index = False)



def restore(timesteps):

    agent  = train_ray(timesteps, offline= False, should_train= False, parallel = False)

    return agent




if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--timestep", default=5, type=int, help="Training timesteps (in thousands)")
    parser.add_argument("-d", "--store_data", default=True, type=str, help="Store Replay Buffer (True/False)")
    parser.add_argument("-gpu", "--use_gpu", default=False, type = str, help="Use GPU (True/False)")
    args = vars(parser.parse_args())

    #timesteps = 5

    ray.init(ignore_reinit_error=True, log_to_driver=False) 

    trained_agent = train_ray(args["timestep"], offline= args["store_data"], should_train= True, parallel = args["use_gpu"])

    if args["store_data"] == True:
        convert_JSON_to_CSV(args["timestep"])

    ray.shutdown()