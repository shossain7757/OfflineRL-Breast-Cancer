import os
import sys
import tensorflow as tf
module_path = os.path.abspath(os.path.join('..'))
resources_path = os.path.abspath(os.path.join('Resources'))
if module_path not in sys.path:
    sys.path.append(module_path)
if resources_path not in sys.path:
    sys.path.append(resources_path)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter




import pandas as pd
from copy import deepcopy
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters, NNImitationModelParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment, GymEnvironment
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters
from rl_coach.base_parameters import TaskParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace, VectorObservationSpace, StateSpace, RewardSpace

################
#  Environment #
################
env_params = GymVectorEnvironment(level='gym_breastcancer.envs:BreastCancerDCIS')
env = GymEnvironment(**env_params.__dict__,visualization_parameters = VisualizationParameters())
task_parameters = TaskParameters()


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
# 100 epochs (we run train over all the dataset, every epoch) of training
schedule_params.improve_steps = TrainingSteps(100)
# we evaluate the model every epoch
schedule_params.steps_between_evaluation_periods = TrainingSteps(1)



#########
# Agent #
#########
# note that we have moved to BCQ, which will help the training to converge better and faster
agent_params = DDQNBCQAgentParameters() 
agent_params.network_wrappers['main'].batch_size = 32
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.action_drop_method_parameters = NNImitationModelParameters()



# NN configuration
agent_params.network_wrappers['main'].learning_rate = 0.0001
agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
agent_params.network_wrappers['main'].l2_regularization = 0.0001
agent_params.network_wrappers['main'].softmax_temperature = 0.2
# agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(1024), Dense(1024), Dense(512)]
# agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(24), Dense(24)]


# reward model params
agent_params.network_wrappers['reward_model'] = deepcopy(agent_params.network_wrappers['main'])
agent_params.network_wrappers['reward_model'].learning_rate = 0.0001
agent_params.network_wrappers['reward_model'].l2_regularization = 0

# imitation model params
agent_params.network_wrappers['imitation_model'] = deepcopy(agent_params.network_wrappers['main'])
agent_params.network_wrappers['imitation_model'].learning_rate = 0.0001
agent_params.network_wrappers['imitation_model'].l2_regularization = 0


# E-Greedy schedule - there is no exploration in Batch RL. Disabling E-Greedy. 
agent_params.exploration.epsilon_schedule = LinearSchedule(initial_value=0, final_value=0, decay_steps=1)
agent_params.exploration.evaluation_epsilon = 0
agent_params.memory = EpisodicExperienceReplayParameters()

spaces = SpacesDefinition(state=StateSpace({'observation': VectorObservationSpace(shape=5)}),
                          goal=None,
                          action=DiscreteActionSpace(6),
                          reward=RewardSpace(1))


graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                    env_params=None,
                                    spaces_definition=spaces,
                                    schedule_params=schedule_params,
                                    vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1,
                                                                       dump_parameters_documentation=False),
                                    reward_model_num_epochs=40,
                                    train_to_eval_ratio=0.4)



def get_cumulative_reward(x):
    test_reward = []
    for i in range(x):
        success = []
        for _ in range(0,100):
            obs = env.reset_internal_state()
            obs.game_over = False
            rewards = []
            actions = []
            while not obs.game_over:
                action_info = graph_manager.get_agent().choose_action(obs.next_state)
                obs = env.step(action_info.action)
                rewards.append(obs.reward)
                actions.append(action_info.action)
            success.append((rewards,actions))
        result = pd.DataFrame(success, columns= ['Rewards', 'Actions'])
        result['sum_reward'] = result['Rewards'].apply(lambda x : sum(x))
        result['cumsum_reward'] = result['sum_reward'].cumsum()
        test_reward.append(result)
    test_reward = pd.concat(test_reward, axis= 1, ignore_index= True)
    

    return test_reward
    
    
    




if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-q", "--question", default=1, type=str, help="Research question to be evaluated")
    parser.add_argument("-t", "--timestep", default=5, type=str, help="Training timestep of the agent to be evaluated")
    parser.add_argument("-e", "--exploration", default=None, type=str, help="Exploration factor in the augmented dataset")
    args = vars(parser.parse_args())
    
    tf.reset_default_graph()

    if args['question'] == '1':
        task_parameters.checkpoint_restore_path = f"./offline_result/RQ{args['question']}/{args['timestep']}/checkpoint"
        graph_manager.create_graph(task_parameters)
        rewards = get_cumulative_reward(5)
        rewards.to_csv(f'offline_result/RQ{args["question"]}/{args["timestep"]}/test_reward_{args["timestep"]}.csv', index = False)

    elif args['question'] == '2' and args['exploration'] != None:
        task_parameters.checkpoint_restore_path = f"./offline_result/RQ{args['question']}/buffer_{args['exploration']}/checkpoint"
        graph_manager.create_graph(task_parameters)
        rewards = get_cumulative_reward(5)
        rewards.to_csv(f'./offline_result/RQ{args["question"]}/buffer_{args["exploration"]}/test_reward_{args["exploration"]}.csv', index=False)

    elif args['question'] == '3' and args['exploration'] != None:
        task_parameters.checkpoint_restore_path = f"./offline_result/RQ{args['question']}/{args['timestep']}/buffer_{args['exploration']}/checkpoint"
        graph_manager.create_graph(task_parameters)
        rewards = get_cumulative_reward(5)
        rewards.to_csv(f'./offline_result/RQ{args["question"]}/{args["timestep"]}/buffer_{args["exploration"]}/test_reward_{args["exploration"]}.csv', index=False)

    
    


