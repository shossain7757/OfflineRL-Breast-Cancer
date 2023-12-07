import os
import sys
import tensorflow as tf
import glob
module_path = os.path.abspath(os.path.join('..'))
resources_path = os.path.abspath(os.path.join('Resources'))
if module_path not in sys.path:
    sys.path.append(module_path)
if resources_path not in sys.path:
    sys.path.append(resources_path)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



from copy import deepcopy
from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters, NNImitationModelParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, CsvDataset
from rl_coach.environments.gym_environment import GymVectorEnvironment, GymEnvironment
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters
from rl_coach.base_parameters import TaskParameters
from rl_coach.spaces import SpacesDefinition, DiscreteActionSpace, VectorObservationSpace, StateSpace, RewardSpace



def BCQ(DATASET_PATH):
    ####################
    # Graph Scheduling #
    ####################

    schedule_params = ScheduleParameters()
    schedule_params.improve_steps = TrainingSteps(100)
    schedule_params.steps_between_evaluation_periods = TrainingSteps(1)



    #########
    # Agent #
    #########
    agent_params = DDQNBCQAgentParameters() 
    agent_params.network_wrappers['main'].batch_size = 32
    agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(100)
    agent_params.algorithm.discount = 0.99
    agent_params.algorithm.action_drop_method_parameters = NNImitationModelParameters()



    # NN configuration>
    agent_params.network_wrappers['main'].learning_rate = 0.0001
    agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
    agent_params.network_wrappers['main'].l2_regularization = 0.0001
    agent_params.network_wrappers['main'].softmax_temperature = 0.2


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

    # Read offline data
    agent_params.memory = EpisodicExperienceReplayParameters()
    agent_params.memory.load_memory_from_file_path = CsvDataset(DATASET_PATH, is_episodic = True)

    spaces = SpacesDefinition(state=StateSpace({'observation': VectorObservationSpace(shape=5)}),
                            goal=None,
                            action=DiscreteActionSpace(6),
                            reward=RewardSpace(1))


    graph_manager = BatchRLGraphManager(agent_params=agent_params,
                                        env_params=None,
                                        spaces_definition=spaces,
                                        schedule_params=schedule_params,
                                        vis_params=VisualizationParameters(dump_signals_to_csv_every_x_episodes=1,
                                                                        dump_parameters_documentation=False,
                                                                        print_networks_summary= True),
                                        reward_model_num_epochs=40,
                                        train_to_eval_ratio=0.4)
    
    return graph_manager


def delete_checkpoints():

    path = task_parameters.checkpoint_save_dir
    all_files = glob.glob(path+'/[0-9]*')
    checkpoint_files = glob.glob(path+'/[9][9]*')
    delete = [i for i in all_files if i not in checkpoint_files]
    for i in delete:
        os.remove(i)



if __name__ == '__main__':

    # Parse Command Line Arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-q", "--question", default=1, type=str, help="Experiment of the question")
    parser.add_argument("-t", "--agent_timestep", default=5, type =str, help="Training timestep of the agent")
    parser.add_argument("-e", "--exploration", default = None, type=str, help = "Exploration factor in augmented dataset")
    args = vars(parser.parse_args())
    if args["question"] == '1':
        DATASET_PATH = f"./offline_data/Agent Generated/offline_data_{args['agent_timestep']}.csv"
        task_parameters = TaskParameters(experiment_path = f'./offline_result/RQ{args["question"]}/{args["agent_timestep"]}')
        graph_manager = BCQ(DATASET_PATH)
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()
        delete_checkpoints()

    elif args['question'] == '2' and args["exploration"] != None:
        DATASET_PATH = f"./offline_data/Augmented Data {args['agent_timestep']}/buffer_{args['exploration']}_agent.csv"
        task_parameters = TaskParameters(experiment_path = f'./offline_result/RQ{args["question"]}/buffer_{args["exploration"]}')
        graph_manager = BCQ(DATASET_PATH)
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()
        delete_checkpoints()

    elif args['question'] == '3' and args['exploration'] != None:
        DATASET_PATH = f"./offline_data/Augmented Data {args['agent_timestep']}/buffer_{args['exploration']}_agent.csv"
        task_parameters = TaskParameters(experiment_path = f'./offline_result/RQ{args["question"]}/{args["agent_timestep"]}/buffer_{args["exploration"]}')
        graph_manager = BCQ(DATASET_PATH)
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()
        delete_checkpoints()