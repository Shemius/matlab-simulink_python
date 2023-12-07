import matlab.engine
import time
import argparse
import torch
import random
import numpy as np
import os
import datetime
import pandas as pd
from tensorboardX import SummaryWriter
from sac import SAC
from utils import setup_seed

rootdir = 'C:\\Users\\mjj26\\PycharmProjects\\PythonMatlabPendulum_share'
filename = os.path.basename(__file__).split(".")[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = rootdir + '/logs/' + filename + '_' + current_time  # logs for tensorboard


def main(args):
    setup_seed(20)
    # define the directory for model saving
    model_name = 'simu_pendulum_dim2'
    save_reward = -200
    model_path_best = rootdir + '/models/' + model_name + '/Best/'  # to save the actor by reinfoecement learning
    model_path_final = rootdir + '/models/' + model_name + '/Final/'  # to save the actor by reinfoecement learning
    if not os.path.exists(model_path_best):
        os.makedirs(model_path_best)
    if not os.path.exists(model_path_final):
        os.makedirs(model_path_final)
    if args.tensorboard:
        writer = SummaryWriter(logdir)

    # define the agent
    state_dim = 2  # sin(theta) cos(theta) dot_theta
    action_dim = 1  # continuous control
    # The torque action signal from the agent to the environment is from –2 to 2 N·m.
    action_bound = 2.0
    action_scale = 1.0

    agent = SAC(state_dim, action_dim, action_scale)

    # define the sample time and stop time
    sample_time = 0.05
    stop_time = 20
    step_max = int(stop_time / sample_time)

    # define the training parameters
    training_index = 0
    batch_size = 256
    auto_entropy = True
    max_episodes = 600
    loss_train = []

    # define the environment
    eng = matlab.engine.start_matlab()
    env_name = 'pendulum_v30'
    eng.load_system(env_name)

    # the training process
    num_training = 0
    for ep in range(max_episodes):
        t1 = time.time()
        # reset the environment
        eng.set_param(env_name, 'StopTime', str(21), nargout=0)  # 21 for 20 seconds
        eng.set_param(env_name + '/pause_time', 'value', str(0.01), nargout=0)
        eng.set_param(env_name + '/input', 'value', str(0), nargout=0)  # initial control signal
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)
        pause_time = 0.0

        obs_list, action_list, reward_list, done_list = [], [], [], []
        clock_list = []
        for step in range(step_max):
            # obtain the status of env
            model_status = eng.get_param(env_name, 'SimulationStatus')
            if model_status == 'paused':
                # obtain the latest observation
                clock = np.array(eng.eval('out.time.Data'))[-1]
                obs = np.array(eng.eval('out.obs.Data'))[-1]
                reward = np.array(eng.eval('out.reward.Data'))[-1]
                # control_singal = np.array(eng.eval('out.control.Data'))[-1]
                action = agent.policy_net.get_action(obs, deterministic=False)

                act = action * action_bound
                act = np.clip(act, -action_bound, action_bound)

                clock_list.append(clock)

                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                done_list.append(0.0)

                pause_time += sample_time
                # training process
                if (pause_time + 0.5) > stop_time:
                    done_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
                    len_list = len(obs_list)
                    for i1 in range(len_list - 1):
                        obs = obs_list[i1]
                        action = action_list[i1]
                        reward = reward_list[i1 + 1]
                        next_obs = obs_list[i1 + 1]
                        done = done_list[i1 + 1]
                        agent.replay_buffer.push(obs, action, reward, next_obs, done)
                    buffer_length = len(agent.replay_buffer)
                    if buffer_length > batch_size:
                        for _ in range(100):
                            value_loss, q_value_loss1, q_value_loss2, policy_loss = agent.train(batch_size,
                                                                                                reward_scale=0.1,
                                                                                                auto_entropy=True,
                                                                                                target_entropy=-1. * action_dim)
                            if args.tensorboard:
                                writer.add_scalar('Loss/V_loss', value_loss, global_step=num_training)
                                writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=num_training)
                                writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=num_training)
                                writer.add_scalar('Loss/pi_loss', policy_loss, global_step=num_training)
                                num_training += 1

                    ep_rd = np.sum(reward_list[1:])
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'
                          .format(ep, max_episodes, ep_rd, time.time() - t1))

                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd', ep_rd, global_step=ep)

                    break
                eng.set_param(env_name + '/input', 'value', str(act), nargout=0)  # initial control signal
                eng.set_param(env_name + '/pause_time', 'value', str(pause_time), nargout=0)
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

            elif model_status == 'running':
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

        if ep % 100 == 0 and ep != 0 and args.save_model:
            model_path_pip_epoch = rootdir + '/models/' + model_name + '/epoch/' + str(ep) + '/'
            if not os.path.exists(model_path_pip_epoch):
                os.makedirs(model_path_pip_epoch)
            agent.save_model(model_path_pip_epoch)
            print('=============The model is saved at epoch {}============='.format(ep))
    if args.save_model:
        agent.save_model(model_path_final)
        print('=============The final model is saved!==========')

    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--save_model', default=True, action="store_true")
    parser.add_argument('--save_data', default=False, action="store_true")
    args = parser.parse_args()
    main(args)