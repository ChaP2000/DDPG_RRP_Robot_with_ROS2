from reinforcement.utils.agent import robot_agent
from reinforcement.utils.misc import plot_success_rate
import numpy as np
import torch
import os
from ament_index_python.packages import get_package_share_directory

def main():
    batch_size = 64
    epsilon = 0.99
    epsilon_decay = 0.996
    env = robot_agent(capacity=50000, epsilon=epsilon, epsilon_decay=epsilon_decay, batch_size=batch_size, critic_learn = 1e-3, actor_learn = 5e-4,
                 tau = 0.005, gamma = 0.9)
    state, _, _ = env.reset()
    iterations = 5000

    done_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for episode in range(iterations):

        done = 0

        state, _, target = env.reset()

        curr_reward = 0

        for i in range(200):

            action = env.choose_action(state)

            obs, _, reward, terminated= env.step(action)

            curr_reward += reward

            if terminated:

                next_state = None
                done = 1

            else:

                next_state = torch.tensor(obs, dtype = torch.float32, device= device).unsqueeze(0)

            state = torch.tensor(state, dtype = torch.float32, device= device).unsqueeze(0)

            action = torch.tensor(action, dtype = torch.float32, device= device).unsqueeze(0)

            reward = torch.tensor(reward, dtype = torch.float32, device= device).unsqueeze(0)

            env.push_memory((state, action,reward, next_state))

            env.optimize()

            if done:
                break

            state = obs

        done_list.append(done)

        env.decay_epsilon()
        print("Current episode:", episode, ", finished with reward:", curr_reward, ", target was at", target, f", and target found:{done}")
    
    path = os.path.join(get_package_share_directory('rl_robot'), 'checkpoint')
    
    env.save_checkpoint(path=path)

    plot_success_rate(done_list=done_list)
    
if __name__ == "__main__":

    main()
