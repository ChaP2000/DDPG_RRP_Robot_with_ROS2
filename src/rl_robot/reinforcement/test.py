from reinforcement.utils.agent import robot_agent
from reinforcement.utils.misc import plot_success_rate
import numpy as np
import torch
import os
from ament_index_python.packages import get_package_share_directory

def main():
    batch_size = 64
    epsilon = 0
    epsilon_decay = 0
    env = robot_agent(capacity=50000, epsilon=epsilon, epsilon_decay=epsilon_decay, batch_size=batch_size, critic_learn = 1e-3, actor_learn = 5e-4,
                 tau = 0.005, gamma = 0.9)
    state, _, _ = env.reset()
    iterations = 200

    done_list = []

    path = os.path.join(get_package_share_directory('rl_robot'), 'checkpoint')

    env.load_checkpoint(path)

    for episode in range(iterations):

        done = 0

        state, _, _ = env.reset()

        curr_reward = 0

        for i in range(200):

            action = env.choose_action(state)

            obs, _, reward, terminated= env.step(action)

            curr_reward += reward

            if terminated:
                
                done = 1

            if done:
                break

            state = obs

        done_list.append(done)

        print("Current episode:", episode, ", finished with:", curr_reward, ", target was at", env.target, f", and target found:{done}")

    plot_success_rate(done_list=done_list)
    print(f"Overall Success rate is {sum(done_list)/len(done_list)*100}%")
    
if __name__ == "__main__":

    main()
