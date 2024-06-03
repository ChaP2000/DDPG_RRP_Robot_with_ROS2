import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(reward_list):

    reward_arr = np.array(reward_list)
    avg = np.zeros(len(reward_arr)-100)

    for i in range(len(avg)):
        avg[i] = np.mean(reward_arr[i:i+100])

    plt.figure()
    plt.plot(list(range(100,len(avg)+100)), avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

def plot_success_rate(done_list):

    done_arr = np.array(done_list)
    avg = np.zeros(len(done_arr)-100)

    for i in range(len(avg)):
        avg[i] = np.mean(done_arr[i:i+100])

    plt.figure()
    plt.plot(list(range(100, len(avg)+100)), avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Success Rate')
    plt.show()

def cubic_polynomial(start, end):

    timestamps = np.arange(0, 1.1 ,0.1)

    a_0 = start[:, None]

    a_1 = np.zeros(len(start))[:,None]

    a_2 = 3/timestamps[-1]**2*(end-start)[:, None]

    a_3 = -2/timestamps[-1]**3*(end-start)[:, None]

    waypoints = a_0 + a_1*timestamps + a_2*timestamps**2 + a_3 * timestamps**3

    return waypoints