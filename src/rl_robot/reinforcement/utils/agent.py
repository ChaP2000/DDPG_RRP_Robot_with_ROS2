import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Actor(nn.Module):

  def __init__(self, n_observations, action_dim):

    super().__init__()

    torch.manual_seed(1111)

    self.layer1 = nn.Linear(n_observations, 512)
    self.layer2 = nn.Linear(512, 256)
    self.layer3 = nn.Linear(256, 128)
    self.layer4 = nn.Linear(128, action_dim)

  def forward(self, x):

    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = F.relu(self.layer3(x))
    x = F.tanh(self.layer4(x))

    return x

class Critic(nn.Module):

  def __init__(self, n_observations, action_dim):

    super().__init__()

    torch.manual_seed(1111)

    self.layer1 = nn.Linear(n_observations + action_dim, 256)
    self.layer2 = nn.Linear(256, 1)

  def forward(self, x, act):

    x = F.relu(self.layer1(torch.cat((x,act), dim = 1)))
    x = self.layer2(F.relu(x))

    return x


class robot_agent:

    def __init__(self, capacity, epsilon, epsilon_decay, 
                 batch_size, critic_learn = 2e-3, actor_learn = 4e-4,
                 tau = 0.005, gamma = 0.9):

        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.init_state = np.array([0,0,0,0.5,1,0.01], dtype= np.float32)
        self.joint_state = np.array([0.5,1,0.01], dtype= np.float32)
        self.target = None
        self.exp = deque([])
        self.max_capacity = capacity
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        n_observations = len(self.init_state)
        action_dim = 3

        self.crit = Critic(n_observations, action_dim)
        self.target_crit = Critic(n_observations, action_dim)
        self.target_crit.load_state_dict(self.crit.state_dict())
        self.crit.to(self.device)
        self.target_crit.to(self.device)

        self.optim_crit = optim.AdamW(self.crit.parameters(), lr = critic_learn)

        self.act = Actor(n_observations, action_dim)
        self.target_act = Actor(n_observations, action_dim)
        self.target_act.load_state_dict(self.act.state_dict())
        self.act.to(self.device)
        self.target_act.to(self.device)

        self.optim_act = optim.AdamW(self.act.parameters(), lr = actor_learn)

    
    def decay_epsilon(self):
        
        self.epsilon = self.epsilon * self.epsilon_decay

    def choose_action(self, states):

        seed = np.random.rand(1)

        states = torch.tensor(states, dtype = torch.float32, device= self.device)

        if seed >= self.epsilon:

            with torch.no_grad():

                return np.array(self.act(states).detach().cpu())

        action = self.sample_action()

        return action

    def sample_memory(self, batch_size):

        return random.sample(self.exp, batch_size)

    def push_memory(self, new_exp):

        self.exp.append(new_exp)
        if self.max_capacity < len(self.exp):
            self.exp.popleft()

    def sample_action(self):

        out_1 = np.random.rand()
        out_2 = np.random.rand()
        out_3 = np.random.rand()*0.05
        out = (out_1, out_2, out_3)
        return out

    def reset(self):

        self.joint_state = np.array([0.5,1,0.01], dtype= np.float32)

        joint_1 = np.random.rand()*6-3
        joint_2 = np.random.rand()+1
        joint_3 = np.random.rand()*0.05

        target_state = np.array([joint_1, joint_2, joint_3])

        self.target = self.forward_kinematics(target_state)

        distance = (self.forward_kinematics(self.joint_state)-self.target)

        self.init_state = np.hstack((distance, self.joint_state))

        return self.init_state, np.copy(self.joint_state), self.target

    def step(self, action):

        terminated = 0

        self.joint_state += 0.1*np.array(action)

        self.joint_state[0] = np.clip(self.joint_state[0], -3, 3)

        self.joint_state[1] = np.clip(self.joint_state[1], 1, 2)

        self.joint_state[2] = np.clip(self.joint_state[2], -0.1, 0.05)

        cartesian_current = self.forward_kinematics(self.joint_state)

        distance = (cartesian_current-self.target)

        new_state = np.hstack((distance, self.joint_state))

        reward = (-(np.sum((cartesian_current-self.target)**2, axis = None))**0.5)

        if reward > - 0.01:

            reward = 10

            terminated = 1

            print(f"the final location is {cartesian_current}")

        return new_state, self.joint_state, reward, terminated

    def forward_kinematics(self, joint_state):

        def create_dh_mat(a, alpha, d, theta):

            out = np.zeros((4,4), dtype= np.float32)
            out[-1,-1] = 1
            out[0, :] = np.array([np.cos(theta), -np.sin(theta), 0 , a], dtype= np.float32)
            out[1, :] = np.array([np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha) , -d*np.sin(alpha)], dtype= np.float32)
            out[2, :] = np.array([np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha) , d*np.cos(alpha)], dtype= np.float32)
            return out

        out_1 = create_dh_mat(0, 0, 0.3, joint_state[0])
        out_2 = out_1 @ create_dh_mat(0.3, 0, 0.05, joint_state[1])
        out_3 = out_2 @ create_dh_mat(0.3, 0, joint_state[2], 0)
        out_4 = out_3 @ create_dh_mat(0, 0, -0.11, 0)

        origin = np.array([0,0,0,1], dtype= np.float32).reshape(-1,1)
        o1 = out_1 @ origin
        o2 = out_2 @ origin
        o3 = out_3 @ origin
        o_final = out_4 @ origin

        oend = o_final.T[:,:-1].squeeze(0)
        return oend

    def optimize(self):

        if len(self.exp) < self.batch_size:

            return

        else:

            batch = self.sample_memory(self.batch_size)

            batch = list(zip(*batch))

            states = torch.cat(batch[0], dim = 0)

            actions = torch.cat(batch[1], dim = 0)

            rewards = torch.cat(batch[2], dim = 0)

            next_states = torch.cat([s for s in batch[-1] if s is not None], dim = 0)

            state_action_values = self.crit(states, actions)

            mask = torch.tensor(tuple(map(lambda s: s is not None, batch[-1])), device= self.device)
            next_state_action_values = torch.zeros(self.batch_size, device= self.device)

            with torch.no_grad():

                action_targets = self.target_act(next_states)

            next_state_action_values[mask] = self.target_crit(next_states, action_targets).squeeze()

            expected_state_action_values = next_state_action_values*self.gamma + rewards

            criterion = nn.MSELoss()

            loss_crit = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

            self.optim_crit.zero_grad()
            loss_crit.backward()
            self.optim_crit.step()

            action_targets = self.act(states)

            loss_act = -self.crit(states, action_targets)

            loss_act = loss_act.mean()

            self.optim_act.zero_grad()
            loss_act.backward()
            torch.nn.utils.clip_grad_value_(self.act.parameters(), 0.05)
            self.optim_act.step()

            crit_state_dict = self.crit.state_dict()
            target_crit_state_dict = self.target_crit.state_dict()

            TAU = self.tau
            for key in crit_state_dict:
                target_crit_state_dict[key] = crit_state_dict[key]*TAU + target_crit_state_dict[key]*(1-TAU)
            self.target_crit.load_state_dict(target_crit_state_dict)

            act_state_dict = self.act.state_dict()
            target_act_state_dict = self.target_act.state_dict()
            for key in act_state_dict:
                target_act_state_dict[key] = act_state_dict[key]*TAU + target_act_state_dict[key]*(1-TAU)
            self.target_act.load_state_dict(target_act_state_dict)

    def load_checkpoint(self, path):
        self.act.load_state_dict(torch.load(os.path.join(path,'act.pth')))
        self.crit.load_state_dict(torch.load(os.path.join(path,'crit.pth')))
        self.target_act.load_state_dict(torch.load(os.path.join(path,'target_act.pth')))
        self.target_crit.load_state_dict(torch.load(os.path.join(path,'target_crit.pth')))
    
    def save_checkpoint(self, path):
        torch.save(self.act.state_dict(), os.path.join(path,'act.pth'))
        torch.save(self.crit.state_dict(), os.path.join(path,'crit.pth'))
        torch.save(self.target_act.state_dict(), os.path.join(path,'target_act.pth'))
        torch.save(self.target_crit.state_dict(), os.path.join(path,'target_crit.pth'))