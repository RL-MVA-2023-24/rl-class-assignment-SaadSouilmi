from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import numpy as np
from copy import deepcopy
import random
import os
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm


seed = 42

random.seed(seed)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

nb_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

parser = ArgumentParser(description="Agent config")
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--gamma", default=0.98, type=float)
parser.add_argument("--buffer_size", default=1000000, type=float)
parser.add_argument("--epsilon_min", default=0.01, type=float)
parser.add_argument("--epsilon_max", default=1., type=float)
parser.add_argument("--epsilon_decay_period", default=10000, type=int)
parser.add_argument("--epsilon_delay_decay", default=400, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--gradient_steps", default=2, type=int)
parser.add_argument("--update_target_strategy", default="ema", type=str)
parser.add_argument("--update_target_freq", default=600, type=int)
parser.add_argument("--update_target_tau", default=0.001, type=float)
parser.add_argument("--monitoring_nb_trials", default=50, type=int)
parser.add_argument("--evaluation_frequency", default=50, type=int)
parser.add_argument("--network_depth", default=5, type=int)
parser.add_argument("--nb_neurons", default=512, type=int)


args = parser.parse_args()

config = {'nb_actions': nb_actions,
          'criterion': torch.nn.SmoothL1Loss(),
          'double_dqn': True,
          'nb_neurons_val': 1024,
          'nb_neurons_adv': 1024,
          'depth_val': 4,
          'depth_adv': 4,
          **vars(args)}


class DQN(nn.Module):
    def __init__(self, input_dim, nb_neurons, output_dim, depth, activation=nn.SiLU()):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(input_dim, nb_neurons)
        self.network = nn.ModuleList([nn.Linear(nb_neurons, nb_neurons) for _ in range(depth - 1)])
        self.out_layer = nn.Linear(nb_neurons, output_dim)
        self.activation = activation
    
    def forward(self, x):
        x = self.activation(self.in_layer(x))
        for hidden_layer in self.network:
            x = self.activation(hidden_layer(x))
        return self.out_layer(x)

class Dueling_DQN(nn.Module):
    def __init__(self, input_dim, nb_neurons_val, nb_neurons_adv, output_dim, depth_val, depth_adv, activation=nn.SiLU()):
        super(Dueling_DQN, self).__init__()
        self.in_layer_val = nn.Linear(input_dim, nb_neurons_val)
        self.network_val = nn.ModuleList([nn.Linear(nb_neurons_val, nb_neurons_val) for _ in range(depth_val-1)])
        self.out_layer_val = nn.Linear(nb_neurons_val, 1)

        self.in_layer_adv = nn.Linear(input_dim, nb_neurons_adv)
        self.network_adv = nn.ModuleList([nn.Linear(nb_neurons_adv, nb_neurons_adv) for _ in range(depth_adv-1)])
        self.out_layer_adv = nn.Linear(nb_neurons_adv, output_dim)

        self.activation = activation

    def forward(self, x):
        val = self.activation(self.in_layer_val(x))
        for hidden_layer in self.network_val:
            val = self.activation(hidden_layer(val))
        val = self.out_layer_val(val)

        adv = self.activation(self.in_layer_adv(x))
        for hidden_layer in self.network_adv:
            adv = self.activation(hidden_layer(adv))
        adv = self.out_layer_adv(adv)

        return val + adv - adv.mean()

def greedy_action(network, state):
    if next(network.parameters()).is_cuda:
        device = "cuda"
    else:
        device = "mps" if next(network.parameters()).is_mps else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def warm_up(agent, warm_up_len):
    s, _ = env.reset()
    for i in tqdm(range(warm_up_len), desc="warm up run for replay buffer"):
        a = agent.act(s)
        s_, r, d, t, _ = env.step(a)
        agent.memory.append(s, a, r, s_, d)
        if d or t:
            s, _ = env.reset()
        else:
            s = s_


class dqn_agent:
    def __init__(self, config, model):
        if next(model.parameters()).is_cuda:
            device = "cuda"
        else:
            device = "mps" if next(model.parameters()).is_mps else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.evaluation_frequency = config['evaluation_frequency']
        self.double_dqn = config["double_dqn"]
    
    def act(self, state):
        return greedy_action(self.model, state)

    def MC_eval(self, env, nb_trials):  
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials): 
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if self.double_dqn:
            if len(self.memory) > self.batch_size:
                X, A, R, Y, D = self.memory.sample(self.batch_size)
                evaluated_action = torch.argmax(self.model(Y).detach(), dim=1) # action is greedy w.r.t online network
                QY = self.target_model(Y).detach() 
                QYmax = QY.gather(1, evaluated_action.unsqueeze(1)).squeeze(1) # action evaluated on target network
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        else:
            if len(self.memory) > self.batch_size:
                X, A, R, Y, D = self.memory.sample(self.batch_size)
                QYmax = self.target_model(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   
        MC_avg_discounted_reward = []  
        V_init_state = []   
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if rng.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0 and episode%self.evaluation_frequency == 0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)   
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   
                    MC_avg_total_reward.append(MC_tr)  
                    MC_avg_discounted_reward.append(MC_dr)  
                    V_init_state.append(V0)   
                    episode_return.append(episode_cum_reward)  
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                    if MC_tr > best_score:
                        best_score = MC_tr
                        self.save(f"{os.getcwd()}/src/best_agent.pth")
                        print(f"Current best score: {best_score}")
                    else:
                        print(f"Current best score: {best_score}")
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()


#model = DQN(state_dim, config["nb_neurons"], nb_actions, depth=config["network_depth"], activation=nn.SiLU()).to(device)
model = Dueling_DQN(state_dim, config["nb_neurons_val"], config["nb_neurons_adv"], nb_actions, depth_val=config["depth_val"], depth_adv=config["depth_adv"], activation=nn.SiLU()).to(device)
agent = dqn_agent(config, model)

class ProjectAgent:
    def __init__(self) -> None:
        self.dqn_agent = dqn_agent(config, model)
        
    def act(self, observation, use_random=False):
        if use_random:
            return rng.choice(env.action_space.n)
        else:
            return self.dqn_agent.act(observation)

    def save(self, path):
        pass

    def load(self):
        dir = f"{os.getcwd()}/src/best_agent.pth"
        self.dqn_agent.load(dir)




if __name__ == "__main__":

    print(device)
    print(config)
    warm_up(agent, 10000)
    ep_return, disc_rewards, tot_rewards, v_init = agent.train(env, 4000)
    
