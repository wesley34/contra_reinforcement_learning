import os
import ptan
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from train import train_handler
from model import Actor,Critic,AgentPPO
from env import create_gym


GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 100000
HIDDEN_SIZE = 32



if __name__ == "__main__":

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #print("current device : ",device) 

    #save_path_name = os.path.join("saves","ppo")
    #os.makedirs(save_path_name,exist_ok=True)
    #print("save path name : ", save_path_name)

    
    env = create_gym()
    test_env = create_gym()

    action_size = env.action_space.n
    observation_size = env.observation_space.shape
    print("actions : ", action_size)
    print("observation_space :", observation_size)


    
    
    actor_net = Actor(action_size,observation_size,HIDDEN_SIZE).to(device)
    critic_net = Critic(action_size,observation_size,HIDDEN_SIZE).to(device)

    agent = AgentPPO(actor_net=actor_net,device=device)
    actor_optimizer = optim.Adam(actor_net.parameters(),lr=LEARNING_RATE_ACTOR)
    critic_optimizer = optim.Adam(critic_net.parameters(),lr=LEARNING_RATE_CRITIC)

    experiment_source = ptan.experience.ExperienceSource(env,agent,1)
    writer = SummaryWriter(comment="ppo_" + "CONTRA")

    train_handler(actor_net,critic_net,actor_optimizer,critic_optimizer,writer,experiment_source,test_env,device,GAMMA,GAE_LAMBDA,TRAJECTORY_SIZE,PPO_EPOCHES,PPO_BATCH_SIZE,PPO_EPS,TEST_ITERS)


