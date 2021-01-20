import torch
import numpy as np
import math

def get_log_probability(mean,log_sigma,action):
    '''
    #################
    algo :
        normal_distribtuion = 1/root(2 pi sigma) * e^(-(x-mean)^2/(2 sigma))
        log(normal_distribtuion) = - log (root(2 pi sigma)) - (x-mean)^2/(2 sigma)
    #################
    return log(normal)
    '''
    
    log_normal = -torch.log(torch.sqrt(2*math.pi*torch.exp(log_sigma))) -((mean-action) ** 2 / (2 * torch.exp(sigma).clamp(min=1e-3)))
    return log_normal

def get_reward_advantage(trajectory, critic_net, state_t,gamma,gae_gamma, device):
    '''
    #################
    algo:
        use gae method to cal advantange and reward
    #################
    return advantange , q value
    '''
    gae = 0
    advantages = []
    values = []

    value_t = critic_net(state_t)

 
    values= value_t.squeeze().data.cpu().numpy()

    # gae [T,T-1,....,1]
    for value, next_value , exp in zip(reversed(values[:-1]),reversed(values[1:]),reversed(trajectory)):
        if exp.done:
            gae = exp.reward - value
        else:
            delta = exp.reward + gamma * next_value - value
            gae = delta + gamma * gae_gamma * gae
        
        advantages.append(gae)
        values.append(gae+value)
    # reverse back to [1,...,T]
    advantages_t = torch.FloatTensor(list(reversed(advantages))).to(device)
    values_t = torch.FloatTensor(list(reversed(values))).to(device)

    return advantages_t , values_t
