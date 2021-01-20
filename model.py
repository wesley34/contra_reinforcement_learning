import ptan
import numpy as np
import torch 
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self,act_size,obs_size,hidden_size):
        super(Actor,self).__init__()


        self.vision = nn.Sequential(
            nn.Conv2d(obs_size[0],hidden_size,kernel_size=8,stride=3),
            nn.ReLU(),
            nn.Conv2d(hidden_size,2*hidden_size,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(2*hidden_size,2*hidden_size,kernel_size=3,stride=1),
            nn.ReLU()
        )

        vision_output_shape = self._get_conv_out(obs_size)

        self.mean = nn.Sequential(
            nn.Linear(vision_output_shape,act_size),
            nn.Softmax(dim=1)
        )

        self.log_sigma = nn.Parameter(torch.zeros(act_size))

    def _get_conv_out(self,input_shape):
        fake_x = torch.zeros(size=(1,*input_shape))
        fake_y = self.vision(fake_x)
        return int(np.prod(fake_y.size()))

    def forward(self,x):
        fx = x.float()/256
        fx = self.vision(fx).view(fx.size()[0],-1)
        return self.mean(fx)

class Critic(nn.Module):
    def __init__(self,act_size,obs_size,hidden_size):
        super(Critic,self).__init__()

        self.vision = nn.Sequential(
            nn.Conv2d(obs_size[0],hidden_size,kernel_size=8,stride=3),
            nn.ReLU(),
            nn.Conv2d(hidden_size,2*hidden_size,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(2*hidden_size,2*hidden_size,kernel_size=3,stride=1),
            nn.ReLU()
        )
        vision_output_shape = self._get_conv_out(obs_size)

        self.value = nn.Sequential(
            nn.Linear(vision_output_shape,act_size),
            nn.ReLU(),
            nn.Linear(act_size,1),   
        )

    def _get_conv_out(self,input_shape):
        fake_x = torch.zeros(size=(1,*input_shape))
        fake_y = self.vision(fake_x)
        return int(np.prod(fake_y.size()))

    def forward(self,x):
        fx = x.float()/256
        fx = self.vision(fx).view(fx.size()[0],-1)
        return self.value(fx)

class AgentPPO(ptan.agent.BaseAgent):
    def __init__(self,actor_net,device):
        self.actor_net = actor_net
        self.device = device
        

    def __call__(self, states, agent_states):
   
        states_t = ptan.agent.float32_preprocessor(np.expand_dims(states[0],0)).to(self.device)

        mean_t = self.actor_net(states_t)

        mean_np = mean_t.data.data.cpu().numpy()
        log_sigma_np = self.actor_net.log_sigma.data.cpu().numpy()
        actions_np = mean_np + np.exp(log_sigma_np) * np.random.normal(size=log_sigma_np.shape)
        best_action = np.argmax(actions_np)
        #print("STASTA",agent_states)
        #print("ACTIONS np",np.argmax(actions_np))
        #actions_np = np.clip(actions_np,0,) # epsilon
        return [best_action] , agent_states