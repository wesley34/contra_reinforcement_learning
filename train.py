import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import ptan

import utils
def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            #print("obs",obs)
            #print(np.expand_dims(obs,0))
            obs_v = ptan.agent.float32_preprocessor(np.expand_dims(obs,0)).to(device)
            mu_v = net(obs_v)[0]
            
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.argmax(action)
            #print("action",action)
            #action = np.clip(action, -1, 1)
            #print("action",action)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def train_handler(actor_net,critic_net,actor_optimizer,critic_optimizer,writer,exp_source,test_env,device,gamma,gae_gamma,trajectory_size,ppo_epoches,ppo_batch_size,ppo_epislon,test_iter):
    trajectory = []
    rewards = None

    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx , exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % test_iter == 0:
                ts = time.time()
                rewards, steps = test_net(actor_net, test_env, count=10,device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, step_idx)
                writer.add_scalar("test_steps", steps, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_act.state_dict(), fname)
                    best_reward = rewards
            
            trajectory.append(exp)
            if len(trajectory) < trajectory_size:
                continue

            trajectory_states = [t[0].state for t in trajectory]
            trajectory_actions = [t[0].state for t in trajectory]
            trajectory_states_t = torch.FloatTensor(trajectory_states).to(device)
            trajectory_actions_t = torch.FloatTensor(trajectory_actions).to(device)

            trajectory_advantages_t , trajectory_q_value_t = utils.get_reward_advantage(trajectory,critic_net,trajectory_actions_t,gamma,gae_gamma,device)
            mean_t = actor_net(trajectory_states_t)
            old_log_probability = utils.get_log_probability(mean,actor_net.log_sigma,trajectory_actions_t)

            trajectory_advantage_t = (trajectory_advantages_t-torch.mean(trajectory_advantages_t))/torch.std(trajectory_advantages_t)

            trajectory = trajectory[:-1]
            old_log_probability_t = old_log_probability[:-1].detach()

            sum_loss_value = 0
            sum_loss_policy = 0
            count_steps = 0

            for epoch in range(ppo_epoches):
                for batch_offsets in range(0,len(trajectory),ppo_batch_size):
                    states = trajectory_states_t[batch_offsets:batch_offsets+ppo_batch_size]
                    actions = trajectory_actions_t[batch_offsets:batch_offsets+ppo_batch_size]
                    
                    batch_advantages = trajectory_advantages_t[batch_offsets:batch_offsets+ppo_batch_size]
                    batch_q_value = trajectory_q_value_t[batch_offsets:batch_offsets+ppo_batch_size]
                    batch_old_log_probability = old_log_probability_t[batch_offsets:batch_offsets+ppo_batch_size]


                    # critic 
                    critic_optimizer.zero_grad()
                    value_v = critic_net(states)
                    loss_value = F.mse_loss(value_v.squeeze(-1),batch_q_value)
                    loss_value.backward()
                    critic_optimizer.step()

                    # actor 
                    actor_optimizer.zero_grad()
                    mean = actor_net(states)
                    log_probability = utils.get_log_probability(mean,actor_net.log_sigma,actions)
                    ratio = torch.exp(log_probability-batch_old_log_probability)

                    surrogate_cost = batch_advantages * ratio
                    clip_surrogate_cost = batch_advantages * torch.clamp(ratio,1-ppo_epislon,1+ppo_epislon)
                    loss_policy = -torch.min(surrogate_cost,clip_surrogate_cost)
                    loss_policy.backward()
                    actor_optimizer.step()

                    sum_loss_value += loss_value
                    sum_loss_policy += loss_policy
                    count_steps+=1
                    
            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
            writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)


