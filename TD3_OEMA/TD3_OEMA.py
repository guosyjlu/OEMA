import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        return q

class Permutation(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.1, expl_noise=0.1):
        super(Permutation, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.phi = phi
        self.expl_noise = expl_noise

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        a = F.relu(self.l1(sa))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = self.phi * self.max_action * a
        a_ = torch.normal(0, self.expl_noise, a.shape).to(device)
        return (a + a_ + action).clamp(-self.max_action, self.max_action)



class TD3_OEMA(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            temperature=1,
            optimism_level=1.0,
            phi=1,
            expl_noise=0.1,
            update_freq=1,
            beta=0.01,
            anneal=True
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.weight = MLP(state_dim, action_dim).to(device)
        self.weight_optimizer = torch.optim.Adam(self.weight.parameters(), lr=3e-4)

        self.permutation = Permutation(state_dim, action_dim, max_action, phi, expl_noise).to(device)
        self.permutation_optimizer = torch.optim.Adam(self.permutation.parameters(), lr=3e-4)

        self.hotplug = Hot_Plug(self.actor)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.temperature = temperature
        self.optimism_level = optimism_level
        self.update_freq = update_freq
        self.beta = beta
        self.anneal_step = beta / float((305000 - 25000) / 2) if anneal else 0

        self.total_it = 0

    def select_behavior_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        action = self.permutation(state, action)
        return action.cpu().data.numpy().flatten()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch, priority_buffer):
        self.total_it += 1

        # Weight net training
        offline_state = torch.FloatTensor(batch["offline_observations"]).to(device)
        offline_action = torch.FloatTensor(batch["offline_actions"]).to(device)
        online_state = torch.FloatTensor(batch["online_observations"]).to(device)
        online_action = torch.FloatTensor(batch["online_actions"]).to(device)

        offline_weight = self.weight(offline_state, offline_action)
        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)
        online_weight = self.weight(online_state, online_action)
        online_f_prime = torch.log(2 * online_weight / (online_weight + 1) + 1e-10)
        weight_loss = (offline_f_star - online_f_prime).mean()
        self.weight_optimizer.zero_grad()
        weight_loss.backward()
        self.weight_optimizer.step()

        # Sample replay buffer
        state = torch.FloatTensor(batch["rl_observations"]).to(device)
        action = torch.FloatTensor(batch["rl_actions"]).to(device)
        next_state = torch.FloatTensor(batch["rl_next_observations"]).to(device)
        reward = torch.FloatTensor(batch["rl_rewards"]).to(device)
        done = torch.FloatTensor(batch["rl_terminals"]).to(device)
        meta_state = torch.FloatTensor(batch["meta_observations"]).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            """ Meta Training"""
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.hotplug.update(3e-4)

            """Meta Testing"""
            self.beta = max(0.0, self.beta - self.anneal_step)
            meta_actor_loss = -self.critic.Q1(meta_state, self.actor(meta_state)).mean()
            weight = self.beta * actor_loss.detach() / meta_actor_loss.detach()
            meta_actor_loss_norm = weight * meta_actor_loss
            meta_actor_loss_norm.backward(create_graph=True)

            """Meta Optimization"""
            self.actor_optimizer.step()
            self.hotplug.restore()
            if self.total_it % 500 == 0:
                print(
                    f"[Meta Learning] actor-loss:{actor_loss:.4f}, meta-actor-loss:{meta_actor_loss:.4f}, meta-actor-loss-norm:{meta_actor_loss_norm:.4f}")

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % self.update_freq == 0:
            # Compute behavior actor loss
            with torch.no_grad():
                target_action = self.actor(state)
            behavior_action = self.permutation(state, target_action)
            Q1, Q2 = self.critic(state, behavior_action)
            mean = (Q1 + Q2) / 2.0
            var = torch.abs(Q1 - Q2) / 2.0
            Q_UB = mean + self.optimism_level * var
            permutation_loss = -Q_UB.mean()
            self.permutation_optimizer.zero_grad()
            permutation_loss.backward()
            self.permutation_optimizer.step()

        # Update priority buffer
        with torch.no_grad():
            weight = self.weight(state, action)
            normalized_weight = (weight ** (1 / self.temperature)) / (
                (offline_weight ** (1 / self.temperature)).mean() + 1e-10
            )
        new_priority = normalized_weight.clamp(0.001, 1000)
        priority_buffer.update_priorities(
            batch["tree_idxs"].squeeze().astype(np.int),
            new_priority.squeeze().detach().cpu().numpy(),
        )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
