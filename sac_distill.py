import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from tqdm import trange
import math
import wandb
from stable_baselines3.common.buffers import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, env, n_layers):
        super().__init__()
        action_space = env.single_action_space

        input_dim = np.prod(env.single_observation_space.shape)
        output_dim = np.prod(action_space.shape)

        # Dynamically create the network based on the number of layers
        in_dims = [input_dim] + [256] * (n_layers + 1) if n_layers > 0 else [input_dim, input_dim]
        out_dims = [256] * n_layers + [output_dim, output_dim]
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])

        # Action Scaling and Bias
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        ) 
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, x):
        # Embed the input
        for i, fc_layer in enumerate(self.fc_list[:-2]):
            x = F.relu(fc_layer(x))
        # Get the mean and log_std
        mean = self.fc_list[-2](x)
        log_std = self.fc_list[-1](x)

        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        x = x.float()
        mean, log_std = self(x)
        std = log_std.exp()
        x_t = torch.randn_like(mean) * std + mean  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = -0.5 * ((x_t - mean) / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    def __init__(self, env, n_layers):
        super().__init__()

        input_dim = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)

        in_dims = [input_dim] + [256] * n_layers
        out_dims = [256] * n_layers + [1]
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)])

    def forward(self, x, a): 
        x = x.float()
        x = torch.cat([x, a], 1)

        for i, fc_layer in enumerate(self.fc_list[:-1]):
            x = F.relu(fc_layer(x))
        x = self.fc_list[-1](x)

        return x

SACComponents = namedtuple("SACComponents", ["actor", "qf1", "qf2", "qf1_target", "qf2_target", "q_optimizer", "actor_optimizer", "rb", "target_entropy", "log_alpha", "a_optimizer", "counter"])

def setup_sac(env, cfg, actor_depth=0):
    device = cfg.device
    actor = Actor(env, n_layers=actor_depth).to(device)  # Linear actor
    qf1 = SoftQNetwork(env, n_layers=2).to(device)
    qf2 = SoftQNetwork(env, n_layers=2).to(device)
    qf1_target = SoftQNetwork(env, n_layers=2).to(device)
    qf2_target = SoftQNetwork(env, n_layers=2).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.sac.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.sac.policy_lr)

    if cfg.sac.alpha_auto == True:
        target_entropy = -torch.prod(torch.tensor(env.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=cfg.sac.q_lr)
    else:
        target_entropy = None
        log_alpha = None
        a_optimizer = None

    # MinMax Replay Buffer so we can add new best or worst experiences
    rb = ReplayBuffer(
        cfg.sac.buffer_size,
        env.single_observation_space,
        env.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    counter = {'n_steps': 0}

    return SACComponents(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, rb, target_entropy, log_alpha, a_optimizer, counter)

def train_sac(cfg, sac, sac_idx):
    if cfg.sac.alpha_auto == True:
        alpha = sac.log_alpha.exp().item()
    else:
        alpha = cfg.sac.alpha

    data = sac.rb.sample(cfg.sac.batch_size)
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = sac.actor.get_action(data.next_observations)
        qf1_next_target = sac.qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = sac.qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg.sac.gamma * (min_qf_next_target).view(-1)

    qf1_a_values = sac.qf1(data.observations, data.actions).view(-1)
    qf2_a_values = sac.qf2(data.observations, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # optimize the model
    sac.q_optimizer.zero_grad()
    qf_loss.backward()
    sac.q_optimizer.step()

    if sac.counter['n_steps'] % cfg.sac.policy_frequency == 0:  # TD 3 Delayed update support
        for _ in range(cfg.sac.policy_frequency):  # compensate for the delay in policy updates
            pi, log_pi, _ = sac.actor.get_action(data.observations)
            qf1_pi = sac.qf1(data.observations, pi)
            qf2_pi = sac.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            sac.actor_optimizer.zero_grad()
            actor_loss.backward()
            sac.actor_optimizer.step()

            if cfg.sac.alpha_auto == True:
                with torch.no_grad():
                    _, log_pi, _ = sac.actor.get_action(data.observations)
                alpha_loss = (-sac.log_alpha.exp() * (log_pi + sac.target_entropy)).mean()

                sac.a_optimizer.zero_grad()
                alpha_loss.backward()
                sac.a_optimizer.step()
                alpha = sac.log_alpha.exp().item()

    if sac.counter['n_steps'] % cfg.sac.target_network_frequency == 0:
        for param, target_param in zip(sac.qf1.parameters(), sac.qf1_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)
        for param, target_param in zip(sac.qf2.parameters(), sac.qf2_target.parameters()):
            target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)

    sac.counter['n_steps']  += 1


if __name__ == "__main__":
    import os
    import argparse
    import logging
    import gymnasium as gym

    from configs.utils import init_cfg

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_local", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=1e6)
    parser.add_argument("--env_id", type=str, default="Hopper-v4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=420)
    new_args = vars(parser.parse_args())

    cfg = init_cfg("configs/hopper.yml")
    cfg.update(new_args)
    cfg.log.update(new_args)

    if cfg.log.wandb == True:
        wandb.init(project="sac", name=new_args["run_name"])
    if cfg.log.log_local == True:
        if not os.path.exists(cfg.log.log_dir):
            os.makedirs(cfg.log.log_dir)
        logging.basicConfig(filename=f"{cfg.log.log_dir}/{new_args['run_name']}.log", level=logging.INFO, format="%(message)s")

    n_envs = 1
    envs = gym.vector.AsyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id,)) for _ in range(n_envs)]
    )
    envs_2 = gym.vector.AsyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(cfg.env_id,)) for _ in range(n_envs)]
    )

    sac_student = setup_sac(envs, cfg, actor_depth=0)
    sac_teacher = setup_sac(envs, cfg, actor_depth=2)

    obs, _ = envs.reset(seed=cfg.seed)
    obs_2, _ = envs_2.reset(seed=cfg.seed)

    for step_idx in trange(new_args["n_steps"]):
        actions = sac_teacher.actor.get_action(torch.tensor(obs, dtype=torch.float32).to(cfg.device))
        actions = actions[0].cpu().detach().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        actions_2 = sac_student.actor.get_action(torch.tensor(obs_2, dtype=torch.float32).to(cfg.device))
        actions_2 = actions_2[0].cpu().detach().numpy()
        next_obs_2, rewards_2, terminations_2, truncations_2, infos_2 = envs_2.step(actions_2)

        if "final_info" in infos_2:
            for info in infos_2["final_info"]:
                if info:
                    if cfg.log.wandb == True:
                        wandb.log({
                            f"sac/episodic_rew": info['episode']['r'],
                            f"sac/log_idx": step_idx,
                        })
                    if cfg.log.log_local == True:
                        logging.info(f"{step_idx}, {info['episode']['r'][0]}")

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        sac_teacher.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        sac_student.rb.add(obs, next_obs, actions, rewards, terminations, infos)

        obs = next_obs
        obs_2 = next_obs_2

        train_sac(cfg, sac_teacher, 0)
        train_sac(cfg, sac_student, 0)
