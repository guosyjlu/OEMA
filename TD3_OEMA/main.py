import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import TD3_OEMA
from env_replay_buffer import EnvReplayBuffer
from prioritized_replay_buffer import PriorityReplayBuffer


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed=0, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.array(state).reshape(1, -1)
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return avg_reward, d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_OEMA")  # Online policy name
    parser.add_argument("--env", default="halfcheetah-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True)  # Whether load offline model
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    # TD3
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    # Balance Replay
    parser.add_argument("--temperature", default=5.0, type=float)
    # Optimistic Exploration
    parser.add_argument("--optimism_level", default=4.0, type=float)  # Optimism level
    parser.add_argument("--phi", default=1.0, type=float)  # maximum permutation bound
    parser.add_argument("--update_freq", default=1, type=int)  # Update frequency of the permutation model
    # Meta Adaptation
    parser.add_argument("--beta", default=0.01, type=float)  # Meta objective weight
    parser.add_argument("--anneal", type=eval, choices=[True, False], default='False')  # Apply annealing or not
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.optimism_level}_{args.beta}_{args.anneal}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print(f"Optimism Level: {args.optimism_level}, Beta: {args.beta}, Anneal: {args.anneal}")
    print("---------------------------------------")

    if not os.path.exists("../results"):
        os.makedirs("../results")

    if args.save_model and not os.path.exists("../models"):
        os.makedirs("../models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # BR
        "temperature": args.temperature,
        # OE
        "optimism_level": args.optimism_level,
        # MA
        "anneal": args.anneal,
        "beta": args.beta
    }

    # Initialize policy
    policy = TD3_OEMA.TD3_OEMA(**kwargs)

    if args.load_model:
        print("Loading Policy...")
        load_file_name = f"../models/TD3_BC_{args.env}_0"
        policy.load(filename=load_file_name)

    ds = d4rl.qlearning_dataset(env)
    dataset_size = ds["observations"].shape[0]
    priority_replay_buffer = PriorityReplayBuffer(int(1e6), env)
    offline_replay_buffer = EnvReplayBuffer(int(1e6), env)
    online_replay_buffer = EnvReplayBuffer(int(1e6), env)
    for i in range(dataset_size):
        obs = ds["observations"][i]
        new_obs = ds["next_observations"][i]
        action = ds["actions"][i]
        reward = ds["rewards"][i]
        done = ds["terminals"][i]
        priority_replay_buffer.add_sample(obs, action, reward, done, new_obs)
        offline_replay_buffer.add_sample(obs, action, reward, done, new_obs)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    evaluations = []
    evaluations.append(eval_policy(policy, args.env))
    best_reward = evaluations[0][0]

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = (policy.select_action(np.array(state))).clip(-max_action, max_action)
        else:
            action = (policy.select_behavior_action(np.array(state)))
        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        online_replay_buffer.add_sample(state, action, reward, done_bool, next_state)
        priority_replay_buffer.add_sample(state, action, reward, done_bool, next_state)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            train_data_online = online_replay_buffer.random_batch(args.batch_size)
            train_data_offline = offline_replay_buffer.random_batch(args.batch_size)
            train_data_meta = online_replay_buffer.meta_random_batch(args.batch_size)
            train_data_rl = priority_replay_buffer.random_batch(args.batch_size)
            train_data = dict()
            train_data["offline_observations"] = train_data_offline["observations"]
            train_data["offline_next_observations"] = train_data_offline["next_observations"]
            train_data["offline_actions"] = train_data_offline["actions"]
            train_data["offline_rewards"] = train_data_offline["rewards"]
            train_data["offline_terminals"] = train_data_offline["terminals"]

            train_data["online_observations"] = train_data_online["observations"]
            train_data["online_next_observations"] = train_data_online["next_observations"]
            train_data["online_actions"] = train_data_online["actions"]
            train_data["online_rewards"] = train_data_online["rewards"]
            train_data["online_terminals"] = train_data_online["terminals"]

            train_data["meta_observations"] = train_data_meta["observations"]

            train_data["rl_observations"] = train_data_rl["observations"]
            train_data["rl_next_observations"] = train_data_rl["next_observations"]
            train_data["rl_actions"] = train_data_rl["actions"]
            train_data["rl_rewards"] = train_data_rl["rewards"]
            train_data["rl_terminals"] = train_data_rl["terminals"]
            train_data["idxs"] = train_data_rl["idxs"]
            train_data["tree_idxs"] = train_data_rl["tree_idxs"]

            policy.train(train_data, priority_replay_buffer)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env))
            if evaluations[-1][0] > best_reward:
                best_reward = evaluations[-1][0]
            print(f"Best reward: {best_reward: .3f}")
            print("---------------------------------------")
            np.save(f"../results/{file_name}", evaluations)
            if args.save_model: policy.save(f"../models/{file_name}")
