**This is the codebase for our paper "Sample Efficient Offline-to-Online Reinforcement Learning" (under review).**

Our codebase is mainly built on top of the official implementations of TD3+BC (https://github.com/sfujim/TD3_BC), TD3 (https://github.com/sfujim/TD3) and BR (https://github.com/shlee94/Off2OnRL).

**Usage**

1. Pretrain agents via offline RL algorithm TD3+BC with 1M gradient steps.

   ```shell
   cd TD3_BC
   python main.py --env halfcheetah-random-v0
   ```

2. Finetune agents via different offline-to-online RL algorithms with 300K environment steps. For example, we can finetune the halfcheetah-random offline RL agents with the proposed OEMA algorithm as follows.

   ```shell
   cd TD3_OEMA
   python main.py --env halfcheetah-random-v0 --optimism_level 5.0 --beta 1.0 --max_timesteps 300000
   ```

**Requirements**

- Python 3.8
- PyTorch 1.10.0
- Gym 0.19.0
- MuJoCo 2.2.0
- mujoco-py 2.1.2.14
- d4rl
