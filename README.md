**This is the codebase for our paper "Sample Efficient Offline-to-Online Reinforcement Learning" (TKDE 2023).**

## OEMA

Our codebase is mainly built on top of the official implementations of TD3+BC (https://github.com/sfujim/TD3_BC), TD3 (https://github.com/sfujim/TD3) and BR (https://github.com/shlee94/Off2OnRL).

### Usage

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

## Hyper-parameter Setting

| D4RL-v0                   | Optimism Level | Meta Weight |
| :------------------------- | :--------------: | :-----------: |
| halfcheetah-random        | 5.0            | 1.0         |
| halfcheetah-medium        | 5.0            | 1.0         |
| halfcheetah-medium-replay | 5.0            | 1.0         |
| halfcheetah-medium-expert | 5.0            | 1.0         |
| hopper-random             | 5.0            | 1.0         |
| hopper-medium             | 5.0            | 1.0         |
| hopper-medium-replay      | 2.0            | 1.0         |
| hopper-medium-expert      | 2.0            | 1.0         |
| walker2d-random           | 8.0            | 1.0         |
| walker2d-medium           | 1.0            | 1.0         |
| walker2d-medium-replay    | 1.0            | 0.1         |
| walker2d-medium-expert    | 1.0            | 1.0         |


### Requirements

- Python 3.8
- PyTorch 1.10.0
- Gym 0.19.0
- MuJoCo 2.2.0
- mujoco-py 2.1.2.14
- d4rl

## Cite
Please cite our work if you find it useful:
```
@article{OEMA,
  title={Sample Efficient Offline-to-Online Reinforcement Learning},
  author={Guo, Siyuan and Zou, Lixin and Chen, Hechang and Qu, Bohao and Chi, Haotian and Philip, S Yu and Chang, Yi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```

