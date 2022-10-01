import numpy as np


def get_performance(filename, fraction=[25, 30]):
    # Performance: Best & avg of last 10 evaluations
    result = np.load(filename)
    assert result.shape[0] >= 61
    performance = []
    for k in fraction:
        first_k = result[:2*k+1, 0]
        best = np.max(first_k)
        avg = np.mean(first_k[-10:])
        performance.append(best)
        performance.append(avg)
    return np.array(performance)


def get_baseline_performance(policy, env, extra_parameter=None, seed_num=5):
    overall_performance = []
    for seed in range(seed_num):
        if extra_parameter:
            filename = f"results/{policy}_{env}_{seed}_{extra_parameter}.npy"
        else:
            filename = f"results/{policy}_{env}_{seed}.npy"
        performance = get_performance(filename)
        overall_performance.append(performance)
    overall_performance = np.array(overall_performance)
    mean = np.mean(overall_performance, 0).tolist()
    std = np.std(overall_performance, 0).tolist()
    print([f"{mean[i]:.2f}"+'±'+f"{std[i]:.2f}" for i in range(len(mean))])
    return mean, std

if __name__ == '__main__':
    p = "TD3_OEME_BR"
    for i in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        for j in [1.0, 0.1]:
            get_baseline_performance(p, "walker2d-medium-v0", extra_parameter=f"{i}_{j}_False", seed_num=5)
    print("="*20)
    for i in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        for j in [1.0, 0.1]:
            get_baseline_performance(p, "walker2d-random-v0", extra_parameter=f"{i}_{j}_False", seed_num=5)
    print("="*20)
    for i in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        for j in [1.0, 0.1]:
            get_baseline_performance(p, "hopper-random-v0", extra_parameter=f"{i}_{j}_False", seed_num=3)
