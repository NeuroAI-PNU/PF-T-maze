# main.py

# load library
import datetime
import numpy as np
from tqdm import tqdm
import pickle

# custom library
from library.training import training_2D
from library.agents import PRAgent

# file manage
save_dir = "./simulated_results/"
today = datetime.datetime.now().strftime("%Y-%m-%d")
obs_mode = "noisy_onehot"
# parameter for environment
grid_size = 9

print(
    "You are testing on "
    + str(grid_size)
    + " x "
    + str(grid_size)
    + " size grid world."
)
# parameter for agent
gamma = 0.95
alpha_r_l = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha_w = 0.1
epsilon = 0.8
explora = True
total_trials = 100
noise_level = 0.05
episode = 100
max_step_length = 500
lambda_l = [0, 0.2, 0.4, 0.6, 0.8]
weight_init = "random"
goal_switch = 20

agent = PRAgent
segment = [x for x in range(0, episode + 1, goal_switch)]


exp = {}

for lambda_ in tqdm(lambda_l, desc="lambda"):
    for alpha_r in tqdm(alpha_r_l, desc="alpha_r"):
        for trial in tqdm(range(total_trials), desc="trial of PF"):
            step_lengths, _, _, rewards, adaptation_history = training_2D(
                agent_=agent,
                episodes=episode,
                alpha_r=alpha_r,
                alpha_w=alpha_w,
                gamma=gamma,
                lambda_=lambda_,
                max_step_length=max_step_length,
                grid_size=grid_size,
                obs_mode=obs_mode,
                noise_level=0.05,
                weight_init=weight_init,
                epsilon_dic=epsilon,
                explora=explora,
                goal_switch=goal_switch,
            )
            adaptation_history = np.array(adaptation_history)
            adaptation_rate_l = []
            adap_step_length_l = []
            for i in range(len(segment) - 1):
                adaptation_rate = (
                    np.argmax(adaptation_history[segment[i] : segment[i + 1]]) + 1
                )
                if adaptation_rate == 1:
                    adaptation_rate = 20
                adaptation_rate_l.append(adaptation_rate)
                adap_step_length_l.append(
                    np.sum(step_lengths[segment[i] : segment[i] + adaptation_rate])
                )

            if trial == 0:
                trials_step_lengths = [step_lengths]
                trials_rewards = [rewards]
                trials_adaptation_rate = [adaptation_rate_l]
                trials_adap_step_length = [adap_step_length_l]
            else:
                trials_step_lengths.append(step_lengths)
                trials_rewards.append(rewards)
                trials_adaptation_rate.append(adaptation_rate_l)
                trials_adap_step_length.append(adap_step_length_l)

        mean_step_lengths = np.mean(np.array(trials_step_lengths), axis=0)
        sem_step_lengths = np.std(
            np.array(trials_step_lengths), axis=0, ddof=1
        ) / np.sqrt(total_trials)
        mean_rewards = np.mean(np.cumsum(np.array(trials_rewards), axis=1), axis=0)
        sem_rewards = np.std(
            np.cumsum(np.array(trials_rewards), axis=1), axis=0, ddof=1
        ) / np.sqrt(total_trials)
        last_cum_rewards = np.cumsum(np.array(trials_rewards), axis=1)[:, -1]
        mean_epi_step = np.mean(np.array(trials_step_lengths), axis=1)
        adaptation_rate = np.array(trials_adaptation_rate)
        # sem_adaptation_rate = np.std(np.array(trials_adaptation_rate), axis=0, ddof = 1)/np.sqrt(total_trials)
        adap_step_length = np.array(trials_adap_step_length)
        # sem_adap_step_length = np.std(np.array(trials_adap_step_length), axis=0, ddof = 1)/np.sqrt(total_trials)

        PF_e_exp = {
            "lambda": lambda_,
            "alpha_r": alpha_r,
            "mean of steps": mean_step_lengths,
            "sem of steps": sem_step_lengths,
            "mean of cum rewards": mean_rewards,
            "sem of cum rewards": sem_rewards,
            "last cum rewards": last_cum_rewards,
            "mean of epi steps": mean_epi_step,
            "adaptation rate": adaptation_rate,
            #'sem of adaptation rate': sem_adaptation_rate,
            "adap step length": adap_step_length,
        }
        #'sem of adap step length': sem_adap_step_length}

        exp.update({"alpha_" + str(alpha_r) + "_lambda_" + str(lambda_): PF_e_exp})


save_file_name = today + "_2Dim" + obs_mode + "Maze_" + str(grid_size) + "state.pkl"

with open(save_dir + save_file_name, "wb") as f:
    pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)

print("\n")

if __name__ == "__main__":
    pass
