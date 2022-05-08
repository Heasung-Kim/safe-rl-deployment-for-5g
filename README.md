# safe-rl-deployment-for-5g
CS394R final Project

## Authors
Heasung Kim and Sravan Ankireddy

## Abstract
   In this project, we consider the problem of network parameter optimization for rate maximization. We frame this as a joint optimization problem of power control, beam forming, and interference cancellation. We consider the setting where multiple Base Stations (BSs) are communicating with multiple user equipments (UEs). 
   Because of the exponential computational complexity of brute force search, we instead solve this non-convex optimization problem using deep reinforcement learning (RL) techniques. The modern communication systems are notorious for their difficulty in exactly modeling their behaviour. This limits us in using RL based algorithms as interaction with the environment is needed for the agent to explore and learn efficiently. Further, it is ill advised to deploy the algorithm in real world for exploration and learning because of the high cost of failure. In contrast to the previous RL based solutions proposed, such as deep-Q network (DQN)  based control, we propose taking an offline model based approach. We specifically consider discrete batch constrained deep Q-learning (BCQ) and show that performance similar to DQN can be acheived with only a fraction of the data and without the need for exploration. This results in maximizing sample efficiency and minimizing risk in the deployment of a new algorithm to commercial networks.



## How to run this Project?
*****
Please run the python file

    python bc_rl_main.py


*****


## How to get the results?
*****
Please see the file

    config.yaml

Your experiment name is defined by algorithm_name of the algorithm_config

    algorithm_config:
      algorithm_name: "YOUR_ENV_NAME"


Your experimental results will be generated in the following directory

    \data\results\"YOUR_ENV_NAME"


*****

## How to change the parameters?
*****
Please change the parameters in config.yaml

    config.yaml

We do not recommend you to change wireless settings.


*****

## Data Analysis
*****
You can use your newly-generated results or you can show the figures by running some plot-generators in the following directory

    \data_analysis


*****


## Results

### Performance comparison (BCMQ vs SAC vs DQN)
<img src="https://github.com/Heasung-Kim/safe-rl-deployment-for-5g/blob/main/data/figures/average_reward_vs_learning_iter.jpg" width="70%" height="70%" title="mainfig" alt="average_reward_vs_learning_iter"></img>
Learning curves of average sum of rewards over 1,000 radio frames (episodes). The width equal to the deviation from the mean is filled with the corresponding color. In order to minimize the uncertainty due to randomness and to measure the exact performance of the algorithms, the same experiment was repeated 10 times with the same set of parameters for each of the approaches.

### Learning Rate Analysis 
<img src="https://github.com/Heasung-Kim/safe-rl-deployment-for-5g/blob/main/data/figures/average_reward_vs_learning_iter_LRdiffer.jpg" width="70%" height="70%" title="mainfig" alt="average_reward_vs_learning_iter_LRdiffer"></img>
Learning curves of average sum of rewards over 1,000 radio frames (episodes). The width equal to the deviation from the mean is filled with the corresponding color. In order to minimize the uncertainty due to randomness and to measure the exact performance of the algorithms, the same experiment was repeated 10 times with the same set of parameters for each of the approaches.

### Batch size 
<img src="https://github.com/Heasung-Kim/safe-rl-deployment-for-5g/blob/main/data/figures/average_reward_vs_learning_iter_CBdiffer.jpg" width="70%" height="70%" title="mainfig" alt="average_reward_vs_learning_iter_CBdiffer"></img>
Learning curves of average sum of rewards over 1,000 radio frames (episodes). The width equal to the deviation from the mean is filled with the corresponding color. In order to minimize the uncertainty due to randomness and to measure the exact performance of the algorithms, the same experiment was repeated 10 times with the same set of parameters for each of the approaches.


### Dataset Quality 
<img src="https://github.com/Heasung-Kim/safe-rl-deployment-for-5g/blob/main/data/figures/average_reward_vs_learning_iter_unifandbiased_caption.jpg" width="70%" height="70%" title="mainfig" alt="average_reward_vs_learning_iter_unifandbiased_caption"></img>
Learning curves of average sum of rewards over 1,000 radio frames (episodes). The width equal to the deviation from the mean is filled with the corresponding color. In order to minimize the uncertainty due to randomness and to measure the exact performance of the algorithms, the same experiment was repeated 10 times with the same set of parameters for each of the approaches.


## Copyrights
The following files are written based on the "TF2RL" codes [1] for fair performance comparison. If you want to get the experiment results of the SAC and the DQN, you need to install the "TF2RL" library and run run_DQN.py and SAC_discrete.py.

    /trainers
    /run_DQN.py
    /SAC_discrete.py

Our main algorithm is based on the Batch Constrained Q-Learning [2,3] and https://github.com/sfujim/BCQ.

    /BCMQ.py
    
The 5G Network Wireless Settings are followed by [4] for the fair performance comparison. All the codes in the following directory is borrowed from [4] and https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks.

    /env/channel

All other code is copyrighted by the authors of this repository.

## References
[1] K. Ota, “Tf2rl,” https://github.com/keiohta/tf2rl/, 2020. \
[2] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in International Conference on Machine Learning. PMLR, 2019, pp. 2052–2062.\
[3] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approxi- mation error in actor-critic methods,” in International conference on machine learning. PMLR, 2018, pp. 1587–1596.\
[4] F. B. Mismar, B. L. Evans, and A. Alkhateeb, “Deep reinforcement learning for 5g networks: Joint beamforming, power control, and inter- ference coordination,” IEEE Transactions on Communications, vol. 68, no. 3, pp. 1581–1592, 2019.
