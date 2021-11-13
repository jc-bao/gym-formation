# Training Environment for formation_gym

## MADDPG

### MADDPG-v1

* Overview: This is a basic MADDPG Algorithm implemented with pytorch. Only MADDPG, without any trick.
  * Parameters: tuned
  
* Pros: 
    * easy to use
    * perform well on easy tasks (training 3 agents formation task only takes 5h on MacBook-pro-2018-13')
    
* Cons: 
    * slow training speed in large number cases and no parallel support (training 9 agents formation task takes 25h on MacBook-pro-2018-13' and cannot fully use the hardware)
    * no gpu support.
    * no normalization, prior replay etc.
    
* Use Cases
  * Train

     ```
     python main.py --scenario-name=formation_hd_env --num-agents 5 --save-dir model_hd_5 --time-steps 2000000 --max-episode-len 30
     # training formation with 6 agents
     
     python main.py --scenario-name=formation_hd_env --num-agents 6 --save-dir model_hd_6 --time-steps 2000000 --max-episode-len 40 --time-steps 2000000
     
     python main.py --scenario-name=formation_hd_env --num-agents 7 --save-dir model_hd_7 --time-steps 3000000 --max-episode-len 50
     
     python main.py --scenario-name=formation_hd_env --num-agents 8 --save-dir model_hd_8 --time-steps 4000000 --max-episode-len 50
     
     python main.py --scenario-name=formation_hd_env --num-agents 9 --save-dir model_hd_9 --time-steps 4000000 --max-episode-len 50
     
     
     
     # training formation with 4 agents
     python main.py --scenario-name=formation_hd_env --num-agents 4 --save-dir model_hd_4
     
     # training formation with partial observation (4 agents)
     python main.py --scenario-name=formation_hd_partial_env --num-agents 5 --save-dir model_hd_par_5 --time-steps 3000000 --max-episode-len 25 --evaluate-episode-len 25
     
     # training formation with obstacles (4 agents)
     python main.py --scenario-name=formation_hd_obs_env --num-agents 4 --save-dir model_hd_obs_4 
     ```
  
  * Evaluate 
  
     ```
     python main.py --scenario-name=formation_hd_env --num-agents 4 --save-dir model_hd_4 --model-idx 12 --evaluate True
     
     python main.py --scenario-name=formation_hd_partial_env --num-agents 4 --save-dir model_hd_par_4 --model-idx 12 --evaluate True
     
     python main.py --scenario-name=formation_hd_obs_env --num-agents 4 --save-dir model_hd_obs_4 --max-episode-len 50 --evaluate-episode-len 50 --evaluate True --model-idx 99
     ```
  
* Result: 

    | --scenario-name=formation_hd_env --num-agents 3              | --scenario-name=formation_hd_env --num-agents 4              | --scenario-name=formation_hd_partial_env --num-agents 4 --save-dir model_hd_par_4 |
    | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | ![Sep-23-2021 13-42-30](https://tva1.sinaimg.cn/large/008i3skNly1guqidc3ox4g60iu0j07c702.gif) | ![002](https://tva1.sinaimg.cn/large/008i3skNly1guqiapgnyrg60iu0j012k02.gif) | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumq7zonu8g60iu0j0wkq02.gif) |
    | ![image-20210923134622867](https://tva1.sinaimg.cn/large/008i3skNly1guqih3g8tuj60y80p2gmw02.jpg) | ![image-20210924105854969](https://tva1.sinaimg.cn/large/008i3skNly1gurj90uqkrj60xm0nkmyh02.jpg) | Setting1: observe the most colse agent, unobservable agent set to 0<br />![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumq89gyv6j60hs0dcmy002.jpg)<br />Setting2: observe fixed 2 agent<br />![image-20210924105614202](https://tva1.sinaimg.cn/large/008i3skNly1gurj69y3rsj60m00famy802.jpg)<br /> |
    | **--scenario-name=formation_hd_obs_env --num-agents 4 --save-dir model_hd_obs_4 --max-episode-len 50 --evaluate-episode-len 50** | --scenario-name=formation_hd_env --num-agents 6              |                                                              |
    | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumqagt0mmg60iu0j07e502.gif) | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumq7zonu8g60iu0j0wkq02.gif) |                                                              |
    | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumq6thy8xj60hs0dc0ti02.jpg)Only learning how to escape [Try smaller and more obstacles] | ![image-20210924105744094](https://tva1.sinaimg.cn/large/008i3skNly1gurj7u57zxj60ym0nkmyp02.jpg) |                                                              |

    

### MADDPG-v2 [Not Converge]

* Overview: This is a classic MADDPG Algorithm implemented with pytorch. Can use **VectorEnv**.

  * Parameters: tuned

* Pros: 

  * VectorEnv Support.

* Cons: 

  * No GPU support. (USE_CUDA will cause some unknown issues.)
  * Still cannot converge in large agent number scenario.

* Use Cases
  * Train `python main.py --env_id=formation_hd_env --agent-num 9`
  * Evaluate `python evaluate.py --env_id=formation_hd_env --agent-num 9 --run_num 11`
  
* Result: 

  | Formation_hd_env                                             | Formation_hd_env                                             |      |      |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ---- |
  | Agent_num = 9 run=11                                         | Agent_num = 3 run=16                                         |      |      |
  | ![image-20210919085205093](https://tva1.sinaimg.cn/large/008i3skNly1gulnhjn8xaj60fu098jrw02.jpg)<br />Not converge. | ![image-20210919100252525](https://tva1.sinaimg.cn/large/008i3skNly1gulpj7ad8sj60go0900tj02.jpg)<br />Not converge. |      |      |

### MADDPG-v3 [Not converge]

* Overview: This is a parallel MADDPG Algorithm implemented with tensorflow based on **RAY** framework. Please install rllib before use it. 

    * Parameters: tuned

* Pros: 

  * Parallel Support.

* Cons: 

    * Still cannot converge in large agent number scenario.

* Use Cases

  * Train `python main.py --scenario=formation_hd_env --agent-num 9`

    ```
    # train simple spread
    python main.py --scenario simple_spread --num-gpus 1 --num-workers 4 --num-envs-per-worker 4  --sample-batch-size 6
    # train formation with 4 agents
    CUDA_VISIBLE_DEVICES=0 python main.py --scenario=formation_hd_boundless_env --num-agents 4 --num-workers 4 --num-envs-per-worker 4  --sample-batch-size 25
    python main.py --scenario=formation_hd_env --num-agents 4 --num-gpus 1 --num-workers 4 --num-envs-per-worker 4  --sample-batch-size 6
    python main.py --scenario=formation_hd_env --num-agents 4 --num-gpus 1 --num-workers 8 --num-envs-per-worker 4  --sample-batch-size 4
    
    ```

    

  * Evaluate `rllib rollout \
        '/Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v3/ray_results/MADDPG_RLLib/MADDPG_mpe_72161_00000_0_2021-09-23_19-43-18/params.pkl' \
        --run coontrib/MADDPG --env simple_spread --steps 100`

* Result: 

    | Formation_hd_env                                             |      |      |      |
    | ------------------------------------------------------------ | ---- | ---- | ---- |
    | Agent_num = 9 run=11                                         |      |      |      |
    | ![image-20210919113901134](https://tva1.sinaimg.cn/large/008i3skNly1gulsb77p9tj60kz07rwf302.jpg)<br />[Not converge] |      |      |      |

    

### MADDPG-v5

* Overview: use off-policy open-source framework

* Pros:

* Cons:

* Tensorboard information:

  * Train `simple_spread`

  | Actor_grad_norm: gradient norm for actor update              | Individual reward: one agent's reward                        | Average Episode Reward                                       |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20210926135104951](https://tva1.sinaimg.cn/large/008i3skNly1gutzgt9k0zj607z05nwek02.jpg) | ![image-20210926135140704](https://tva1.sinaimg.cn/large/008i3skNly1gutzhdw105j607t05mjrg02.jpg) | ![image-20210926135234454](https://tva1.sinaimg.cn/large/008i3skNly1gutzibneuoj608005qmxc02.jpg) |
  | The norm for critic to update                                | dist_entropy: action entropies (more high means more chance to explore) | policy loss: TD Error * Action ratio                         |
  | ![image-20210926135258174](https://tva1.sinaimg.cn/large/008i3skNly1gutziq3dwvj608905n0sq02.jpg) | ![image-20210926135827132](https://tva1.sinaimg.cn/large/008i3skNly1gutzofosd9j607x05iwek02.jpg) | ![image-20210926143335221](https://tva1.sinaimg.cn/large/008i3skNly1guu0ozx87ij608205naa902.jpg) |
  | Ratio: difference bewteen two actions                        | value loss: loss of value function(TD Error)                 |                                                              |
  | ![image-20210926143353065](https://tva1.sinaimg.cn/large/008i3skNly1guu0pb4d9lj607s05i3yh02.jpg) | ![image-20210926143405114](https://tva1.sinaimg.cn/large/008i3skNly1guu0piruodj607i05ojrg02.jpg) |                                                              |

  

* Use cases:

  * Train

    ```
    # formation HD with 2 agents
    python train.py --env_name formation --algorithm_name maddpg --experiment_name hd_2 --scenario_name formation_hd_boundless_env --num_agents 2 --n_rollout_threads 32 --n_rollout_threads 32 --episode_length 25 --lr 7e-4 --update_interval 1  # 32(run11) 8(run12) 1(run10)
    # formation HD with 4 agents
    CUDA_VISIBLE_DEVICES=7 python train.py --env_name formation --algorithm_name maddpg --experiment_name hd_4 --scenario_name formation_hd_env --num_agents 4 --n_rollout_threads 128
    # formation HD with 9 agents
    CUDA_VISIBLE_DEVICES=0 python train.py --env_name formation --algorithm_name maddpg --experiment_name hd_9 --scenario_name formation_hd_env --num_agents 9 --n_rollout_threads 200 --num_env_steps 20000000 --buffer_size 20000 --share_policy False --layer_N 2 --batch_size 128 --train_interval 12800
    # formation obstacles with 4 agents
    CUDA_VISIBLE_DEVICES=4 python train.py --env_name formation --algorithm_name maddpg --experiment_name obs_4 --scenario_name formation_hd_obs_env --num_agents 4 --n_rollout_threads 128
    # formation partial observation with 4 agents
    CUDA_VISIBLE_DEVICES=6 python train.py --env_name formation --algorithm_name maddpg --experiment_name obs_4 --scenario_name formation_hd_partial_env --num_agents 4 --n_rollout_threads 128
    ```

  * render

    ```
    # formation HD with 2 agents
    python render.py --env_name formation --algorithm_name maddpg --experiment_name hd_2 --scenario_name formation_hd_env --num_agents 2 --n_rollout_threads 8 --episode_length 25 --model_dir /Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v5/results/formation_hd_env/maddpg/hd_2/run1/models/
    # formation HD with 4 agents
    python render.py --env_name formation --algorithm_name maddpg --experiment_name hd_4 --scenario_name formation_hd_env --num_agents 4 --buffer_size 1 --num_random_episodes 0 --model_dir /Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v5/results/formation_hd_env/maddpg/hd_4/run9/models/
    # formation HD with 9 agents
    python render.py --env_name formation --algorithm_name maddpg --experiment_name hd_9 --scenario_name formation_hd_env --num_agents 9 --buffer_size 10 --n_rollout_threads 1 --model_dir /Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v5/results/formation_hd_env/maddpg/hd_9/run1/models/
    # formation obstacles with 4 agents
    python render.py --env_name formation --algorithm_name maddpg --experiment_name obs_4 --scenario_name formation_hd_obs_env --num_agents 4 --n_rollout_threads 1 --model_dir /Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v5/results/formation_hd_obs_env/maddpg/obs_4/run3/models/
    
    python render.py --env_name formation --algorithm_name rmaddpg --experiment_name hd_5 --scenario_name formation_hd_env --num_agents 5 --n_rollout_threads 1 --episode_length 25 --model_dir /Users/reedpan/Desktop/Research/gym_formation/train/maddpg-v5/results/formation_hd_env/rmaddpg/hd_5/run1/models/
    ```

* Issues:

  ```
  /mnt/ssd_raid0/home/pancy/Documents/gym-formation/formation_gym/core.py:311: RuntimeWarning: invalid value encountered in true_divide
    force = self.contact_force * delta_pos / dist * penetration
  average_episode_rewards: -90.15602982652563 final_step_rewards: nan
  ```

  ```
  /Users/reedpan/opt/anaconda3/envs/rl/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
    return _methods._mean(a, axis=axis, dtype=dtype,
  /Users/reedpan/opt/anaconda3/envs/rl/lib/python3.9/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
    ret = ret.dtype.type(ret / rcount)
  eval_final_episode_rewards is nan
  ```

  

## [MAPPO](https://github.com/zoeyuchao/mappo)

* Overview: MAPPO algorithm which support parallel.

* Pros:

* Cons:

* Note: remember add `nn.parameter.Paramter()` to convert varibles if use torch 1.9

* Use Cases:
  * Train `./train_formation.sh`

* Issues

  ```
  on-policy/onpolicy/algorithms/utils/distributions.py", line 68, in forward
      return FixedCategorical(logits=x)
  ValueError: The parameter logits has invalid values
  ```

  ```
  formation_gym/core.py:311: RuntimeWarning: invalid value encountered in true_divide
    force = self.contact_force * delta_pos / dist * penetration
  ```

  

* Result:

<iframe src="https://wandb.ai/jc-bao/MPE/reports/Train-Formation-HD-with-MAPPO--VmlldzoxMjE5NTE2" style="border:none;height:1024px;width:100%"/>
  
  
