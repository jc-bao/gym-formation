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
     # training formation with 9 agents
     python main.py --scenario-name=formation_hd_env --num-agents 9 --save-dir model_hd_9
     # training formation with 4 agents
     python main.py --scenario-name=formation_hd_env --num-agents 4 --save-dir model_hd_4
     # training formation with partial observation (4 agents)
     python main.py --scenario-name=formation_hd_partial_env --num-agents 4 --save-dir model_hd_par_4
     # training formation with obstacles (4 agents)
     python main.py --scenario-name=formation_hd_obs_env --num-agents 4 --save-dir model_hd_obs_4 --max-episode-len 50 --evaluate-episode-len 50
     ```
  
  * Evaluate 
  
     ```
     python main.py --scenario-name=formation_hd_env --num-agents 4 --save-dir model_hd_4 --model-idx 12 --evaluate True
     
     python main.py --scenario-name=formation_hd_partial_env --num-agents 4 --save-dir model_hd_par_4 --model-idx 12 --evaluate True
     ```
  
* Result: 

    | --scenario-name=formation_hd_env --num-agents 3              | --scenario-name=formation_hd_env --num-agents 4              | --scenario-name=formation_hd_partial_env --num-agents 4 --save-dir model_hd_par_4 |
    | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumpqowoswg60iu0j0gqv02.gif) | ![Large GIF (678x684)](/Users/reedpan/Downloads/Sep-20-2021 10-40-35/Large GIF (678x684).gif) | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumq7zonu8g60iu0j0wkq02.gif) |
    | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumpsfau3ij60hs0dcdgb02.jpg) | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumpswmsnuj60hs0dc0td02.jpg) | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumq89gyv6j60hs0dcmy002.jpg) |
    | **--scenario-name=formation_hd_obs_env --num-agents 4 --save-dir model_hd_obs_4 --max-episode-len 50 --evaluate-episode-len 50** |                                                              |                                                              |
    | ![Large GIF (678x684)](https://tva1.sinaimg.cn/large/008i3skNly1gumqagt0mmg60iu0j07e502.gif) |                                                              |                                                              |
    | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gumq6thy8xj60hs0dc0ti02.jpg)Only learning how to escape [Try smaller and more obstacles] |                                                              |                                                              |

    

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
  * Evaluate `python evaluate.py --env_id=formation_hd_env --agent-num 9 --run_num 11`

* Result: 

    | Formation_hd_env                                             |      |      |      |
    | ------------------------------------------------------------ | ---- | ---- | ---- |
    | Agent_num = 9 run=11                                         |      |      |      |
    | ![image-20210919113901134](https://tva1.sinaimg.cn/large/008i3skNly1gulsb77p9tj60kz07rwf302.jpg)<br />[Not converge] |      |      |      |

    

### MADDPG-v5

* Overview: use off-policy open-source framework

* Pros:

* Cons:

* Use cases:

  * Train

    ```
    # formation HD with 4 agents
    python train.py --env_name formation --algorithm_name maddpg --experiment_name hd_4 --scenario_name formation_hd_env --num_agents 4 --n_rollout_threads 128
    ```

    

## MAPPO[Not converge]

* Overview: MAPPO algorithm which support parallel.
* Pros:
* Cons:
* Note: remember add `nn.parameter.Paramter()` to convert varibles if use torch 1.9
* Use Cases:
  * Train `./train_formation.sh`

* Result:

  | Formation_hd_env                                             |      |      |      |
  | ------------------------------------------------------------ | ---- | ---- | ---- |
  | Agent_num = 9 run=11                                         |      |      |      |
  | ![image-20210919113701695](https://tva1.sinaimg.cn/large/008i3skNly1guls972v8rj60jo05jaa802.jpg) |      |      |      |

  
