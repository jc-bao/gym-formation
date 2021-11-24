# Formation environment based on MPE

multi-agent formation control environment implemented with MPE.

## Installation

```
git clone https://github.com/jc-bao/gym-formation.git
cd gym-formation
pip install -e .
```

## Test
```
python test.py -s formation_hd_env --num-layer 1
```
Note: use `-r` flag to use random policy.

## TODO

- [ ] Observation: reduce the number of observation in hierarchy policy. (now use fully observation)

- [ ] Leader&Communication: choose the group leader in each layer smartly and communicate smartly. (now use the first agent as leader)

- [ ] Target shape: achieve asymmetic shape. (now only symetric shape in higher level control)

- [ ] Group: divide the group smartly to reduce formation time. (now the group was previously divided, more layers, less distributional)

## Extend to more agent use hierarchy policy

```python
num_agents_per_layer = 3 # number of agents of your original policy network (or you can use ezpolicy provided by the package)
num_layer = 2 # number of control layer, extend agent number to n^{layers}
env = formation_gym.make_env('formation_hd_env', benchmark=False, num_agents = anum_agents_per_layer**num_layer)
obs_n = env.reset()
while True:
  		# use BFS to extend your policy to larger scale
      act_n = formation_gym.get_action_BFS(YOUR_POLICY_HERE, obs_n, num_agents_per_layer)
      # step environment
      obs_n, reward_n, done_n, _ = env.step(act_n)
      ...
```

Note: 

* not recommend to use layer larger than 5, which will run 3^5 network in parallel. 
* make sure your policy network can correct turn the observation of single agent into action.
* the `get_action_BFS` is based on [Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search)
* **get any target shape**: by using the function provided in env by calling `ideal_shape = env.generate_shape(num_layers = 3, layer_shapes = YOUR_TARGET_LAYER_SHAPE).reshape(-1,2)`  and replace the observation counter part with it. 

## Train

Please Refer to `train/README.md`
If you want to use another algorithm, here is the template:

```
import formation_gym

env = formation_gym.make_env(your_scenario_name, if_use_benchmark, number_of_agents, episode_length)
```

## Scenarios

| basic_formation_env                                          | formation_hd_env                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The reimplemtation for OpenAI MPE spread enviroment. The target it reach the landmark. | Try to mimic the topology of landmarks only with relative observation. |
|                                                              | ![Nov-24-2021 14-10-59](https://tva1.sinaimg.cn/large/008i3skNly1gwq7m2aj1pg30ii0i0e82.gif) |
| ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfvhkxraj60hs0dcaal02.jpg) | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfuj9pr7j60hs0dc3yz02.jpg) |

## Further information

```
action space = [if_moveable, action_1, ... action_n,  comm_1, ... comm_n]
```

### MVE Support

* Action: ``
