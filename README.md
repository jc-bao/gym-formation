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
python interactive.py -s formation_hd_env -d
```
Note: -d means show demo, you can control with your keyboard without -d flag

## Train

Please Refer to `train/README.md`
If you want to use another algorithm, here is the template:
```
import formation_gym

env = formation_gym.make_env(your_scenario_name, if_use_benchmark, number_of_agents, episode_length)
```

## Settings

| basic_formation_env                                          | formation_hd_env                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The reimplemtation for OpenAI MPE spread enviroment. The target it reach the landmark. | Try to mimic the topology of landmarks only with relative observation. |
| ![](https://tva1.sinaimg.cn/large/008i3skNly1gukg5r99sij606105sjr602.jpg)     | ![Large GIF (702x582)](https://tva1.sinaimg.cn/large/008i3skNly1gukfsomxebg60ji0g6to302.gif) |
| ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfvhkxraj60hs0dcaal02.jpg) | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfuj9pr7j60hs0dc3yz02.jpg) |

## Further information

```
action space = [if_moveable, action_1, ... action_n,  comm_1, ... comm_n]
```

### MVE Support

* Action: ``
