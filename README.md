# Formation environment based on MPE

multi-agent formation control environment implemented with MPE.

## Installation

```
git clone https://github.com/jc-bao/gym-formation.git
cd gym-formation
pip install -e .
```

## Testing
```
python interactive.py -s formation_hd_env  
```

## Settings

| basic_formation_env                                          | formation_hd_env                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The reimplemtation for OpenAI MPE spread enviroment. The target it reach the landmark. | Try to mimic the topology of landmarks only with relative observation. |
| ![](https://tva1.sinaimg.cn/large/008i3skNly1gukg537oljj608w08yq2v02.jpg)     | ![Large GIF (702x582)](https://tva1.sinaimg.cn/large/008i3skNly1gukfsomxebg60ji0g6to302.gif) |
| ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfvhkxraj60hs0dcaal02.jpg) | ![plt](https://tva1.sinaimg.cn/large/008i3skNly1gukfuj9pr7j60hs0dc3yz02.jpg) |



```
action space = [if_moveable, action_1, ... action_n,  comm_1, ... comm_n]
```
