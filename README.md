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

'''
action space = [if_moveable, action_1, ... action_n,  comm_1, ... comm_n]
'''