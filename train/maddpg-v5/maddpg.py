class MADDPG:
    def __init__(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError

    def _select_action(self):
        raise NotImplementedError

    def _update_network(self):
        raise NotImplementedError

    def _update_target_network(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError