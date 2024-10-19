from collections import deque

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
