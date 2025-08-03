import logging
import random
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MEMORY_SIZE = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
EXPERIENCE_REPLAY_ALPHA = 0.6
EXPERIENCE_REPLAY_BETA = 0.4

class MemoryType(Enum):
    """Enum for memory types"""
    REPLAY = 1
    PRIORITIZED_REPLAY = 2

class Memory(ABC):
    """Abstract base class for memory"""
    def __init__(self, memory_type: MemoryType, capacity: int):
        self.memory_type = memory_type
        self.capacity = capacity
        self.memory = []
        self.lock = Lock()

    @abstractmethod
    def add_experience(self, experience: Dict):
        """Add experience to memory"""
        pass

    @abstractmethod
    def sample_batch(self) -> List[Dict]:
        """Sample batch from memory"""
        pass

class ReplayMemory(Memory):
    """Replay memory implementation"""
    def __init__(self, capacity: int):
        super().__init__(MemoryType.REPLAY, capacity)

    def add_experience(self, experience: Dict):
        with self.lock:
            self.memory.append(experience)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)

    def sample_batch(self) -> List[Dict]:
        with self.lock:
            batch = random.sample(self.memory, BATCH_SIZE)
            return batch

class PrioritizedReplayMemory(Memory):
    """Prioritized replay memory implementation"""
    def __init__(self, capacity: int):
        super().__init__(MemoryType.PRIORITIZED_REPLAY, capacity)
        self.priorities = deque(maxlen=capacity)

    def add_experience(self, experience: Dict):
        with self.lock:
            self.memory.append(experience)
            self.priorities.append(1.0)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)
                self.priorities.popleft()

    def sample_batch(self) -> List[Tuple[Dict, float]]:
        with self.lock:
            batch = []
            for _ in range(BATCH_SIZE):
                priority = random.random()
                idx = np.argmax([p >= priority for p in self.priorities])
                batch.append((self.memory[idx], self.priorities[idx]))
                self.priorities[idx] *= (1 - EXPERIENCE_REPLAY_ALPHA + EXPERIENCE_REPLAY_BETA * np.random.rand())
            return batch

class ExperienceReplay:
    """Experience replay class"""
    def __init__(self, memory_type: MemoryType, capacity: int):
        self.memory = memory_type(capacity)
        self.memory_type = memory_type

    def add_experience(self, experience: Dict):
        self.memory.add_experience(experience)

    def sample_batch(self) -> List[Dict]:
        return self.memory.sample_batch()

class ExperienceReplayBuffer:
    """Experience replay buffer class"""
    def __init__(self, memory_type: MemoryType, capacity: int):
        self.memory_type = memory_type
        self.capacity = capacity
        self.experience_replay = ExperienceReplay(memory_type, capacity)
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, experience: Dict):
        self.buffer.append(experience)
        self.experience_replay.add_experience(experience)

    def sample_batch(self) -> List[Dict]:
        return self.experience_replay.sample_batch()

    def get_buffer(self) -> List[Dict]:
        return list(self.buffer)

class ExperienceReplayAgent:
    """Experience replay agent class"""
    def __init__(self, memory_type: MemoryType, capacity: int):
        self.memory_type = memory_type
        self.capacity = capacity
        self.experience_replay_buffer = ExperienceReplayBuffer(memory_type, capacity)
        self.experience_replay = ExperienceReplay(memory_type, capacity)

    def add_experience(self, experience: Dict):
        self.experience_replay_buffer.add_experience(experience)

    def sample_batch(self) -> List[Dict]:
        return self.experience_replay.sample_batch()

    def get_buffer(self) -> List[Dict]:
        return self.experience_replay_buffer.get_buffer()

# Example usage
if __name__ == "__main__":
    memory_type = MemoryType.REPLAY
    capacity = MEMORY_SIZE
    agent = ExperienceReplayAgent(memory_type, capacity)

    # Add experiences to memory
    for _ in range(1000):
        experience = {"state": np.random.rand(4), "action": np.random.rand(1), "reward": np.random.rand(1), "next_state": np.random.rand(4), "done": False}
        agent.add_experience(experience)

    # Sample batch from memory
    batch = agent.sample_batch()
    for experience in batch:
        logger.info(experience)