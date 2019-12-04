import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
	"""Class to store memories
	
	The following values need to be stored:
		('state', 'action', 'next_state', 'reward', 'done')
	"""

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.ready = False

	def push(self, *args):
		"""Saves a transition."""
		self.ready = True
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		if(len(self.memory) < batch_size):
			return random.sample(self.memory, len(self.memory)), len(self.memory)
		return random.sample(self.memory, batch_size), batch_size
	
	def is_ready(self):
		return len(self.memory)>2

	def __len__(self):
		return len(self.memory)