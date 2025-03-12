import numpy as np
import random
import math

class brain:
    def __init__(self, nodes):
        self.weights = [random.uniform(-1, 1) for _ in range(nodes+1)]
        self.last_return = 0        

    def think(self, input_data):
        """Decide action based on input and past experiences"""
        total = 0
        input_data.append(1)
        for i in range(len(self.weights)):
            total += self.weights[i] * input_data[i]
        self.last_return = self.sigmoid(total)
        return self.last_return

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


class agent:
    def __init__(self, nodes, mutation_rate = 0.1, parent = None):
        self.brain = brain(nodes)
        self.score = 0
        if parent:
            self.brain.weights = [parent.brain.weights[i] + (random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.02 else 0) for i in range(len(self.brain.weights))]

    def think(self, input_data):
        return self.brain.think(input_data)


class neural_network:
    def __init__(self, max_population = 50, keep_best = 10, nodes = 4, mutation_rate = 0.1):
        self.agents = []
        self.max_population = max_population
        self.keep_best = keep_best
        self.nodes = nodes
        self.mutation_rate = mutation_rate
        self.generation = 0
    
    def populate(self):
        for i in range(self.max_population):
            self.agents.append(agent(self.nodes))

    def next_generation(self):
        self.agents.sort(key=lambda x: x.score, reverse=True)
        best_agents = self.agents[:self.keep_best]
        new_agents = []
        # Keep the best agents
        for best_agent in best_agents:
            new_agent = agent(self.nodes)
            new_agent.brain.weights = best_agent.brain.weights.copy()
            new_agents.append(new_agent)
        
        for i in range(self.max_population - self.keep_best):
            parent = random.choice(best_agents)
            new_agent = agent(self.nodes, self.mutation_rate, parent)
            new_agents.append(new_agent)
        
        print("Generation: ", self.generation)
        print("Mutation Rate: ", self.mutation_rate)
        print(new_agents[0].brain.weights)
        print(new_agents[1].brain.weights)
        
        self.agents = new_agents
        self.generation += 1


