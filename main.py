import matplotlib.pyplot as plt
import numpy as np
import random
from math import exp
from Particle import Particle
from Fitness import Data
import math
no_particles = 10   # number of particles
no_epochs = 100     # number of epochs
c1 = .5     # cognitive part constant
c2 = .5     # social part constant
particles = []  # store objects of each paricle
vmin,vmax = -4,4    # min and max velocity of particle
w = .9  # inertia of particle
fitness_overall = []
pos_overall = []


def run(obj,dim,algo=1):
    """
    obj : (Data) used for data manipulation that is an object of Data
    dim : (int) the dimensionality of the dataset 
    algo : (int) 1 --> DecisionTree
                 2 --> Navie Bayes
                 3 --> KNN
    """
    for epoch in range(0,no_epochs):
        if epoch == 0:
            initialize(obj,dim,algo)
            best_fitness = fitness_overall[-1]
            gbest_pos = pos_overall[-1]
            print("Iteration : ",epoch+1," -> ",best_fitness,end='\t')
            print(gbest_pos)
        else:
            
            cal(obj,dim,algo,best_fitness,gbest_pos)
            best_fitness = fitness_overall[-1]
            gbest_pos = pos_overall[-1]
            print("Iteration : ",epoch+1," -> ",best_fitness,end='\t')
            print(gbest_pos)

    x = np.arange(0,no_epochs)
    y = np.array(fitness_overall)
    #plt.scatter(x,y)
    plt.ylim((50,110))
    plt.plot(x,y)
    plt.show()


def initialize(obj,dim,algo):
    fitness = []
    for i in range(0,no_particles):
        p =Particle()   # Object for the Particle Class
        particles.append(p) # Add the particle object to the :code:`particles` list
        p._pos = np.random.randint(2,size=dim)    # create of particles and dimension
        p._velocity = np.random.uniform(-1,1,size=dim)  # create a velocity for all the particle
        p._fitness = obj.getAccuracy(obj.data_cleaning(p._pos),algo) # Fitness of the particle
        fitness.append(p._fitness)  # An array that stores all the fitness of each and every particle
        p._personalBest = p._pos     # the personal best of the particle 
    max_fitness_index = fitness.index(max(fitness)) # stores the index of particle with best fitness
    best_fitness_obj = particles[max_fitness_index] # best fitness object in the swarm 
    best_pos = particles[max_fitness_index]._pos     # stores the position of the best particle  with max fitness
    best_fitness = particles[max_fitness_index]._fitness    # the best in the swarm 
    fitness_overall.append(best_fitness)    # add the current best fitness to fitness overall
    pos_overall.append(best_pos)    # add the current best position to position overall
    return


def cal(obj, dim, algo, gbest_fitness, gbest_position):
    fitness = []
    for i in range(0,no_particles):
        p = particles[i]    # get the particle object
        old_pos = p._pos    # get the old position
        old_velocity = p._velocity  # get the old velocity
        old_fitness = p._fitness    # get the old fitness
        old_pbest = p._personalBest # get the old personalBest
        cognitive = c1 * np.random.uniform(0,1, dim) * (old_pbest - old_pos) # the cognitive value
        social = c2 * np.random.uniform(0,1, dim) * (gbest_position - old_pos)   # the social value
        # the velocity update 
        temp_velocity = w * old_velocity + cognitive + social # summing the values
        _b = np.logical_and(temp_velocity >= vmax, temp_velocity<= vmin)
        new_velocity = np.where(~_b,old_velocity,temp_velocity) # masking the velocities that are reqired
        new_pos = (np.random.random_sample(size=dim) < sigmoid(new_velocity) )* 1   # the squezing of the velocity
        present_fitness = obj.getAccuracy(obj.data_cleaning(new_pos),algo)  # current fitness
        
        if present_fitness > old_fitness:
            p._personalBest = new_pos   # update the personal best if present pos is giving better fitness
            p._fitness = present_fitness    # update the fitness   if present pos is giving better fitness  
        p._pos = new_pos    # update the pos irrespective of the fitness
        p._velocity = new_velocity  # update the velocity of the particle
        fitness.append(p._fitness)  # add the local fitness


    max_fitness_index = fitness.index(max(fitness)) # stores the index of particle with best fitness
    best_fitness_obj = particles[max_fitness_index] # best fitness object in the swarm 
    best_pos = particles[max_fitness_index]._personalBest     # stores the position of the best particle  with max fitness
    best_fitness = particles[max_fitness_index]._fitness    # the best in the swarm 
    fitness_overall.append(best_fitness)    # (gbest_fitness)add the current best fitness to fitness overall
    pos_overall.append(best_pos)    # (gbest_pos)add the current best position to position overall
    return


def sigmoid(x):
    '''Helper function 
    Inputs
    ------
    x : np.ndarray
    input vector to compute the sigmoid form
    returns:
    np.ndarray
        output of sigmoiod 
    '''
    return 1/(1+np.exp(-x))

algo = 3  # Algorithm Value
if algo == 1:
    print("DecisionTree Algorithm")
elif algo == 2:
    print("NavieBayes")
else :
    print("k-Nearest Neighbour")

d = Data('abalone.csv',False, algo) # Object for Data
dim = d.getDimension()  # Dimensionality of the Features
run(d,dim,algo) # invoking




