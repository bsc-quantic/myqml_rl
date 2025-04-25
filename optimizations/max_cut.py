from mc_graph import mc_graph
from qaoa import qaoa

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize
from collections import Counter


class max_cut():

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.graph = mc_graph(self.nodes, self.edges)
        self.max_cut_hamiltonian()
        self.max_cut_classical_hamiltonian()


    def __init__(self, graph):
        self.graph = graph
        self.nodes = self.graph.nodes
        self.edges = self.graph.edges
        self.max_cut_hamiltonian()
        self.max_cut_classical_hamiltonian()


    '''
       creates list of clauses
    '''
    def max_cut_hamiltonian(self):
        self.clauses = []
        for edge in self.graph.get_edges():
            # clause_list.append([-1, "Z"+str(edge[0])+" "+"Z"+str(edge[1])])
            self.clauses.append([edge[0], edge[1]])


    '''
       create energy function
    '''
    def max_cut_classical_hamiltonian(self):
        self.energy = []
        for i, edge in enumerate(self.graph.get_edges()):
            self.energy.append(lambda bitstring, edgev=edge: (1-(1-2*bitstring[edgev[0]])*(1-2*bitstring[edgev[1]]))/2)


    '''
       get clauses
    '''
    def get_clauses(self):
        return self.clauses

    '''
       get energy
    '''
    def get_energy(self):
        return self.energy

    '''
       get number of nodes
    '''
    def get_nodes(self):
        return self.nodes




    def solve_qaoa_lattice(self, partitions, p):
            #	p = parameters['p']
            #	iterations = parameters['iterations']
            #	partitions = parameters['partitions']
            # Initialize table of rewards
            rewards = np.zeros([partitions for i in range(2*p)])
            partitioned_array = np.reshape(np.meshgrid(*np.reshape([j for j in range(partitions)]*(2*p), [2*p, partitions])), [2*p, -1, ])
            
            # Iterate trying all the angle values in a lattice.
            for i in range(len(partitioned_array[0])):
                #print(i)
                state_int = np.array(partitioned_array[:, i])
                state = np.zeros(2*p)
                #state[0:p] = np.pi/2.0 * state_int[0:p] / partitions
                state[0:p] = np.pi * state_int[0:p] / partitions
                state[p:] = np.pi * state_int[p:] / partitions
                print(self.nodes, self.clauses)
                qc = qaoa(self.nodes, self.clauses)
                qc.set_params(state)
                result = qc.evaluate_circuit(1000)
                rewards[tuple(state_int)] = result/2.0
                #print(i)
                #print(state)
                #print(result)
                #print(state, result)

            #self.plot_rewards_surface(partitions, rewards)
            return rewards



    def plot_rewards_surface(self, partitions, rewards):
        fig = plt.figure()
        ax = Axes3D(fig)
        delta_angle = np.pi/partitions
        x = np.arange(0, partitions, 1)
        y = np.arange(0, partitions, 1)
        gamma, beta = np.meshgrid(x, y)
        beta = beta * delta_angle / 2.0
        gamma = gamma * delta_angle
        ax.plot_surface(gamma,beta,rewards, cmap=cm.coolwarm)
        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\beta$')
        plt.show()


    def solve_qaoa_minimization(self, method='L-BFGS-B', iterations=1):

        def max_cut_objective(params):
            qc = qaoa(self.get_nodes(), self.get_clauses())
            qc.set_params(params)
            return qc.evaluate_circuit(1000)

        def max_cut_mes(params):
            qc = qaoa(self.get_nodes(), self.get_clauses())
            qc.set_params(params)
            qc.create_circuit()
            return qc.measure_circuit(100000)

        params = np.random.uniform(0, np.pi/2, size=2*iterations)
        #method = 'Nelder-Mead'
        #method = 'TNC'
        #method = 'SLSQP'
        #method='L-BFGS-B'
        minimization = minimize(max_cut_objective, params, method=method, tol=1e-40, options={'disp': True, 'maxiter':500})
        
        return minimization, max_cut_mes(minimization.x)

    def evaluate(self, measures,qc):
        rewardvector = []
        avg_reward = 0.0
        max_reward = 0.0
        best_classic_state = []

        for key in measures.keys():
            reward = 0.0
            bitstring = str(key)
            qbit_observation = [int(s) for s in bitstring]
            # Set the reward equal to the number of clauses fulfilled by the measured |gamma,beta>.

            for j in range(len(self.energy)):
                reward += self.energy[j](qbit_observation)
            rewardvector.append([bitstring, reward, measures[key]/100000.0])
            avg_reward += reward * measures[key]
            if reward == 10:
                best_classic_state = qbit_observation
                max_reward = reward
        return avg_reward/100000, best_classic_state, rewardvector

def plot_histogram(reward_vector, filename):
    fig = plt.figure()
    color = ['b','r','k','g']
    for k in range(4):
        counts = np.array(np.array(reward_vector[k])[:, 2], dtype=float)
        rewards = np.array(np.array(reward_vector[k])[:, 1], dtype=float)
        labels, _ = zip(*Counter(rewards).items())
        values = np.zeros(len(labels))
        labels =np.array(labels)
        for i, label in enumerate(labels):
            values[i] = np.sum([counts[rewards == label]])
            labels[i] = labels[i] -0.3+0.2*k
        plt.bar(labels, values / sum(values), width=0.175, label= 'p = '+str(k+1), color = color[k])

    plt.xlabel('Cut Size')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('D:/Master/TFM/Github/bsc_master_ml/QAOA/Plots/Histogram/' + filename + '.png', bbox_inches='tight')
    plt.show(fig)
    return labels, values
