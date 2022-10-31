import numpy as np
import matplotlib.pyplot as plt
from random import uniform, randint
from collections import namedtuple
import gym
from gym import spaces


pauli_x = np.array([[0, 1],[1, 0]])


action = namedtuple('action', ['pos', 'act'])
title = 'title'



        
class ToricModel():
    def __init__(self, dim, p, position_lattice, action_lattice, next_state=False):
        self.next_state = next_state
        self.dim = dim
        self.p = p
        self.position_lattice = position_lattice
        self.action_lattice = action_lattice
        self.qubit_lattice = np.zeros((2, self.dim, self.dim), dtype=object)
        self.error_lattice = np.zeros((2, self.dim, self.dim))
        self.star_lattice = np.zeros((self.dim, self.dim))
        self.next_star_lattice = np.zeros((self.dim, self.dim))
        self.error_pos = []
        self.ground = True
        
        for i in range(self.qubit_lattice.shape[0]):
            for j in range(self.qubit_lattice.shape[1]):
                for k in range(self.qubit_lattice.shape[2]):
                    self.qubit_lattice[i][j][k] = np.array([1, 0])


    # def action_to_node(self, action):
    #     n1 = action[0]
    #     n2 = (action[1][0])*self.dim + action[1][1]
    #     return (n1 + n2)
    
    # def node_to_action(self, act):
    #     a, b, c = np.where(self.action_lattice == act)
    #     action[0] = a[0]
    #     action[1] = np.array([b[0], c[0]])
    #     #print(node)
    #     #print(action)
    #     return action
    
    def done(self, state):
        done = np.all(state == 0)
        return done

    #I don't know if this works
    def is_ground(self):
        if np.sum(np.sum(self.error_lattice[0], axis=0)) % 2 == 1 or np.sum(np.sum(self.error_lattice[0], axis=0)) % 2 == 1:
            self.ground = False
        return self.ground
        
    def memory(self, a, r):
        pass

    #action is a position (x,y) in the lattice with a direction
    def step(self, action):
        #action = node_to_action(action)
        x, y = action[1]
        act = action[0]
        x, y, act = self.position_lattice[act][x, y]
        self.qubit_lattice[x, y][act] = pauli_x.dot(self.qubit_lattice[x, y][act])
        self.error_lattice[x, y][act] += 1 % 2
        self.syndrome()
        return self.star_lattice

    def star_x(self, vertex):
        for i in range(self.position_lattice.shape[0]):
            a, b, c= self.position_lattice[i][vertex[0]][vertex[1]]
            self.qubit_lattice[a][b][c] = pauli_x.dot(self.qubit_lattice[a][b][c])
        self.qubit_lattice_to_error_lattice()
            #print(a,b,c)
        #print('______')
        #return self.qubit_lattice
    
    def qubit_lattice_to_error_lattice(self):
        for i in range(self.qubit_lattice.shape[0]):
            for j in range(self.qubit_lattice.shape[2]):
                for k in range(self.qubit_lattice.shape[2]):
                    if all(self.qubit_lattice[i][j][k] == np.array([1,0])):
                        self.error_lattice[i,j,k] = 0
                    if all(self.qubit_lattice[i][j][k] == np.array([0,1])):
                        self.error_lattice[i,j,k] = 1
    
    # def star_lattice_to_qubit_lattice(self, stars):
    #     vertex_sites =  np.transpose(np.where(stars == 1))
    #     for i in range(vertex_sites.shape[0]):
    #         #print(vertex_sites[i])
    #         self.star_x(vertex_sites[i])
        
        #return self.qubit_lattice

    # def loop(qubits, positions):
    #     print(qubits)
    #     qubit= star_x(qubits, positions, [2,2])
    #     print(qubit)
    #     qubit= star_x(qubit, positions, [2,3])
    #     print(qubit)
    #     qubit= star_x(qubit, positions, [3,3])
    #     print(qubit)
    #     qubit= star_x(qubit, positions, [3,2])
    #     print(qubit)
    #     return qubit
    
    # def identity(qubits, positions):
    #     qubit= qubits
    #     for i in range(d):
    #         for j in range(d):
    #             qubit = star_x(qubits, positions, [i,j])
    #     return qubit
    
    def n_random_errors(self, n):
        #apply star operator to n random vertices
        for i in range(n):
            x = randint(0,self.dim-1)
            y = randint(0,self.dim-1)
            self.star_x([x, y])
            self.star_lattice[x][y] = 1
        
        #return qubits, stars

    def p_random_errors(self):
        #use p at ever vertex to determine an error
        errors = np.random.uniform(0, 1, size=(2, self.dim, self.dim))
        error = errors < self.p
        no_error = errors > self.p
        errors[error] = 1
        errors[no_error] = 0
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                for k in range(error.shape[2]):
                    if error[i,j,k] == 1:
                        self.qubit_lattice[i,j,k] = pauli_x.dot(self.qubit_lattice[i,j,k])
        self.error_lattice = errors
        self.syndrome()
        #title='title'
        #self.plot_toric_code(star_lattice, title)
        #return errors, self.star_lattice, self.qubit_lattice
        return self.star_lattice

    def syndrome(self):            
        x_errors0 = self.error_lattice[0]
        x_errors1 = self.error_lattice[1]
        x_errors0 = ((np.roll(x_errors0, 1, axis=0) + x_errors0) == 1).astype(int)
        x_errors1 = ((np.roll(x_errors1, 1, axis=1) + x_errors1) == 1).astype(int)
        if self.next_state == True:
            self.next_star_lattice = ((x_errors0 + x_errors1) == 1).astype(int)
        else:
            self.star_lattice = ((x_errors0 + x_errors1) == 1).astype(int)

    def plot_toric_code(self, state, title):
        x_error_qubits1 = np.where(self.error_lattice[0] == 1)

        x_error_qubits2 = np.where(self.error_lattice[1] == 1)

        vertex_defect_coordinates = np.where(self.star_lattice == 1)
        #plaquette_defect_coordinates = np.where(self.plaquette_lattice == 1)

        xLine = np.linspace(0, self.dim, self.dim)
        x = range(self.dim)
        X, Y = np.meshgrid(x,x)
        XLine, YLine = np.meshgrid(x, xLine)

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        ax = plt.subplot(111)
        ax.plot(XLine, -YLine, 'black', linewidth=linewidth)
        ax.plot(YLine, -XLine, 'black', linewidth=linewidth)
        ax.plot(XLine[:,-1] + 1.0, -YLine[:,-1], 'black', linewidth=linewidth)
        ax.plot(YLine[:,-1], -YLine[-1,:], 'black', linewidth=linewidth)
        ax.plot(X + 0.5, -Y, 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)
        ax.plot(X, -Y -0.5, 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)
        ax.plot(X[-1,:] + 0.5, -Y[-1,:] - 1.0, 'o', color = 'black', markerfacecolor = 'grey', markersize=markersize_qubit+1)
        ax.plot(X[:,-1] + 1.0, -Y[:,-1] - 0.5, 'o', color = 'black', markerfacecolor = 'grey', markersize=markersize_qubit+1)
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color = 'r', label="x error", markersize=markersize_qubit)
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color = 'r', markersize=markersize_qubit)
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$X$')
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color = 'black', markersize=markersize_symbols  , marker=r'$X$')

        #ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0], 'x', color = 'blue', label="charge", markersize=markersize_excitation)
        ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0], 'o', color = 'blue', label="charge", markersize=markersize_excitation)
        #ax.plot(plaquette_defect_coordinates[1] + 0.5, -plaquette_defect_coordinates[0] - 0.5, 'o', color = 'red', label="flux", markersize=markersize_excitation)
        ax.axis('off')
        
        #plt.title(title)
        plt.axis('equal')
        #plt.savefig('plots/graph_'+str(title)+'.png')
        plt.show()
        plt.close()
    