from qibo.models import Circuit
from qibo.gates import RX, CNOT, X, R, H,RZ
from qibo.run import run
from qibo.backends import qcgpu, vqmlite

import numpy as np
from scipy import sparse

class qaoa():


    def __init__(self, N, clauses):
        self.N = N
        self.clauses = clauses
        self.build_operator()


    def set_params(self, params):
        self.params = params
        self.iterations = int(params.shape[0]/2)


    def set_clauses(self, clauses):
        self.clauses = clauses


    def create_circuit(self):
        self.circuit = Circuit(self.N)

        ''' C operator '''
        def add_C_operator(gamma):
            for qbit1, qbit2 in self.clauses:	
                '''self.circuit.add(X(qbit2))
                self.circuit.add(R(qbit2, -gamma/2))
                self.circuit.add(X(qbit2))
                self.circuit.add(R(qbit2, -gamma/2))
                self.circuit.add(CNOT(qbit2, qbit1))
                self.circuit.add(X(qbit1))
                self.circuit.add(R(qbit1, gamma/2))
                self.circuit.add(X(qbit1))
                self.circuit.add(R(qbit1, -gamma/2))
                self.circuit.add(CNOT(qbit2, qbit1))'''
                '''self.circuit.add(CNOT(qbit1, qbit2))
                self.circuit.add(RZ(qbit2, gamma))
                self.circuit.add(CNOT(qbit1, qbit2))'''            
            self.circuit.gate(X, target=qbit2, control=qbit1)
            self.circuit.gate(rz(gamma), target=qbit2)
            self.circuit.gate(X, target=qbit2, control=qbit1)
        
        ''' B operator '''
        def add_B_operator(beta): 
            for pos in range(self.N):
                self.circuit.add(RX(pos, -beta))
                self.circuit.gate(H, target=pos)


        for pos in range(self.N):
            #self.circuit.add(H(pos))
            self.circuit.gate(H, target=pos)
        for it in range(self.iterations):
            add_C_operator(self.params[it*2])
            add_B_operator(self.params[it*2+1])



    def measure_circuit(self, shots):    
        #backend = qcgpu.QCGPU(self.circuit)
        backend = vqmlite.VQMlite(self.circuit)
        results = run(backend, shots=shots)
        return results



    def evaluate_circuit(self, shots):
        self.create_circuit()
        results = self.measure_circuit(shots)
        wf = results['wave_func'].get_all()
        return np.real(np.dot(np.transpose(np.conj(wf)), self.H.dot(wf)))


    def build_operator(self):

        def build_term(pos1, pos2):
            gates_1 = [sparse.identity(pow(2, pos1)),sparse.csr_matrix(pauliZ),sparse.identity(pow(2, pos2 - pos1 - 1)), sparse.csr_matrix(pauliZ),sparse.identity(pow(2, self.N - pos2 - 1))]
            g_1 = gates_1[0]
            for elem in range(1, len(gates_1)):
                g_1 = sparse.kron(g_1, gates_1[elem])
            return sparse.identity(pow(2, self.N)) - g_1


        self.H = sparse.csr_matrix( (2**self.N,2**self.N) )
        pauliZ = np.array([[1, 0], [0, -1]])
        for clause in self.clauses:
            self.H = self.H + build_term(clause[0], clause[1])


  


