from networkx import random_regular_graph, draw_networkx_edges, spring_layout, draw_networkx_nodes, draw_networkx_labels
from networkx import Graph

import matplotlib.pyplot as plt


class mc_graph():
    def __init__(self):
        self.nodes = 0
        self.edges = 0

    '''
       create general graph
    '''
    def generate_list(self, list_edges):
        self.graph = Graph()
        self.graph.add_edges_from(list_edges)
        self.nodes = self.graph.number_of_nodes()
        self.edges = self.graph.number_of_edges()
        self.values = [1 for node in self.graph.nodes()]
        #self.filename = 'Initial' + str(self.nodes)+'Q_'+str(self.edges)+'E_'


    '''
       create regular graph
    '''
    def generate_regular(self, edges, nodes):
        self.edges = edges
        self.nodes = nodes
        self.graph = random_regular_graph(self.edges, self.nodes)
        self.values = [1 for node in self.graph.nodes()]
        #self.filename = 'Initial' + str(self.nodes)+'Q_'+str(self.edges)+'E_'


    '''
       plot graph
    '''
    def plot(self):
        pos = spring_layout(self.graph)
        draw_networkx_nodes(self.graph, pos, node_color=self.values, node_size=500)
        draw_networkx_labels(self.graph, pos)
        draw_networkx_edges(self.graph, pos)
        #plt.savefig('./Plots/Graphs/'+self.filename+'.png', bbox_inches='tight')
        plt.show()



    '''
       dump self vars
    '''
    def dump(self):
        return self.edges, self.nodes, self.graph, self.values

    '''
       get nodes
    '''
    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.graph.edges


 







