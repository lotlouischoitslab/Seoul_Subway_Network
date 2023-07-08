import numpy as np 
import pandas as pd 
import copy 

class Graph:
    def __init__(self,nodes):
        self.graph = dict()
        self.nodes = [tuple(n) for n in nodes]

        for n in self.nodes:
            self.graph[n] = []
    
    def copy_graph(self,to_copy):
        self.graph = copy.deepcopy(to_copy.graph)
        
    def add_node(self,add_node):
        add_node = tuple(add_node)
        if add_node not in self.nodes:
            self.nodes.append(add_node)
            self.graph[add_node] = [] 
    
    def add_edge(self,u,v,weight):
        u = tuple(u)
        v = tuple(v)
        if u in self.nodes and v in self.nodes:
            self.graph[u].append((v,weight))
    
    def remove_node(self,to_remove):
        to_remove = tuple(to_remove) 
        self.nodes.remove(to_remove)
        for n in list(self.graph.keys()):
            if n == to_remove:
                del self.graph[n]
            else:
                for neighbor,weight in self.graph[n]:
                    if neighbor == to_remove:
                        self.graph[n].remove((neighbor,weight))
    
    def remove_edge(self,u,v,weight):
        u = tuple(u)
        v = tuple(v) 
        if u in self.nodes and v in self.nodes:
            self.graph[u].remove((v,weight))
    
    def num_nodes(self):
        return len(self.nodes) 
    
    def num_edges(self):
        counter = 0 
        for n in list(self.graph.keys()):
            for neighbor,weight in self.graph[n]:
                counter += 1 
        return counter 
    
    def print_graph(self):
        for n in self.nodes:
            print(n,'-->',self.graph[n])