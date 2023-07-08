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
    
    def degree(self,node):
        node = tuple(node) 
        if node in self.nodes:
            in_degree = 0 
            out_degree = len(self.graph[node])

            for n in self.nodes: 
                if node != n:
                    for neighbor,weight in self.graph[n]:
                        if neighbor == node:
                            in_degree += 1

            return in_degree,out_degree
        else:
            return None,None 

    def print_graph(self):
        for n in self.nodes:
            print(n,'-->',self.graph[n]) 
    
    def bfs(self,source):
        if len(self.graph) == 0:
            return []
        
        source = tuple(source)
        to_traverse = [source]
        queue = [source]
        visited = set() 

        while len(queue) != 0:
            v = queue.pop(0) 
            visited.add(v) 

            for neighbor,weight in self.graph[v]:
                if neighbor not in visited:
                    queue.append(neighbor) 
                    to_traverse.append(neighbor)
                else:
                    continue 
        
        return to_traverse
