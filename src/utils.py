import numpy as np 
import pandas as pd 
import copy 
import heapq as hq
from queue import PriorityQueue

class Graph: # Graph Class
    def __init__(self,nodes):
        self.graph = dict() # Initialize an empty graph 
        self.nodes = [tuple(n) for n in nodes] # convert all the nodes into tuples

        for n in self.nodes: # put all the nodes in the graph
            self.graph[n] = []
    
    def copy_graph(self,to_copy): # perform deep copy
        self.graph = copy.deepcopy(to_copy.graph)
        
    def add_node(self,add_node): # add a node to the graph
        add_node = tuple(add_node) # convert the node to tuple
        if add_node not in self.nodes: # if the node is not in the nodes
            self.nodes.append(add_node) # we want to add the node
            self.graph[add_node] = [] # obviously initialize this
    
    def add_edge(self,u,v,weight):
        u = tuple(u) # convert u into tuples
        v = tuple(v) # convert v into tuples
        if u in self.nodes and v in self.nodes: # if both are not in the nodes
            self.graph[u].append((v,weight)) # then add 
    
    def remove_node(self,to_remove): # remove the node
        to_remove = tuple(to_remove) # convert to tuple
        self.nodes.remove(to_remove) # remove this from the nodes
        for n in list(self.graph.keys()): # for each node
            if n == to_remove: # if this is the node we want to remove
                del self.graph[n] # remove this node from the graph
            else: # otherwise
                for neighbor, weight in self.graph[n]: # for each neighboring node
                    if neighbor == to_remove: # if this is what we want to remove
                        self.graph[n].remove((neighbor,weight)) # remove this from the graph 
    
    def remove_edge(self,u,v,weight): # remove the edge between u and v
        u = tuple(u) # convert u into tuples
        v = tuple(v) # convert v into tuples
        if u in self.nodes and v in self.nodes: # if both u and v are in the nodes
            self.graph[u].remove((v,weight)) # then remove the edge between them 
    
    def num_nodes(self): # get the number of nodes
        return len(self.nodes) # length of the node list 
    
    def num_edges(self): # get the number of edges
        counter = 0 # counter is set to 0
        for n in list(self.graph.keys()): # for each node 
            for neighbor,weight in self.graph[n]: # for each neighboring node
                counter += 1 # increment the number of edges
        return counter # return the number of edges
    
    def degree(self,node): # calculate the degree of the graph
        node = tuple(node) # convert the node into tuple
        if node in self.nodes: # if the node is in the nodes
            in_degree = 0 # in degree is 0
            out_degree = len(self.graph[node]) # out degree is simply the length of the nodes going out from that node

            for n in self.nodes: # for each node
                if node != n: # if both nodes are not the same
                    for neighbor,weight in self.graph[n]:
                        if neighbor == node: # if the neighbor node is coming into the node
                            in_degree += 1 # increment the number of incoming degrees

            return in_degree,out_degree # return the in degree and out degree
        else:
            return None,None # otherwise return none

    def print_graph(self): # print out the entire graph
        for n in self.nodes: # for each node
            print(n,'-->',self.graph[n]) # print out the adjacency list
    
    def bfs(self,source): # BFS Algorithm
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
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor) 
                    to_traverse.append(neighbor)
        
        return to_traverse

    def dfs(self,source):
        visited = set()
        source = tuple(source)
        return self.dfs_recursive(source,visited,[])
    
    def dfs_recursive(self,source,visited,to_traverse):
        visited.add(source) 
        to_traverse.append(source) 

        for neighbor,weight in self.graph[source]:
            if neighbor not in visited:
                self.dfs_recursive(neighbor,visited,to_traverse)
        
        return to_traverse

    #A* Shortest Path Algorithm
    def heuristic(self,n):
        H_dist = dict()
        for n in self.nodes:
            H_dist[n] = 0
        return 2

    def astar(self, source, target):
        source = tuple(source)
        target = tuple(target)
        open_list = set([source])
        closed_list = set([])

        g = {}
        g[source] = 0

        distances = {}
        distances[source] = 0

        parents = {}
        parents[source] = source

        while len(open_list) > 0:
            n = None
            for v in open_list:
                if n is None or g[v] + self.heuristic(v) < g[n] + self.heuristic(n):
                    n = v
                    distances[n] = g[v] + self.heuristic(v)

            if n is None:
                return None  # No path exists

            if n == target:
                path = []

                while parents[n] != n:
                    path.append([n, distances[n]])
                    n = parents[n]
                path.append([source, distances[source]])
                path.reverse()
                return path, distances[target]

            for neighbor, weight in self.graph[n]:
                if neighbor not in open_list and neighbor not in closed_list:
                    open_list.add(neighbor)
                    parents[neighbor] = n
                    g[neighbor] = g[n] + weight

                else:
                    if g[neighbor] > g[n] + weight:
                        g[neighbor] = g[n] + weight
                        parents[neighbor] = n

                        if neighbor in closed_list:
                            closed_list.remove(neighbor)
                            open_list.add(neighbor)

            open_list.remove(n)
            closed_list.add(n)

        return None  # No path exists

    def dijkstra(self,source,target): 
        source = tuple(source)
        target = tuple(target)
        queue = [(source,0)] #Initialize queue (node,weight)
        distances = {n : float('inf') for n in self.nodes} #Initialize all the nodes to be +inf
        
        distances[source] = 0 #Initialize the starting source to be 0
        parent = dict() #dictionary of parent nodes which will be backtracked to reconstruct paths

        while queue: #BFS (Barbequeue)
            node,w = hq.heappop(queue) #pop the queue
            dist = distances[node] #update the distance to be the current distance 
            
            for neighbor,weight in self.graph[node]:
                if distances[node] != float('inf') and distances[node]+weight < distances[neighbor]:  #if this path is optimal
                    hq.heappush(queue, (neighbor,weight)) #let's add it to the priority queue
                    distances[neighbor] = distances[node]+weight #update the distances as well
                    parent[neighbor] = node #set the neighbor parent to the current node
                
        path = self.construct_path(source, target, parent, distances)
        return path,distances[target] #return the path and optimal weight to arrive to that final target
    
    #Just a helper function to reconstruct path
    def construct_path(self,source,target,parent,distances):
        path = [] #set the path to be empty initially
        while True:
            path.append([target,distances[target]]) #keep appending the elements
            if source == target: #if we arrive at a source
                break  #Stop and break out
            target = parent[target] #Reassign the target
        path.reverse() #Reverse because this is backtracking
        return path #Finally return the path!
