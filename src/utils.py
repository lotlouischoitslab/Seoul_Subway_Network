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
    
    def add_edge(self,u,v,weight): # add an edge between the nodes
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
        if len(self.graph) == 0: # if the graph is empty
            return [] # just return the empty list 
        
        source = tuple(source) # source node
        to_traverse = [source] # add the source node to the traversed list 
        queue = [source] # push the source to the queue
        visited = set() # create an empty visited set

        while len(queue) != 0: # while the queue is not empty
            v = queue.pop(0) # pop the element from the queue
            visited.add(v) # add the popped element into the visited set

            for neighbor,weight in self.graph[v]: # for each neighbor and weight
                if neighbor not in visited and neighbor not in queue: # if the neighbor is not visited and not in the queue
                    queue.append(neighbor) # add the neighbor into the queue
                    to_traverse.append(neighbor) # add the neighbor to the traversed list
        
        return to_traverse # return the traversed nodes

    def dfs(self,source): # DFS
        visited = set() # visited
        source = tuple(source) # convert source to tuple
        return self.dfs_recursive(source,visited,[]) # call the recursive dfs function
    
    def dfs_recursive(self,source,visited,to_traverse): # helper function for dfs
        visited.add(source) # add the visited set
        to_traverse.append(source) # add the traverse list

        for neighbor,weight in self.graph[source]: # for each neighbor and weight
            if neighbor not in visited: # if the neighbor is not visited
                self.dfs_recursive(neighbor,visited,to_traverse) # call the recursive function
        
        return to_traverse # return the to_traverse

    #A* Shortest Path Algorithm
    def heuristic(self,n):
        H_dist = dict()
        for n in self.nodes:
            H_dist[n] = 0
        return H_dist[n]

    def astar(self, source, target): # A* Shortest Paths
        source = tuple(source) # convert source to tuple
        target = tuple(target) # convert target to tuple
        open_list = set([source]) # open list to store nodes to be explored
        closed_list = set([]) # closed list to store visited nodes

        g = {} # dictionary to store the cost from the start node to each node
        g[source] = 0

        distances = {} # dictionary to store the total estimated distance from the start node to each node
        distances[source] = 0

        parents = {} # dictionary to store the parent node of each node in the shortest path
        parents[source] = source

        while len(open_list) > 0: # begins a loop that continues until there are no more nodes left to explore in open_list
            n = None # n is set None to represent the current node being examined
            for v in open_list: # loop iterates over each node v in the open list
                if n is None or g[v] + self.heuristic(v) < g[n] + self.heuristic(n): # it checks if n is None or the estimated total cost of reaching v is lower than the estimated cost
                    n = v # set n to v
                    distances[n] = g[v] + self.heuristic(v) # update the distance

            if n is None: # if n is None
                return None  # No path exists and no more paths to explore and no path exists from source to target

            if n == target: # if the current node n is equal to the target, this means that the shortest path has been found
                path = [] # we start with empty path and iteratively add each node n and its corresponding path

                while parents[n] != n: # continue to add path until reaching the source node
                    path.append([n, distances[n]]) # add the path and distance
                    n = parents[n] # n = parent
                path.append([source, distances[source]]) # append the path
                path.reverse() # reverse the path
                return path, distances[target] # return the path and the distances

            for neighbor, weight in self.graph[n]: # for each neighbor node and weight in graph[n]
                if neighbor not in open_list and neighbor not in closed_list: # for each neighbor, it checks if the neighbor is in open_list or closed_list
                    open_list.add(neighbor) # if the neighbor is not in either of them, add to open list and its cost and parent information are updated
                    parents[neighbor] = n # update the parent node
                    g[neighbor] = g[n] + weight # update the weight

                else: # If the neighbor is in the closed_list, it means it was visited before
                    if g[neighbor] > g[n] + weight: # check if shorter path has been found
                        g[neighbor] = g[n] + weight
                        parents[neighbor] = n

                        if neighbor in closed_list: # if the neighbor is in the closed list, this means that it was visited before
                            closed_list.remove(neighbor) # neighbor is removed from closed list
                            open_list.add(neighbor) # neighbor is added on to the open list

            open_list.remove(n) # once all the neighbors of n have been processed, n is removed from open list
            closed_list.add(n) # then added to the closed list

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
