from queue import PriorityQueue
import heapq as hq
import pandas as pd

class Graph:
    def __init__(self,nodes):
        self.nodes = nodes #Nodes in the graph network
        self.graph = dict() #Because this is an empty dictionary, I will assume this to be empty dictionary

        #Now I will be adding empty edges!
        for n in self.nodes:
            self.graph[n] = []

    #Copy example graph to this one    
    def copy_graph(self,graph):
        self.graph = graph.copy()

    #Network Generator Function using CSV files 
    def network_generator(self,file1,file2):
         df1 = pd.read_excel(file1) #excel file with train station names
         df2 = pd.read_excel(file2) #excel file with weights from that train station to another

         self.nodes = list(df1.keys()) #train station nodes

         for key1,key2 in zip(df1,df2): #generate the graph
            self.graph[str(key1)] = []
            for elem,weight in zip(df1[key1],df2[key2]):
                elem = str(elem)
                if elem != 'nan':
                    self.graph[key1].append([elem,int(weight)])

    #Add edge to the graph network
    def addedge(self,u,v,weight):
        if u in self.nodes and v in self.nodes:
            self.graph[u].append([v,weight]) #What this means is that 'A':[['B',10]]
    
    #Add a node/vertex to the graph network
    def addnode(self,node):
        if node not in self.nodes:
            self.nodes.append(node)

    #Remove edge in the graph network
    def removeedge(self,u,v,weight):
        if u in self.nodes and v in self.nodes:
            self.graph[u].remove([v,weight])

    #Remove node in a graph network
    def removenode(self,to_remove):
        for node in self.nodes:
            for neighbor,weight in self.graph[node]:
                if neighbor == to_remove:
                    self.graph[node].remove([neighbor,weight])
        
        self.graph.pop(to_remove)
        self.nodes.remove(to_remove)
    
    #Degree of the node
    def degree(self,node):
        return len(self.graph[node])

    #Number of Edges
    def numedges(self):
        if self.directed:
            return (self.numvertices()*(self.numvertices()-1))/2
        return self.numvertices()*(self.numvertices()-1)
    
    #Number of Vertices/Nodes
    def numvertices(self):
        return len(self.nodes)

    #Print the entire Graph Network
    def print_graph(self):
        for node in self.nodes:
            print(node,'->',self.graph[node])
    
    #Fundamental Graph Algorithms from UIUC CS 374 Introduction to Algorithms & Models of Computation! 

    #Search Algorithms 
    # - Breadth-First Search (BFS) (Runtime: O(V+E) )Usually works on many different type of graphs
    # - Depth-First Search (DFS) (Runtime: O(V+E) )Likewise BFS 

    #Breadth-First-Search (BFS) (BarbeQueue)
    def bfs(self,source):
        if len(self.graph) == 0: #check if the graph network is empty
            return [] #return an empty list if the graph is empty
        
        visited = set() #Initialize all the elements in the visited set to be False
        visited.add(source) #The source has to be marked True
        queue = [source] #We are going to push the source vertex into the queue
        traversed = [] #This list will return which nodes in the graph network the BFS algorithm had to traverse
        
        while queue: #while the queue is not empty
            v = queue.pop(0) #pop the first-in element in the queue
            traversed.append(v) #append to the traverse element
            for node in self.graph: #for each node in graph
                if node not in visited: #if we have NOT visited the node
                    queue.append(node) #you need to append this node to the queue
                    visited.add(node) #mark this node as visited
        return traversed #return the traversed array

    #Depth-First-Search (DFS) (Developer Full Stack)
    def dfs(self,source): #we are going to start from source vertex
        visited = set() #construct a visited set
        return self.dfs_recursive(source,visited,[]) #call the dfs recursive helper function

    #DFS Recursive Helper Function
    def dfs_recursive(self,source,visited,to_return): 
        visited.add(source) #add the source vertex to the visited set
        to_return.append(source) #append the to return array each time we visit the array

        for neighbor in self.graph[source]: #for each neighbor in that graph node
            if neighbor[0] not in visited: #if this neigboring node is not visited
                self.dfs_recursive(neighbor[0],visited,to_return) #recurse
        return to_return #return the traversed nodes list 

    #Path Optimization Algorithms 
    #Very important to calculate shortest paths between source and target node
    # - A-Star Search (A*) (Runtime: O(V^2)) Heuristic Approach is good for Transit Network Optimization
    # - Dijkstra (Runtime: O(ElogV) if we use priority queues else O(E+VlogV))
    # - Bellman-Ford (Runtime: O(VE))
    # - Floyd-Warshall (Runtime: O(V^3))
    # - Topological Sort (Runtime: O(V+E))

    #A* Shortest Path Algorithm
    def heuristic(self,n):
        H_dist = dict()
        for n in self.nodes:
            H_dist[n] = 0
        return H_dist[n]

    #https://www.algorithms-and-technologies.com/a_star/python 
    def astar(self,source,target):
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
                if n == None or g[v]+self.heuristic(v) < g[n]+self.heuristic(n):
                    n = v 
                    distances[n] = g[v]+self.heuristic(v)
            
            if n == None:
                return Exception('PATH DOES NOT EXIST!')
            
            if n == target:
                path = []

                while parents[n] != n:
                    path.append([n,distances[n]])
                    n = parents[n]
                path.append([source,distances[source]])
                path.reverse()
                return path,distances[target]

            for neighbor,weight in self.graph[n]:
                if neighbor not in open_list and neighbor not in closed_list:
                    open_list.add(neighbor)
                    parents[neighbor] = n 
                    g[neighbor] = g[n]+weight  
                
                else:
                    if g[neighbor] > g[n]+weight:
                        g[neighbor] = g[n]+weight 
                        parents[neighbor] = n 

                        if neighbor in closed_list:
                            closed_list.remove(neighbor)
                            open_list.add(neighbor)
            
            open_list.remove(n)
            closed_list.add(n) 
        return Exception('PATH DOES NOT EXIST!')

   
    #Dijkstra's Single-Source Shortest Path Algorithm
    #Dijkstra: all possible shortest paths from one source given that it is DAG (Directed Acyclic Graphs) with no negative-weight edges!
    def dijkstra(self,source,target): 
        queue = [(source,0)] #Initialize queue (node,weight)
        distances = {}
        for n in self.nodes:
            distances[n] = float('inf')
        
        distances[source] = 0 #Initialize the starting source to be 0
        visited = set() #create a set of visited nodes
        parent = dict() #dictionary of parent nodes which will be backtracked to reconstruct paths

        while queue: #BFS (Barbequeue)
            node,w = hq.heappop(queue) #pop the queue
            if node in visited: #If the current node is visited
                continue #Don't stop and please proceed onwards

            visited.add(node) #add the current node to the visited set
            dist = distances[node] #update the distance to be the current distance 
            
            for neighbor,weight in self.graph[node]:
                if neighbor in visited: #Likewise if the neighbor is visited
                    continue #Don't stop proceed to the next stage
                
                weight += dist #modify the weight
                if weight < distances.get(neighbor, float('inf')): #if this path is optimal
                    hq.heappush(queue, (neighbor,weight)) #let's add it to the priority queue
                    distances[neighbor] = weight #update the distances as well
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

    #Bellman-Ford Algorithm: All single source shortest paths from one source. This works on DAGS with negative-weight edges.
    def bellman_ford(self,source,target):
        distances = {} #Initialize all the distances 
        for n in self.nodes:
            distances[n] = float('inf')
        distances[source] = 0
        parent = dict() #Parent dictionary
        V = len(self.nodes) #Number of vertices in the graph

        for i in range(V-1):
            for node in self.nodes: #for each node in the node list we created
                for neighbor,weight in self.graph[node]: #we are then going to check each neighbor
                    if distances[node] != float('inf') and distances[node]+weight < distances[neighbor]: #if this path is optimal
                        distances[neighbor] = distances[node]+weight #update the distances as well
                        parent[neighbor] = node #assign the parent node

        
        for node in self.nodes: #for each node in nodes
            for neighbor,weight in self.graph[node]: #for each neighbor
                if distances[node] != float('inf') and distances[node]+weight < distances[neighbor]: #if this is the case
                    return Exception('GRAPH CONTAINS NEGATIVE WEIGHT CYCLE!') #Give a warning
        
        path = self.construct_path(source, target, parent, distances) #Construct Path
        return path,distances[target] #return the path and the minimum distance

    
    def convert_to_matrix(self,V,yes_inf):
        dist = [[float('inf') if yes_inf == True else 0 for j in range(V)] for i in range(V)] #all distances will be initialized as +inf
        
        for node in self.nodes: #for each node in the graph network
            dist[self.nodes.index(node)][self.nodes.index(node)] = 0 #diagonal to be set to 0
            for neighbor,weight in self.graph[node]: #for each neighbor in the node
                dist[self.nodes.index(node)][self.nodes.index(neighbor)] = weight #distance from i->j is assigned as respective weight
        return dist 
        
    #Floyd-Warshall Algorithm is much more optimized version of Bellman-Ford Algorithm
    def floyd_warshall(self,source,target):
        V = len(self.nodes) #V is going to be the number of nodes/vertices in the graph
        Next = [[float('inf') for j in range(V)] for i in range(V)] #likewise all the next values
        yes_inf = True #We are going to by default set everything to +inf
        dist = self.convert_to_matrix(V,yes_inf) #convert this to adjacency matrix

        for i in range(V): #we are going to iterate through the graph network 
            for j in range(V): #same situation for here too
                if dist[i][j] == float('inf'): #if the distance from i to j is infinite, no path exists
                    Next[i][j] = -1 #Next[i][j] is assigned as -1
                else:
                    Next[i][j] = j #Otherwise the neighbor index

        for i in range(V): #calculate shortest-paths
            for j in range(V):
                for k in range(V):
                    if (dist[i][k] == float('inf') or dist[k][j] == float('inf')):
                        continue #continue on
                    if (dist[i][j] > dist[i][k] + dist[k][j]): #we found minimum path
                        dist[i][j] = dist[i][k] + dist[k][j] #assign it to the distance from i->j
                        Next[i][j] = Next[i][k] #assign the next column value
        
        src_index = self.nodes.index(source) #source index
        targ_index = self.nodes.index(target) #target index
        path = self.floyd_path(Next,src_index,targ_index) #reconstruct the Floyd-Warshall Path
        return path,dist[src_index][targ_index] #return the path and the distance from source to target 

    def floyd_path(self,Next,u,v): #Helper function for Floyd-Warshall
        if Next[u][v] == -1: #If Next is -1 this means no path exists
            return [] #return empty list
        path = [[self.nodes[u],u]] #Assign the path to be the soure node first
        while u != v: #while the source node is NOT equal to the target node
            u = Next[u][v] #keep updating the source
            path.append([self.nodes[u],u]) #keep updating the path
        return path #return the path

    #Topological Sort Algorithm
    #We use Topological Sort Algorithm in cases when the source vertex of the graph is unknown 
    def topologicalsort(self):
        indegrees = {node : 0 for node in self.nodes} #incoming degree nodes

        for node in self.nodes: #for each node in our graph nodes
            for neighbor,weight in self.graph[node]: #for each neighbor between two nodes
                indegrees[neighbor]+=1  #we are going to try incrementing the degree by 1
        
        nodes_with_no_incoming_edges = [] #We are going to determine the nodes with no incoming edges
        
        for node in self.nodes: #for each node in the graph
            if indegrees[node] == 0: #we are going to check if there are no incoming edges
                nodes_with_no_incoming_edges.append(node) #if none, go ahead and append it to the list
        
        topological_ordering = [] #Initially, no nodes in our topological ordering

        while nodes_with_no_incoming_edges: #while there are nodes with incoming edges
            node = nodes_with_no_incoming_edges.pop() #Pop it off from the stack
            topological_ordering.append(node) #Add one of those nodes to the ordering
            
            for neighbor,weight in self.graph[node]: #for each neighbor and weight in the graph
                indegrees[neighbor]-=1 #Decrement the indegree of that node's neighbor
                if indegrees[neighbor] == 0: #If there are no incoming edges 
                    nodes_with_no_incoming_edges.append(neighbor) #Go ahead and add it onto the stack
        
        if len(topological_ordering) == len(self.graph): #Assertion just to make that we are working with DAGS
            return topological_ordering 
        return Exception('WARNING! GRAPH HAS A CYCLE! THIS IS NOT A DAG!') #Otherwise give a big warning!


    #Minimum Spanning Tree Algorithms
    # - Kruskal's Minimum Spanning Tree (Runtime: O(ElogV)) Sparse Graphs
    # - Prim's Minimum Spanning Tree (Runtime: O(V^2) -> O(ElogV) (Fibonacci Heap)) Dense Graphs
    # - REMEMBER KRUSKAL AND PRIMS Only works with Undirected Weighted Graphs
    
    #Kruskal's Minimum Spanning Tree Algorithm
    #We are going to be applying the Union-Find Algorithm using disjoint sets
    def find(self,parent,i):
        if parent[i] != i: #if the parent of i is not i
            return self.find(parent,parent[i]) #keep performing the union find
        return parent[i] #return the parent
    
    def union(self,parent,rank,x,y): 
        if rank[x] < rank[y]: #if the rank of x is less than rank of y
            parent[x] = y #parent of x is assigned as y
        elif rank[x] > rank[y]: #if rank of x is higher than rank of y
            parent[y] = rank[x] #assign the parent of y to be rank of x
        else:
            parent[y] = x #otherwise parent of y is x
            rank[y] +=1 #increment the rank of y by 1

    
    def convert_node_to_int(self):
        d = dict() #dictionary
       
        for key,val in enumerate(self.nodes):
            d[val] = key #assign each key with value

        int_nodes = list(d.values()) #we are going to then convert the nodes into integer numbers
        temp = Graph(int_nodes) #temporary graphs
        
        for node in self.nodes: #for each node
            for neighbor,weight in self.graph[node]: #retrive each neighbor and weight
                temp.graph[d[node]].append([d[neighbor],int(weight)]) #build the graph
 
        return temp,int_nodes #we then return the temp and the nodes

    def kruskal(self):
        result = [] #result path
        new_graph = Graph(self.nodes) #Deep Copy
        new_graph,int_nodes = self.convert_node_to_int() #convert the nodes into integers

        for node in int_nodes: #we are going to sort the graph by edge weights
            new_graph.graph[node] = sorted(new_graph.graph[node],key=lambda item:item[1]) #we are going to sort the graph by their edges
        
        parent = [] #parents of the nodes
        rank = [] #rank of the nodes
        V = len(new_graph.nodes) #number of vertices

        for i in range(V):
            parent.append(i) #append all the parents
            rank.append(0) #initialize all the ranks to be 0
        
        
        idx,e = 0,0 #idx and edge index
        
        while e < V-1 and idx < len(new_graph.nodes): #we don't want the edge index to exceed one less than the number of vertices
            u = new_graph.nodes[idx] #node u
            v,weight = new_graph.graph[new_graph.nodes[idx]][0] #node v and the edge weight
            idx+=1 #increment the index
            x = new_graph.find(parent,u) #find the parent of u
            y = new_graph.find(parent,v) #find the parent of v

            if x != y: #if x and y are not equal to each other
                e +=1 #increment the edge index by 1
                result.append([u,v,weight]) #append the results 
                new_graph.union(parent, rank, x, y) #perform union-find operator
        
        min_cost = 0 #calculate the min cost
        results = {
            'Edge':[],
            'Weight':[]
        }
        for u,v,weight in result: #for each node u,v and edge weight
            min_cost+=weight #add the weight to the min cost
            results['Edge'].append(str(u)+'-'+str(v))
            results['Weight'].append(weight)
        
        return results,min_cost #return the results and the min cost


    #Prim's Minimum Spanning Tree Algorithm
    def minkey(self,key,mst): #Function to find vertex with minimum distance value from the set of vertices not included yet
        local_min = float('inf') #local minimum value
        min_index = -1 #min index
        V = len(self.nodes) #number of vertices
        for v in range(V): #for each vertex in vertices
            if key[v] < local_min and mst[v] == False: #if the key is less than the local minimum and mst is not visited
                local_min = key[v] #assign the local min with the key[v]
                min_index = v #minimum index becomes v
        return min_index #return the minimum index
    
    def prims(self):
        V = len(self.nodes) #number of vertices
        key = [float('inf')]*V #initialize all the keys to be +inf
        parent = [None]*V #all parents will be initialized as None value
        key[0] = 0 #first one if initialized as 0
        mst = [False]*V  #all are initialized as False
        parent[0] = -1 #First node is always the root of tree

        new_graph = Graph(self.nodes) #Create new Graph object so we can perform Deep Copy 
        new_graph.copy_graph(self.graph) #Deep Copy of the graph
        new_graph,int_nodes = new_graph.convert_node_to_int() #convert all the nodes into integers
        yes_inf = False  #we are going to by default set all the nodes in the matrix to 0
        
        adj_matrix = new_graph.convert_to_matrix(V,yes_inf) #build the adjacency list

        for i in range(V):
            u = self.minkey(key, mst) #now for each vertex 
            mst[u] = True  #mark the mst as visited
            
            for j in range(V): #for each neighbor
                if adj_matrix[u][j] and mst[j] == False and key[j] > adj_matrix[u][j]: #relaxation process
                    key[j] = adj_matrix[u][j] #assign the key to be that
                    parent[j] = u #update the parents
        result = {
            'Edge':[],
            'Weight':[]
        }
        for i in range(1,V):
            result['Edge'].append(str(parent[i])+'-'+str(i))
            result['Weight'].append(adj_matrix[i][parent[i]])
        return result,sum(result['Weight'])
