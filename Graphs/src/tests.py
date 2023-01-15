from utils import Graph
import pandas as pd 

# Test Case 1 
def test1():
    example_graph1 = {
        'U': [['V',2], ['W', 5], ['X', 1]],
        'V': [['U',2], ['X', 2], ['W', 3]],
        'W': [['V',3], ['U', 5], ['X', 3],['Y', 1], ['Z', 5]],
        'X': [['U',1], ['V', 2], ['W', 3], ['Y',1]],
        'Y': [['X',1], ['W', 1], ['Z', 1]],
        'Z': [['W',5], ['Y', 1]]
    }
    nodes1= list(example_graph1.keys())
    test1 = Graph(nodes1)
    test1.copy_graph(example_graph1)
    #test1.print_graph()

    
    source1 = 'U'
    target1 = 'V'
    print('Source:',source1)
    print('Target:',target1)

    #BFS
    bfs_path = test1.bfs(source1)
    print('BFS Path:',bfs_path)

    #DFS
    dfs_path = test1.dfs(source1)
    print('DFS Path:',dfs_path)

    #Dijkstra
    path,distance = test1.dijkstra(source1,target1)
    print('Path:',path)
    print('Minimum Distance:',distance)

def testbellman():
    nodes = [0,1,2,3,4]
    g = Graph(nodes)
    g.addedge(0, 1, -1)
    g.addedge(0, 2, 4)
    g.addedge(1, 2, 3)
    g.addedge(1, 3, 2)
    g.addedge(1, 4, 2)
    g.addedge(3, 2, 5)
    g.addedge(3, 1, 1)
    g.addedge(4, 3, -3)
 
    # function call
    source = 0
    target = 3
    print('Source:',source)
    print('Target:',target)
    path,distance = g.bellman_ford(source,target)
    print('Path:',path)
    print('Minimum Distance:',distance)

def testfloyd_warshall():
    nodes = ['0','1','2','3']
    g = Graph(nodes)
    g.addedge('0', '1', 3)
    g.addedge('0', '3', 7)
    g.addedge('1', '0', 8)
    g.addedge('1', '2', 2)
    g.addedge('2', '0', 5)
    g.addedge('2', '3', 1)
    g.addedge('3', '0', 2)
 
    # function call
    source = '0'
    target = '2'
    print('Source:',source)
    print('Target:',target)
    path,distance = g.floyd_warshall(source,target)
    print('Path:',path)
    print('Minimum Distance:',distance)

    
def testcs():
    cs_core_graph_example = {
        'CS124':[['CS128',0],['CS173',0]],
        'CS128':[['CS222',0],['CS225',0],['CS233',0]],
        'CS173':[['CS225',0],['CS233',0]],
        'CS222':[],
        'CS225':[['CS374',0],['CS233',0],['CS222',0],['CS341',0]],
        'CS233':[['CS341',0],['CS421',0]],
        'CS374':[['CS421',0]],
        'CS341':[],
        'CS421':[]
    }
    cs_core_nodes = list(cs_core_graph_example.keys())
    assert(len(cs_core_nodes)==len(cs_core_graph_example.keys()))
    cs_core_graph = Graph(cs_core_nodes)
    cs_core_graph.copy_graph(cs_core_graph_example)
    cs_core_topo_ordering = cs_core_graph.topologicalsort()
    print('Computer Science Core Requirements Topological Ordering:',cs_core_topo_ordering)

    cs_math_req_example = {
        'MATH221':[['MATH231',0],['MATH241',0],['MATH257',0]],
        'MATH231':[['MATH241',0]],
        'MATH241':[['CS357',0]],
        'MATH257':[['CS357',0],['CS361',0]],
        'CS357':[],
        'CS361':[]
    }
    cs_math_req_nodes = cs_math_req_example.keys()
    assert(len(cs_math_req_nodes)==len(cs_math_req_example.keys()))
    cs_math_req_graph = Graph(cs_math_req_nodes)
    cs_math_req_graph.copy_graph(cs_math_req_example)
    cs_math_req_topo_ordering = cs_math_req_graph.topologicalsort()
    print('Computer Science Math Requirements Topological Ordering:',cs_math_req_topo_ordering)

def test_kruskal():
    g = Graph([0,1,2,3])
    g.addedge(0, 1, 10)
    g.addedge(1, 0, 10)

    g.addedge(0, 2, 6)
    g.addedge(2, 0, 6)

    g.addedge(0, 3, 5)
    g.addedge(3, 0, 5)

    g.addedge(1, 3, 15)
    g.addedge(3, 1, 15)

    g.addedge(2, 3, 4)
    g.addedge(3, 2, 4)
 
    # Function call
    
    result,min_cost = g.kruskal()
    print('Kruskal Result:',result)
    print('Kruskal Minimum Cost',min_cost)
    
def test_prims():
    g = Graph([0,1,2,3,4])
    g.addedge(0,1,2)
    g.addedge(1,0,2)

    g.addedge(0,3,6)
    g.addedge(3,0,6)

    g.addedge(1,2,3)
    g.addedge(2,1,3)

    g.addedge(1,3,8)
    g.addedge(3,1,8)

    g.addedge(1,4,5)
    g.addedge(4,1,5)

    g.addedge(2,4,7)
    g.addedge(4,2,7)

    g.addedge(3,4,9)
    g.addedge(4,3,9)
  
    result,min_cost = g.prims() 
    print('Prims Result:',result)
    print('Prim Minimum Cost',min_cost)

def test_remove():
    example_graph1 = {
        'U': [['V',2], ['W', 5], ['X', 1]],
        'V': [['U',2], ['X', 2], ['W', 3]],
        'W': [['V',3], ['U', 5], ['X', 3],['Y', 1], ['Z', 5]],
        'X': [['U',1], ['V', 2], ['W', 3], ['Y',1]],
        'Y': [['X',1], ['W', 1], ['Z', 1]],
        'Z': [['W',5], ['Y', 1]]
    }
    nodes1= list(example_graph1.keys())
    test1 = Graph(nodes1)
    test1.copy_graph(example_graph1)
    print('Before Removal')
    test1.print_graph()
    print('Nodes:',test1.nodes)
    print()
    to_remove = 'U'
    test1.removenode(to_remove)
    test1.removeedge('W','X',3)
    print('After Removal')
    test1.print_graph()
    print('Nodes:',test1.nodes)

#test_remove()

def testnetworkgenerator():
    print('Testing Network Generator')
    file1 = 'train_stations.xlsx'
    file2 = 'station_weights.xlsx'
    net = Graph([]) 
    net.network_generator(file1,file2)
    # net.print_graph()

    source = 'Jongno_3_Ga'
    target = 'Hoehyeon'
    #target = 'Dongdaemun_Design_Plaza'

    print('A-Star Algorithm:')
    path,distances = net.astar(source,target)
    print(f'Shortest Path from {source} to {target}:',path)
    print(f'Minimum Distance',distances)
    print()

    print('Bellman-Ford Algorithm:')
    path,distances = net.bellman_ford(source,target)
    print(f'Shortest Path from {source} to {target}:',path)
    print(f'Minimum Distance',distances)
    print()

    print('Dijkstra Algorithm:')
    path,distances = net.dijkstra(source,target)
    print(f'Shortest Path from {source} to {target}:',path)
    print(f'Minimum Distance:',distances)
    print()

    print('Floyd-Warshall Algorithm:')
    path,distances = net.floyd_warshall(source,target)
    print(f'Shortest Path from {source} to {target}:',path)
    print(f'Minimum Distance:',distances)
    print()


    print('Kruskal MST:')
    result,min_dist = net.kruskal()
    print(f'Kruskal MST:',result)
    print(f'Minimum Distance',min_dist)
    print()

    print('Prims MST:')
    result,min_dist = net.prims()
    print(f'Prims MST:',result)
    print(f'Minimum Distance',min_dist)
    print()

testnetworkgenerator()
#test1() #Graph Algorithms
#testcs() #Computer Science Courses Topological Sorting
#testbellman() #Bellman-Ford Algorithm
#testfloyd_warshall() #Floyd-Warshal Algorithm
#test_kruskal() #Kruskal Minimum-Spanning Tree Algorithm
#test_prims() #Prims Minimum-Spanning Tree Algorithm