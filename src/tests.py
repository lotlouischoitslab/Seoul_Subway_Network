from utils import Graph

def test1(): # Just general graph data structure
    print('Test 1')
    nodes = ['A','B','C','D','E']
    graph = Graph(nodes)

    # Add edges to the graph
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('B', 'D', 1)
    graph.add_edge('C', 'D', 1)
    graph.add_edge('C', 'E', 1)

    graph.print_graph()

def test2(): # Check to delete
    print('Test 2')
    nodes = ['A','B','C','D','E']
    graph = Graph(nodes)

    # Add edges to the graph
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('B', 'D', 1)
    graph.add_edge('C', 'D', 1)
    graph.add_edge('C', 'E', 1)

    graph.print_graph()

    graph.remove_node('C')
    graph.print_graph()
    print('Remove',graph.nodes)

def test3(): # BFS & DFS
    print('Test 3')
    nodes = ['A','B','C','D','E']
    graph = Graph(nodes)

    # Add edges to the graph
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('B', 'D', 1)
    graph.add_edge('C', 'D', 1)
    graph.add_edge('C', 'E', 1)

    source = 'A'
    bfs_result = graph.bfs(source)
    print('BFS:',bfs_result) 

    dfs_result = graph.dfs(source)
    print('DFS:',dfs_result)


if __name__ == '__main__':
    test1()
    test2()
    test3()