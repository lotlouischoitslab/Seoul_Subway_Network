from utils import Graph

def test1():
    nodes = ['A','B','C','D','E']
    graph = Graph(nodes)

    # Add edges to the graph
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 1)
    graph.add_edge('B', 'D', 1)
    graph.add_edge('C', 'D', 1)
    graph.add_edge('C', 'E', 1)

    graph.print_graph()

    source = 'A'
    result = graph.bfs(source)
    print(result) 

if __name__ == '__main__':
    test1()