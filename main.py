import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import networkx as nx
from src.utils import Graph

'''
Seoul Metro Shortest Paths: 서울지하철 최단경로찾기
Author: Louis Sungwoo Cho: 조성우 
Created: 7/8/2023
'''

def load_data():
    with open('Data/Seoul_Subway.txt', 'r', encoding = "UTF-8") as file:
        S = file.read().split('\n')
        
    subway = {'No':[], 'Station Name':[], 'Line':[]}
    for i,x in enumerate(S):
        if i == 724: # end of the (no, name, line) pair
            break
        
        no, name, line = x.split()
        subway['No'].append(no)
        subway['Station Name'].append(name)
        subway['Line'].append(line)

    subway = pd.DataFrame(subway)
    return subway

def graph_visualizer(graph):
    G = nx.Graph()

    for node in graph.nodes:
        for neighbor,weight in graph.graph[node]:
            G.add_edge(node,neighbor,weight=weight)

    
    pos=nx.spring_layout(G)
    nx.draw_networkx(G,pos,node_color = "g",font_size=3)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
    plt.axis('off')
    plt.draw()
    plt.savefig("../images/metro_network.png")


def main():
    print('Seoul Metro Shortest Paths by Louis Sungwoo Cho')
    print('조성우 서울지하철 최단경로찾기')

    subway = load_data()
    print(subway)
    subway.to_csv('Data/stations.csv')
   
    
if __name__ == '__main__':
    main()