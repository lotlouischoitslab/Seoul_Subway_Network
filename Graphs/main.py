import numpy as np 
import pandas as pd 
from src.utils import Graph

'''
Seoul Metro Shortest Paths: 서울지하철 경로찾기
Author: Louis Sungwoo Cho: 조성우 
Created: 1/9/2023
Dataset: https://develop-dream.tistory.com/89 

Seoul Metro Subway Names:   서울지하철역 
Jongno_3_Ga:                종로3가
Jongno_5_Ga:                종로5가
Dongdaemun:                 동대문
Jonggak:                    종각
City_Hall:                  시청
Euljiro_1_Ga:               을지로입구
Euljiro_3_Ga:               을지로3가
Euljiro_4_Ga:               을지로4가
Dongdaemun_Design_Plaza:    동대문역사문화공원
Seoul_Station:              서울역
Hoehyeon:                   회현
Myeongdong:                 명동
Chungmuro:                  충무로
'''

def load_data():
    file1 = 'train_stations.xlsx'  #Get the Excel file1
    file2 = 'station_weights.xlsx' #Get the Excel file2
    return file1,file2

def main():
    print('Seoul Metro Shortest Paths by Louis Sungwoo Cho')
    df1,df2 = load_data() #DataFrame of subway networks
    louis_network = Graph([]) #This should be a Graph object receiving a 1-D array
    louis_network.network_generator(df1, df2)
    louis_network.print_graph()
    
if __name__ == '__main__':
    main()