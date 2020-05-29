from dijkstra import Dijkstra
import pandas as pd
import numpy as np


mat = pd.read_csv("example.csv", index_col = 0)
mat = np.array(mat)

mat_size = len(mat)

while(1):
    start, finish = map(int, input().split())
    w, p = Dijkstra(mat_size, start, mat)
    
    path = []
    ans = 0
    while(finish != start):
        path.append(finish)
        ans += mat[p[finish]][finish]
        finish = p[finish]

    path = path[::-1]
    
    print("weight path")
    print(ans)
    print("weight")
    print(w)
    print("path")
    print(path)
    print()
