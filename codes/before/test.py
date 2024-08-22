import numpy as np
t = np.array([[1,2,3],[1,2,0],[1,0,0]])
a=[]
for i in range(4):
    a.append(t)
temp = np.stack(a,axis=1).reshape(-1,3)
print(temp)
# print(np.argpartition(t,3))