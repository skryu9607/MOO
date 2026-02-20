import numpy as np
w_star = [0,0.536317,0.463683]
f_star = [29.9075,1.62665,3.93782]
w = [[1,0,0],[0,1,0],[0,0,1]]
f= [[20.0154,0.959798,10.0077,], 
        [20.3631,0.959532,10.1816],
        [24.5904,2.81677,2.21509]]
max_corner = np.max(np.array(f),axis =0)
print(max_corner)
u_star = sum([w_star[i]*f_star[i]/max_corner[i] for i in range(3)])
print(u_star)
for i in range(3):
    u = [sum([w_star[j]*f[i][j]/max_corner[j] for j in range(3)])]
    print(u)
    print(u[0]-u_star)
    print(w[i])
