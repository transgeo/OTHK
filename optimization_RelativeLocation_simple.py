import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import cplex
import scipy.linalg as sl
from scipy.linalg import block_diag

w1 = 1
w2 = 1
w3 = 4
w4 = 4

N = 8
L = 8
NN = 8
LL = 0
my_obj = [w1] * N \
         + [w2] * N \
         + [w3] * (NN - 2 * LL) \
         + [w4] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
my_ub = [2*math.pi] * N \
         + [16] * N \
         + [20] * (NN - 2 * LL) \
         + [20] * L \
         + [2*math.pi] * N \
         + [16] * N \
         + [2*math.pi] * L \
         + [1] * L * 4 \
         + [1] * L * 2 \
         + [1] * 2 * L \
         + [1] * L
my_lb = [0] * N \
         + [0] * N \
         + [0] * (NN - 2 * LL) \
         + [0] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
my_ctype = "C" * N \
         + "C" * N \
         + "I" * (NN - 2 * LL) \
         + "I" * L \
         + "C" * N \
         + "C" * N \
         + "C" * L \
         + "I" * L * 4 \
         + "I" * L * 2 \
         + "I" * 2 * L \
         + "I" * L
my_colnames = ["Z1_"+str(i) for i in range(N)] \
              + ["Z2_"+str(i) for i in range(N)]\
              + ["Z3_"+str(i) for i in range(NN - 2 * LL)]\
              + ["Z4_"+str(i) for i in range(L)]\
              + ["a"+str(i) for i in range(N)]\
              + ["r"+str(i) for i in range(N)]\
              + ["al"+str(i) for i in range(L)]\
              + ["Xa"+str(i)+"_"+str(j) for i in range(L) for j in range(4)]\
              + ["Xae"+str(i)+"_"+str(j) for i in range(L) for j in range(2)]\
              + ["Xr"+str(i)+"_"+str(j) for i in range(L) for j in range(2)]\
              + ["Xre"+str(i) for i in range(L)]

sta = pd.read_excel('A_simple.xlsx', sheet_name='station')
xx = np.array(sta['x'])
yy = np.array(sta['y'])
r = []
a = []
for i in range(xx.__len__()):
    r.append(math.sqrt(xx[i]*xx[i] + yy[i]*yy[i]))
    if math.atan2(yy[i], xx[i]) < 0:
        a.append(math.atan2(yy[i], xx[i]) + 2 * math.pi)
    else:
        a.append(math.atan2(yy[i], xx[i]))

link = pd.read_excel('A_simple.xlsx', sheet_name='link')
e = 0.001
M = 1000
pi = math.pi
rows = []
my_rhs = []
my_sense = []
# C1
C1Z1 = np.zeros([4*L, N])
C1Z2 = np.zeros([4*L, N])
C1Z3 = np.zeros([4*L, NN-2*LL])
C1Z4 = np.zeros([4*L, L])
C1a = np.zeros([4*L, N])
C1r = np.zeros([4*L, N])
C1al = np.zeros([4*L, L])
C1Xa = np.zeros([4*L, 4*L])
C1Xae = np.zeros([4*L, 2*L])
C1Xr = np.zeros([4*L, 2*L])
C1Xre = np.zeros([4*L, L])

for i in range(L):
    C1a[4 * i][link['ori'][i] - 1] = 1
    C1al[4 * i][i] = -1
    C1a[4 * i + 1][link['ori'][i] - 1] = -1
    C1al[4 * i + 1][i] = 1
    C1a[4 * i + 2][link['des'][i] - 1] = 1
    C1al[4 * i + 2][i] = -1
    C1a[4 * i + 3][link['des'][i] - 1] = -1
    C1al[4 * i + 3][i] = 1
C1Xa = np.eye(4*L) * 2 * pi

C1_arr = np.c_[C1Z1, C1Z2, C1Z3, C1Z4, C1a, C1r, C1al, C1Xa, C1Xae, C1Xr, C1Xre]
C1_list = C1_arr.tolist()
for i in range(C1_list.__len__()):
    rows.append([my_colnames, C1_list[i]])
C1_rhs = np.ones([4*L, 1]) * (2 * pi - e)

# C2
C2Z1 = np.zeros([4*L, N])
C2Z2 = np.zeros([4*L, N])
C2Z3 = np.zeros([4*L, NN-2*LL])
C2Z4 = np.zeros([4*L, L])
C2a = np.zeros([4*L, N])
C2r = np.zeros([4*L, N])
C2al = np.zeros([4*L, L])
C2Xa = np.zeros([4*L, 4*L])
C2Xr = np.zeros([4*L, 2*L])
C2Xre = np.zeros([4*L, L])

for i in range(L):
    C2a[4 * i][link['ori'][i] - 1] = 1
    C2al[4 * i][i] = -1
    C2a[4 * i + 1][link['ori'][i] - 1] = -1
    C2al[4 * i + 1][i] = 1
    C2a[4 * i + 2][link['des'][i] - 1] = 1
    C2al[4 * i + 2][i] = -1
    C2a[4 * i + 3][link['des'][i] - 1] = -1
    C2al[4 * i + 3][i] = 1
C2Xae = np.ones([2, 1]) * 2 * pi
for i in range(2*L-1):
    C2Xae = sl.block_diag(C2Xae, np.ones([2, 1]) * 2 * pi)

C2_arr = np.c_[C2Z1, C2Z2, C2Z3, C2Z4, C2a, C2r, C2al, C2Xa, C2Xae, C2Xr, C2Xre]
C2_list = C2_arr.tolist()
for i in range(C2_list.__len__()):
    rows.append([my_colnames, C2_list[i]])
C2_rhs = np.ones([4*L, 1]) * 2 * pi

# C3
C3Z1 = np.zeros([2*L, N])
C3Z2 = np.zeros([2*L, N])
C3Z3 = np.zeros([2*L, NN-2*LL])
C3Z4 = np.zeros([2*L, L])
C3a = np.zeros([2*L, N])
C3r = np.zeros([2*L, N])
C3al = np.zeros([2*L, L])
C3Xr = np.zeros([2*L, 2*L])
C3Xre = np.zeros([2*L, L])

C3Xae = np.eye(2*L)
C3Xa = np.array([1, 1])
for i in range(2*L-1):
    C3Xa = sl.block_diag(C3Xa, np.array([1, 1]))

C3_arr = np.c_[C3Z1, C3Z2, C3Z3, C3Z4, C3a, C3r, C3al, C3Xa, C3Xae, C3Xr, C3Xre]
C3_list = C3_arr.tolist()
for i in range(C3_list.__len__()):
    rows.append([my_colnames, C3_list[i]])
C3_rhs = np.ones([2*L, 1])

# C4
C4Z1 = np.zeros([2*L, N])
C4Z2 = np.zeros([2*L, N])
C4Z3 = np.zeros([2*L, NN-2*LL])
C4Z4 = np.zeros([2*L, L])
C4a = np.zeros([2*L, N])
C4r = np.zeros([2*L, N])
C4al = np.zeros([2*L, L])
C4Xa = np.zeros([2*L, 4*L])
C4Xae = np.zeros([2*L, 2*L])
C4Xr = np.zeros([2*L, 2*L])
C4Xre = np.zeros([2*L, L])

for i in range(L):
    C4r[2 * i][link['ori'][i] - 1] = 1
    C4r[2 * i][link['des'][i] - 1] = -1
    C4r[2 * i + 1][link['ori'][i] - 1] = -1
    C4r[2 * i + 1][link['des'][i] - 1] = 1
C4Xr = np.eye(2*L) * M

C4_arr = np.c_[C4Z1, C4Z2, C4Z3, C4Z4, C4a, C4r, C4al, C4Xa, C4Xae, C4Xr, C4Xre]
C4_list = C4_arr.tolist()
for i in range(C4_list.__len__()):
    rows.append([my_colnames, C4_list[i]])
C4_rhs = np.ones([2*L, 1]) * (M - e)

# C5
C5Z1 = np.zeros([2*L, N])
C5Z2 = np.zeros([2*L, N])
C5Z3 = np.zeros([2*L, NN-2*LL])
C5Z4 = np.zeros([2*L, L])
C5a = np.zeros([2*L, N])
C5r = np.zeros([2*L, N])
C5al = np.zeros([2*L, L])
C5Xa = np.zeros([2*L, 4*L])
C5Xae = np.zeros([2*L, 2*L])
C5Xr = np.zeros([2*L, 2*L])
C5Xre = np.zeros([2*L, L])

for i in range(L):
    C5r[2 * i][link['ori'][i] - 1] = 1
    C5r[2 * i][link['des'][i] - 1] = -1
    C5r[2 * i + 1][link['ori'][i] - 1] = -1
    C5r[2 * i + 1][link['des'][i] - 1] = 1
C5Xre = np.ones([2, 1]) * M
for i in range(L-1):
    C5Xre = sl.block_diag(C5Xre, np.ones([2, 1]) * M)

C5_arr = np.c_[C5Z1, C5Z2, C5Z3, C5Z4, C5a, C5r, C5al, C5Xa, C5Xae, C5Xr, C5Xre]
C5_list = C5_arr.tolist()
for i in range(C5_list.__len__()):
    rows.append([my_colnames, C5_list[i]])
C5_rhs = np.ones([2*L, 1]) * M

# C6
C6Z1 = np.zeros([L, N])
C6Z2 = np.zeros([L, N])
C6Z3 = np.zeros([L, NN-2*LL])
C6Z4 = np.zeros([L, L])
C6a = np.zeros([L, N])
C6r = np.zeros([L, N])
C6al = np.zeros([L, L])
C6Xa = np.zeros([L, 4*L])
C6Xae = np.zeros([L, 2*L])
C6Xr = np.zeros([L, 2*L])

C6Xre = np.eye(L)
C6Xr = np.array([1, 1])
for i in range(L-1):
    C6Xr = sl.block_diag(C6Xr, np.array([1, 1]))

C6_arr = np.c_[C6Z1, C6Z2, C6Z3, C6Z4, C6a, C6r, C6al, C6Xa, C6Xae, C6Xr, C6Xre]
C6_list = C6_arr.tolist()
for i in range(C6_list.__len__()):
    rows.append([my_colnames, C6_list[i]])
C6_rhs = np.ones([L, 1])

# C7
C7Z2 = np.zeros([2*N, N])
C7Z3 = np.zeros([2*N, NN-2*LL])
C7Z4 = np.zeros([2*N, L])
C7r = np.zeros([2*N, N])
C7al = np.zeros([2*N, L])
C7Xa = np.zeros([2*N, 4*L])
C7Xae = np.zeros([2*N, 2*L])
C7Xr = np.zeros([2*N, 2*L])
C7Xre = np.zeros([2*N, L])

C7a = np.vstack([np.eye(N), np.eye(N) * (-1)])
C7Z1 = np.vstack([np.eye(N) * (-1), np.eye(N) * (-1)])

C7_arr = np.c_[C7Z1, C7Z2, C7Z3, C7Z4, C7a, C7r, C7al, C7Xa, C7Xae, C7Xr, C7Xre]
C7_list = C7_arr.tolist()
for i in range(C7_list.__len__()):
    rows.append([my_colnames, C7_list[i]])
C7_rhs = np.vstack([np.array(a).reshape(a.__len__(), 1), -np.array(a).reshape(a.__len__(), 1)])

# C8
C8Z1 = np.zeros([2*N, N])
C8Z3 = np.zeros([2*N, NN-2*LL])
C8Z4 = np.zeros([2*N, L])
C8a = np.zeros([2*N, N])
C8al = np.zeros([2*N, L])
C8Xa = np.zeros([2*N, 4*L])
C8Xae = np.zeros([2*N, 2*L])
C8Xr = np.zeros([2*N, 2*L])
C8Xre = np.zeros([2*N, L])

C8r = np.vstack([np.eye(N), np.eye(N) * (-1)])
C8Z2 = np.vstack([np.eye(N) * (-1), np.eye(N) * (-1)])


C8_arr = np.c_[C8Z1, C8Z2, C8Z3, C8Z4, C8a, C8r, C8al, C8Xa, C8Xae, C8Xr, C8Xre]
C8_list = C8_arr.tolist()
for i in range(C8_list.__len__()):
    rows.append([my_colnames, C8_list[i]])
C8_rhs = np.vstack([np.array(r).reshape(r.__len__(), 1), -np.array(r).reshape(r.__len__(), 1)])

# C9
C9Z1 = np.zeros([2*(NN-2*LL), N])
C9Z2 = np.zeros([2*(NN-2*LL), N])
C9Z4 = np.zeros([2*(NN-2*LL), L])
C9a = np.zeros([2*(NN-2*LL), N])
C9r = np.zeros([2*(NN-2*LL), N])
C9al = np.zeros([2*(NN-2*LL), L])
C9Xae = np.zeros([2*(NN-2*LL), 2*L])
C9Xr = np.zeros([2*(NN-2*LL), 2*L])
C9Xre = np.zeros([2*(NN-2*LL), L])

sl = pd.read_excel('A_simple.xlsx', sheet_name='sl')

C9Xaa = np.zeros([(NN-2*LL), 4*L])
for i in range(sl.shape[0]):
    C9Xaa[i][(sl['OriLineID'][i]-1) * 4 + 2] = 1
    C9Xaa[i][(sl['OriLineID'][i]-1) * 4 + 3] = 1
    C9Xaa[i][(sl['DesLineID'][i]-1) * 4] = -1
    C9Xaa[i][(sl['DesLineID'][i]-1) * 4 + 1] = -1
C9Xa = np.vstack([C9Xaa, -C9Xaa])
C9Z3 = np.vstack([np.eye(NN-2*LL) * (-1), np.eye(NN-2*LL) * (-1)])

C9_arr = np.c_[C9Z1, C9Z2, C9Z3, C9Z4, C9a, C9r, C9al, C9Xa, C9Xae, C9Xr, C9Xre]
C9_list = C9_arr.tolist()
for i in range(C9_list.__len__()):
    rows.append([my_colnames, C9_list[i]])
C9_rhs = np.zeros([2*(NN-2*LL), 1])

# C10
C10Z1 = np.zeros([2*L, N])
C10Z2 = np.zeros([2*L, N])
C10Z3 = np.zeros([2*L, NN-2*LL])
C10a = np.zeros([2*L, N])
C10r = np.zeros([2*L, N])
C10al = np.zeros([2*L, L])
C10Xae = np.zeros([2*L, 2*L])
C10Xr = np.zeros([2*L, 2*L])

C10Xaa = np.ones([1, 4])
C10Xaa1 = np.ones([1, 4])
for i in range(L-1):
    C10Xaa1 = block_diag(C10Xaa1, C10Xaa)
C10Xa = np.vstack([C10Xaa1, -1 * C10Xaa1])
C10Xree = np.eye(L)
C10Xre = np.vstack([-2 * C10Xree, 2 * C10Xree])
C10_rhs = np.zeros([2*L, 1])
C10Z4 = np.vstack([np.eye(L) * (-1), np.eye(L) * (-1)])
C10_arr = np.c_[C10Z1, C10Z2, C10Z3, C10Z4, C10a, C10r, C10al, C10Xa, C10Xae, C10Xr, C10Xre]
C10_list = C10_arr.tolist()
for i in range(C10_list.__len__()):
    rows.append([my_colnames, C10_list[i]])



# #######################################
# Relative location constraints
C11Z1 = np.zeros([N*(N-1), N])
C11Z2 = np.zeros([N*(N-1), N])
C11Z3 = np.zeros([N*(N-1), NN-2*LL])
C11Z4 = np.zeros([N*(N-1), L])
C11al = np.zeros([N*(N-1), L])
C11Xa = np.zeros([N*(N-1), 4*L])
C11Xae = np.zeros([N*(N-1), 2*L])
C11Xr = np.zeros([N*(N-1), 2*L])
C11Xre = np.zeros([N*(N-1), L])

aarr = np.zeros([0, N])
for i in range(N-1):
    aarr0 = np.hstack((np.zeros([N-1-i, i]), np.ones([N-1-i, 1]), -np.eye(N - 1 - i)))
    aarr = np.vstack((aarr, aarr0))

C11ar1 = np.hstack((aarr, np.zeros([N*(N-1)/2, N])))
C11ar2 = np.hstack((np.zeros([N*(N-1)/2, N]), aarr))
C11ar = np.vstack((C11ar1, C11ar2))
C11_arr = np.c_[C11Z1, C11Z2, C11Z3, C11Z4, C11ar, C11al, C11Xa, C11Xae, C11Xr, C11Xre]
C11_list = C11_arr.tolist()
for i in range(C11_list.__len__()):
    rows.append([my_colnames, C11_list[i]])
C11_rhs = np.zeros([N*(N-1), 1])

# my_sense_11
ms11 = ""
for i in range(N-1):
    for j in range(i+1, N):
        if a[i] < a[j] or a[i] == a[j]:
            ms11 = ms11 + "L"
        else:
            ms11 = ms11 + "G"
for i in range(N-1):
    for j in range(i+1, N):
        if r[i] < r[j] or r[i] == r[j]:
            ms11 = ms11 + "L"
        else:
            ms11 = ms11 + "G"
# #######################################

my_rh = np.vstack([C1_rhs, C2_rhs, C3_rhs, C4_rhs, C5_rhs, C6_rhs, C7_rhs, C8_rhs, C9_rhs, C10_rhs, C11_rhs])
my_rhs = []
for i in range(my_rh.shape[0]):
    my_rhs.append(my_rh[i][0])
my_sense = "L" * (4 * L) \
           + "L" * (4 * L)\
           + "E" * (2 * L)\
           + "L" * (2 * L)\
           + "L" * (2 * L)\
           + "E" * L\
           + "L" * (2 * N)\
           + "L" * (2 * N)\
           + "L" * (2 * (NN - 2 * LL)) \
           + "L" * (2 * L) \
           + ms11

my_rownames = ["C1_"+str(i) for i in range(4 * L)] \
              + ["C2_"+str(i) for i in range(4 * L)]\
              + ["C3_"+str(i) for i in range(2 * L)]\
              + ["C4_"+str(i) for i in range(2 * L)]\
              + ["C5_"+str(i) for i in range(2 * L)]\
              + ["C6_"+str(i) for i in range(L)]\
              + ["C7_"+str(i) for i in range(2 * N)]\
              + ["C8_"+str(i) for i in range(2 * N)]\
              + ["C9_"+str(i) for i in range(2 * (NN - 2 * LL))] \
              + ["C10_"+str(i) for i in range(2 * L)]\
              + ["C11_"+str(i) for i in range(N*(N-1))]


my_prob = cplex.Cplex()
my_prob.objective.set_sense(my_prob.objective.sense.minimize)
my_prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype, names=my_colnames)
my_prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs, names=my_rownames)
my_prob.solve()


# solution.get_status() returns an integer code

numcols = my_prob.variables.get_num()
numrows = my_prob.linear_constraints.get_num()

slack = my_prob.solution.get_linear_slacks()
x = my_prob.solution.get_values()

theta = np.arange(0, 2*np.pi, 0.01)
r = x[40]
x_C = r * np.cos(theta)
y_C = r * np.sin(theta)
plt.plot(x_C, y_C, '--', color='darkgrey')
r = x[41]
x_C = r * np.cos(theta)
y_C = r * np.sin(theta)
plt.plot(x_C, y_C, '--', color='darkgrey')
r = x[43]
x_C = r * np.cos(theta)
y_C = r * np.sin(theta)
plt.plot(x_C, y_C, '--', color='darkgrey')
r = x[45]
x_C = r * np.cos(theta)
y_C = r * np.sin(theta)
plt.plot(x_C, y_C, '--', color='darkgrey')
r = x[47]
x_C = r * np.cos(theta)
y_C = r * np.sin(theta)
plt.plot(x_C, y_C, '--', color='darkgrey')



ao = x[32:40]
ro = x[40:48]
alo = x[48:56]
xxo = np.cos(ao) * ro
yyo = np.sin(ao) * ro
plt.scatter(xxo, yyo, c='r', s=30)

for i in range(L):
    ori_r = ro[link['ori'][i]-1]
    des_r = ro[link['des'][i]-1]
    ori_t = ao[link['ori'][i]-1]
    des_t = ao[link['des'][i]-1]
    l_t = alo[i]
    if abs(ori_r - des_r) < 0.01:
        if abs(ori_t - des_t) > math.pi:
            ori_tt = min(ori_t, des_t)
            des_tt = max(ori_t, des_t) - 2 * math.pi
        else:
            ori_tt = ori_t
            des_tt = des_t
        t = np.linspace(ori_tt, des_tt, 100)
        x1, y1 = np.cos(t) * ori_r, np.sin(t) * des_r
        plt.plot(x1, y1, c='r', alpha=0.7)
    if abs(ori_t - des_t) < 0.001:
        plt.plot([np.cos(ori_t) * ori_r, np.cos(des_t) * des_r], [np.sin(ori_t) * ori_r, np.sin(des_t) * des_r],
                 c='r', alpha=0.7)
    if abs(ori_r - des_r) > 0.01 and abs(ori_t - des_t) > 0.001:
        if abs(ori_t - des_t) > math.pi:
            ori_tt = min(ori_t, des_t)
            des_tt = max(ori_t, des_t) - 2 * math.pi
        else:
            ori_tt = ori_t
            des_tt = des_t
        t = np.linspace(ori_tt, des_tt, 100)
        x1, y1 = np.cos(t) * ori_r, np.sin(t) * ori_r
        plt.plot(x1, y1, c='r', alpha=0.7)
        plt.plot([np.cos(des_t) * ori_r, np.cos(des_t) * des_r], [np.sin(des_t) * ori_r, np.sin(des_t) * des_r],
                 c='r', alpha=0.7)

Z1_A = [1] * N \
         + [0] * N \
         + [0] * (NN - 2 * LL) \
         + [0] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
Z2_A = [0] * N \
         + [1] * N \
         + [0] * (NN - 2 * LL) \
         + [0] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
Z3_A = [0] * N \
         + [0] * N \
         + [1] * (NN - 2 * LL) \
         + [0] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
Z4_A = [0] * N \
         + [0] * N \
         + [0] * (NN - 2 * LL) \
         + [1] * L \
         + [0] * N \
         + [0] * N \
         + [0] * L \
         + [0] * L * 4 \
         + [0] * L * 2 \
         + [0] * 2 * L \
         + [0] * L
Z1 = sum(np.array(Z1_A) * np.array(x))
Z2 = sum(np.array(Z2_A) * np.array(x))
Z3 = sum(np.array(Z3_A) * np.array(x))
Z4 = sum(np.array(Z4_A) * np.array(x))

plt.scatter(xx, yy, c='midnightblue', s=15)

for i in range(L):
    ori_x = xx[link['ori'][i]-1]
    des_x = xx[link['des'][i]-1]
    ori_y = yy[link['ori'][i]-1]
    des_y = yy[link['des'][i]-1]
    l_t = alo[i]
    plt.plot([ori_x, des_x], [ori_y, des_y], color='#00B0F0')

print("Z1:= %.3f", Z1)
print("Z2:= %.3f", Z2)
print("Z3:= %.3f", Z3)
print("Z4:= %.3f", Z4)