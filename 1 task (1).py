from time import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import random
from scipy import stats
import pandas as pd
import pyqiopt as pq



df = pd.read_csv('task-1-stocks.csv')
u = 20

q = np.array(df.iloc[:, :]).T

A = np.zeros([100])
for a in range(100):
    A[a] = q[a,-1] - q[a,0]
A = A[A >= 0]
P1 = 10**6
k = len(A)

h = np.zeros([k,u])
for α in range(k):
    for β in range(u):
        # h[α, β] = (A[α] + 2*10*P1*q[α, 0])*(2**β)
        h[α, β] = - A[α]*(2**β) - 10*2*P1*q[α, 0]*(2**β)
        # h[α, β] = (2*P1*q[α, 0])*(2**β)*10
        # h[α, β] = (2*P1*q[α, 0])*2**β
        # h[α, β] = (A[α])*2**β
h_ = h.reshape(k*u)

R = np.zeros((k,u, k, u))


for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
              # J[a,b][a_, b_] = 10**6*q[a,0]*q[a_,0]*(2**b)*(2**b_)
                if a==a_ and b==b_:
                    R[a,b][a_, b_] = sum(sum(1/P1**2 * (q[a,i+1] - q[a, i])*(q[a_,j+1] - q[a_,j]) * 2**b * 2**b_ for j in range(99))for i in range(99))
                      # J[a,b][a_, b_] = - h[a_,b_]
                else:
                    R[a,b][a_, b_] = - sum(sum(1/P1**2 / (100-1) * (q[a,i+1] - q[a, i])*(q[a_,j+1] - q[a_,j]) * 2**b * 2**b_ for j in range(99) ) for i in range(99))
                    
J = np.zeros((k,u, k, u))


for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
              # J[a,b][a_, b_] = 10**6*q[a,0]*q[a_,0]*(2**b)*(2**b_)
                if a==a_ and b==b_:
                    J[a,b][a_, b_] = 10*q[a,0]*q[a_,0]*(2**b)*(2**b_) + h[a_,b_] + R[a,b][a_,b_]
                      # J[a,b][a_, b_] = - h[a_,b_]
                else:
                    J[a,b][a_, b_] = 10* q[a,0]*q[a_,0]*(2**b)*(2**b_) + R[a,b][a_,b_]
J_= J.reshape(k * u, k * u)
# for i in range(len(h_)):
#     J_[i, i] -= h_[i]
# J_ = np.zeros([100*u,100*u])

# for α in range(100):
#     for β in range(u):
#         for α_ in range(100):
#             for β_ in range(u):
#                 J_[u*α + β, u*α_ + β_] = J[α,β][α_,β_]




start = time()
sol = pq.solve(J_, number_of_runs=10, number_of_steps=200, return_samples=False, verbose=30)
print(sol.vector, sol.objective)
print(len(sol.vector))
np.savetxt("vector.txt", sol.vector, fmt='%d')
y = sol.vector.reshape(k, u)
# y = np.zeros([100,u])
# for g in range(len(sol.vector)):
#     a = g//u
#     b = g % u
#     y[a,b] = sol.vector[g]
print(y)


n = np.zeros(k)
for a in range(k):
  n[a] = 0
  print(y[a,:])
  for b in range(u):      
      n[a] += y[a,b]*(2**b)
      

print(n)

sum1 = 0
for a in range(k):
    sum1 += n[a]*q[a,0]

print(sum1)  

sum = 0
for a in range(k):
    sum = sum + n[a]*A[a]

print(sum)
    
sigma = 0
for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
              sigma += R[a,b][a_,b_] * y[a,b] * y[a_,b_]

print(sigma)


