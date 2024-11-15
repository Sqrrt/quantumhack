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

# Вычисление массива A, оставляя только положительные значения
A = np.array([q[a, -1] - q[a, 0] for a in range(100)])
A = A[A >= 0]
P1 = 10**6
k = len(A)

# Создание и заполнение массива h
exponents = 2 ** np.arange(u)
q_a_0_scaled = -10 * 2 * P1 * q[:k, 0][:, None] * exponents
h = -A[:, None] * exponents + q_a_0_scaled
h_ = h.ravel()  # Преобразование в одномерный массив

# Создание массива R
R = np.zeros((k, u, k, u))

# Предварительные вычисления разностей для q
q_diffs = np.diff(q[:k, :], axis=1)

# Вычисляем ковариационную матрицу для всех пар a, a_
cov_matrix = np.einsum('ai,aj->ij', q_diffs, q_diffs)
scaling_factor_same = 1 / P1**2
scaling_factor_diff = 1 / (P1**2 * (100 - 1))

# Векторизация коэффициентов для степеней двойки
exponents_outer = np.outer(exponents, exponents)

# Заполнение главной диагонали R (a == a_ и b == b_)
for a in range(k):
    for b in range(u):
        R[a, b, a, b] = cov_matrix[a, a] * scaling_factor_same * exponents[b] * exponents[b]

# Заполнение элементов R для a != a_ или b != b_
for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
                if not (a == a_ and b == b_):
                    R[a, b, a_, b_] = -cov_matrix[a, a_] * scaling_factor_diff * exponents[b] * exponents[b_]


J = np.zeros((k,u, k, u))


for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
 
                if a==a_ and b==b_:
                    J[a,b][a_, b_] = 10*q[a,0]*q[a_,0]*(2**b)*(2**b_) + h[a_,b_] + R[a,b][a_,b_]

                else:
                    J[a,b][a_, b_] = 10* q[a,0]*q[a_,0]*(2**b)*(2**b_) + R[a,b][a_,b_]
J_= J.reshape(k * u, k * u)

start = time()
sol = pq.solve(J_, number_of_runs=50, number_of_steps=200, return_samples=False, verbose=30)
# print(sol.vector, sol.objective)
# print(len(sol.vector))
np.savetxt("vector.txt", sol.vector, fmt='%d')
y = sol.vector.reshape(k, u)
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

print("Стоимость купленных акций: ", sum1, "$")  

sum = 0
for a in range(k):
    sum = sum + n[a]*A[a]

print("Прибыль: ", sum, "$")

sigma = 0
for a in range(k):
    for b in range(u):
        for a_ in range(k):
            for b_ in range(u):
              sigma += R[a,b][a_,b_] * y[a,b] * y[a_,b_]

print("Стандартное отклонение:",sigma)
print("Риск: ", np.sqrt(sigma))
