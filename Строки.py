'''a = str(input())
g = ''
for i in range(len(a)-1):
    if a[i] != a[i+1]:
        g += a[i]
if a[-1] != g[-1]:
    g += a[-1]
print(g)
h = ''.join(sorted(set(a), key=a.index))
print(h)
j = ''.join(dict.fromkeys(a))
print(j)'''
'''from random import *
def rand_matrix(n,m):
    A = [[randint(0,9) for j in range(m)] for i in range(n)]
    return A
def unit_matrix(n):
    A = [[int(i == j) for i in range(n)] for j in range(n)]
    return A
def mult_matrix(A,B):
    n = len(A)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] = A[i][k]*B[k][j]
    return C
def show_matrix(A):
    for a in A:
        for b in A:
            print(b, end=" ")
        print()
seed(2014)
A = rand_matrix(3, 5)
print("Список: ", A)
print("Эта же матрица: ")
show_matrix(A)
E = unit_matrix(4)
print("Единичная матрица: ")
show_matrix(E)
A1 = rand_matrix(3,3)
A2 = rand_matrix(3,3)
A3 = mult_matrix(A1, A2)
print("Первая матрица: ")
show_matrix(A1)
print("Вторая матрица: ")
show_matrix(A2)
print("Произведение матриц: ")
show_matrix(A3)'''
n = 100
E = {s for s in range(1, n + 1)}
A1 = {s for s in E if s % 5 == 2}
A2 = {s for s in E if s % 5 == 4}
A = A1 | A2
B = {s for s in E if s % 7 == 3}
C = {s for s in E if s % 3 == 1}
D = A & B - C
F = list(D)
F.sort()
print("Приведенные ниже числа от 1 до", n)
print("1) при делении на 5 дают в остатке 2 или 4:\n", A)
print("2) при делении на 7 дают в остатке 3:\n", B)
print("3) при делении на 3 не дают в остатке 1:", C)
print("4) при делении на 5 дают в остатке 2 или 4, \n"
      "при делении на 7 дают в остатке 3, а при делении \n"
      "на 3 не дают в остатке 1:", F)