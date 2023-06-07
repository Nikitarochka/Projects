import math as Math
def f(x):
    l = int(input("l = "))
    m = int(input("m = "))
    h = 0
    z = 0
    for i in range(l, m):
        y = l + l/i
        z = z + y
    s: float = z + x
    for k in range(1, m):
        g = (x + 5/k)**2
        h = h + g
    u: float = l + h
    t: float = s / u
    return t
def f1(x, y):
    if x > -1 and x < 1 and y > -1 and y < 1:
        if x < 0 and x > -1 and y > 0 and y < 0.5:
            if Math.sqrt(x**2 + y**2) <= Math.sqrt((-1)**2 + (0.5)**2):
                print("Точка лежит в тёмной области, 2 октант")
        if x > 0 and x < 1 and y > -1 and y < 0:
            if Math.sqrt(x ** 2 + y ** 2) <= 3.14*1**2 - Math.sqrt((-1) ** 2 + (0.5) ** 2):
                print("Точка лежит в тёмной области, 4 октант")
        if x > 0 and x < 1 and y > 0 and y < 1:
            if Math.sqrt(x ** 2 + y ** 2) <= 3.14*1**2 - Math.sqrt((-1) ** 2 + (0.5) ** 2):
                print("Точка лежит в тёмной области, 4 октант")
print(f(5))



