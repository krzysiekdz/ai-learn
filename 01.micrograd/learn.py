import numpy as np
import matplotlib.pyplot as plt

#przykładowa funkcja x^2+x
def f(x):
    return x**2 + x 

x = np.arange(-20, 20, 0.1)
y = f(x)
plt.plot(x, y)

#obliczenie pochodnej metodą numeryczna -> obieramy mały przyrost "h"
h = 0.0001
x1 = 3 #punkt w jakim liczymy pochodną
grad = (f(x1+h) - f(x1)) / h
print(f(x1))
print(f(x1+h))
print(grad)

#x^2 + x to funkcja ktorej wykresem jest nadal parabola x^2 przesunieta odpowiednio w dol i lewo
#wynika to np z pochodnej - dla obu stron paraboli pochodna jest taka sama (rózni sie znakiem) - a wiec parabola jest taka sama, bo pochodna ma taki sam wspolczynnik kierunkowy w kazdym punkcie
#przesuniecie w lewo tez wynika z pochodnej - wierzcholek jest w punkcie -0.5 a nie w 0.0
#przesuniecie w dol wynika z tego, ze sa 2 miejsca zerowe funkcji x^2+x
#pochodna mówi czy funkcja w danym punkcie dla dx rosnie czy maleje

# 25 minuta ; zrobic ogolne repo do nauki ai, i  tam podfolder to np micrograd
# https://www.youtube.com/watch?v=VMj-3S1tku0&t=732s
