import numpy as np
import matplotlib.pyplot as plt
from engine import *

#całośc na podstawie video: # https://www.youtube.com/watch?v=VMj-3S1tku0

#przykładowa funkcja x^2+x
def f(x: int):
    return x**2 + x 

x = np.arange(-20, 20, 0.1)
y = f(x)
# plt.plot(x, y)

#obliczenie pochodnej metodą numeryczna -> obieramy mały przyrost "h"
h = 0.0001
x1 = 3 #punkt w jakim liczymy pochodną - czyli patrzymy jak funkcja zachowuje sie w obliczu malych zmian argumentu (czy rosnie, maleje lub pozostaje bez zmian i jak mocno to robi)
grad = (f(x1+h) - f(x1)) / h
# print(f(x1))
# print(f(x1+h))
# print(grad)

#odnosnie samej funkcji f(x):
#x^2 + x to funkcja ktorej wykresem jest nadal parabola x^2 przesunieta odpowiednio w dol i lewo
#wynika to np z pochodnej - dla obu stron paraboli pochodna jest taka sama (rózni sie znakiem) - a wiec parabola jest taka sama, bo pochodna ma taki sam wspolczynnik kierunkowy w kazdym punkcie
#np: x=1,5 -> pochodna = 4; x=-2,5 -> pochodna = -4 ; tak samo dla punktow odsunietych o 3 od srodka x=-0,5 -> to punkty 2,5 (pochodna 6) oraz -3,5 (pochodna -6)
#przesuniecie w lewo tez wynika z pochodnej - wierzcholek jest w punkcie -0.5 a nie w 0.0
#przesuniecie w dol wynika z tego, ze sa 2 miejsca zerowe funkcji x^2+x
#pochodna mówi czy funkcja w danym punkcie dla dx rosnie czy maleje; 
#pochodna jest przydatna do analizy funkcji - mozna dzieki niej stwierdzic czy np funkcja jest symetryczna (taka sama pochodna ale przeciwny znak)

#pochodna funkcji wielu zmiennych:
a = 4 
b = -3 
c = 5 
def d(): 
    return a*b + c 
#pochodna wzgledem a|b|c
d1 = d()
# a += h  # dy|da = -3
# b += h  # dy|db = 4
# c += h  # dy|dc = 1
d2 = d()
g = (d2-d1)/h 
# print(f'd1=${d1}, d2=${d2}, g=${g}')

