import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

ima1 = cv2.imread('foto1.png')
ima2 = cv2.imread('foto2.png')
ima1 = cv2.resize(ima1, (300,250))
ima2 = cv2.resize(ima2, (300,250))

"""
#suma metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("suma")
suma = ima1+ima2
print(suma)
cv2.waitKey(0)
cv2.destroyAllWindows()

#suma metodo 2
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("adicion")
adicion =cv2.add( ima1,ima2)
print(adicion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#suma metodo 3
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("metodo 3")
adicion =cv2.add( ima1,ima2)
print(np.add(ima1,ima2))
cv2.waitKey(0)
cv2.destroyAllWindows()"""
"""
#resta metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("resta 1")
resta = ima1-ima2
print(resta)
cv2.waitKey(0)
cv2.destroyAllWindows()

#resta metodo 2
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("metodo 2")
print(np.subtract(ima1,ima2))
cv2.waitKey(0)
cv2.destroyAllWindows()

#resta metodo 3
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("resta3")
subtract =cv2.subtract( ima1,ima2)
print(subtract)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
'''
#division metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("division 1")
division = ima1/ima2
print(division)
cv2.waitKey(0)
cv2.destroyAllWindows()

#division metodo 2
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("metodo 2")
print(np.divide(ima1,ima2))
cv2.waitKey(0)
cv2.destroyAllWindows()

#division metodo 3
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("division 3")
division =cv2.divide( ima1,ima2)
print(division)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''
#multiplicacion metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("multiplicacion 1")
multiplicacion = ima1*ima2
print(multiplicacion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#multiplicacion metodo 2
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print('multiplicacion metodo 2')
print(np.multiply(ima1,ima2))
cv2.waitKey(0)
cv2.destroyAllWindows()

#multiplicacion metodo 3
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("multiplicacion 3")
multiplicacion =cv2.multiply ( ima1,ima2)
print(multiplicacion)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''
#logn metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("logn 1")
logn = np.log(ima1)
print(logn)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#potencia metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("potencias 1")
potencias = ima1**2
print(potencias)
cv2.waitKey(0)
cv2.destroyAllWindows()

#potencia metodo 2
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print('potencia metodo 2')
print(np.power(ima1,2))
cv2.waitKey(0)
cv2.destroyAllWindows()

#potencia metodo 3
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("potencias 3")
potencias = cv2.pow(ima1,2)
print(potencias)
cv2.waitKey(0)
cv2.destroyAllWindows()

#derivada metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print ('derivada metodo 1')
dx = np.diff(ima1)
dy = np.diff(ima2)
d = dy/dx
print(d)
cv2.waitKey(0)
cv2.destroyAllWindows()
