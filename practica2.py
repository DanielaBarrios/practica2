import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

def suma(m1, m2):
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    mr = m1 + m2
    cv2.imshow("suma",mr)
    cv2.waitKey(0)
    
def suma2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)    
    mr = cv2.add( ima1,ima2)
    cv2.imshow("suma2",mr)
    cv2.waitKey(0)
    
def suma3(m1, m2):
    cv2.destroyAllWindows()  
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2) 
    mr = np.add( ima1,ima2)
    cv2.imshow("suma3",mr)
    cv2.waitKey(0)
    
def resta1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1-ima2
    cv2.imshow("resta1",mr)
    cv2.waitKey(0)
    
def resta2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.subtract(ima1,ima2)
    cv2.imshow("resta2",mr)
    cv2.waitKey(0)
    
def resta3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.subtract( ima1,ima2)
    cv2.imshow("resta3",mr)
    cv2.waitKey(0)
    
def division1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1/ima2
    cv2.imshow("division1",mr)
    cv2.waitKey(0)
    
def division2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.divide(ima1,ima2)
    cv2.imshow("division2",mr)
    cv2.waitKey(0)
    
def division3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.divide( ima1,ima2)
    cv2.imshow("division3",mr)
    cv2.waitKey(0)
    
def multiplicacion1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1*ima2
    cv2.imshow("multiplicacion1",mr)
    cv2.waitKey(0)
    
def multiplicacion2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.multiply(ima1,ima2)
    cv2.imshow("multiplicacion2",mr)
    cv2.waitKey(0)
    
def multiplicacion3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.multiply ( ima1,ima2)
    cv2.imshow("multiplicacion3",mr)
    cv2.waitKey(0)  
    
def raiz1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1**(0.5)
    cv2.imshow("raiz1",mr)
    cv2.waitKey(0) 
    
def raiz2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = pow(ima1,0.5)
    cv2.imshow("raiz2",mr)
    cv2.waitKey(0)    
    
def raiz3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.sqrt(ima1)
    cv2.imshow("raiz3",mr)
    cv2.waitKey(0) 
    
def derivada1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    dx = np.diff(ima1)
    dy = np.diff(ima2)
    d = dy/dx
    cv2.imshow("derivada1",d)
    cv2.waitKey(0)    
    
def potencia1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = m1**2
    cv2.imshow("potencia1",mr)
    cv2.waitKey(0)    
    
def potencia2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.power(m1,2)
    cv2.imshow("potencia2",mr)
    cv2.waitKey(0)    
    
def potencia3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.pow(m1,2)
    cv2.imshow("potencia3",mr)
    cv2.waitKey(0)  
    
def conjuncion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_and(m1, m2)
    cv2.imshow("conjuncion",mr)
    cv2.waitKey(0)
    
def disyuncion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_or(m1,m2)
    cv2.imshow("disyuncion",mr)
    cv2.waitKey(0)
    
def negacion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_not(m1)
    cv2.imshow("negacion",mr)
    cv2.waitKey(0)
    
def trasafin(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    ancho = m1.shape[1] #columnas
    alto = m1.shape[0] # filas
    mr = np.float32([[1,0,100],[0,1,150]])
    imageOut = cv2.warpAffine(m1,mr,(ancho,alto))
    cv2.imshow("traslacion afin",imageOut)
    cv2.waitKey(0)
    
def escalado(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.resize(m1,(600,300), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("escalado",mr)
    cv2.waitKey(0)
    
def rotacion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    ancho = m1.shape[1] #columnas
    alto = m1.shape[0] # filas
    mr = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
    imageOut = cv2.warpAffine(m1,mr,(ancho,alto))
    cv2.imshow("rotacion",imageOut)
    cv2.waitKey(0)

    
ima1 = cv2.imread('foto1.png')
ima2 = cv2.imread('foto2.png')
ima1 = cv2.resize(ima1, (300,250))
ima2 = cv2.resize(ima2, (300,250))

cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
x=cv2.waitKey(0)

while True:

    if x == ord("d"):
        suma(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        suma2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        suma3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        resta1(ima1,ima2)
        x=cv2.waitKey(0)

        
    if x == ord("d"):
        resta2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        resta3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        division1(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        division2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        division3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion1(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion3(ima1,ima2)
        x=cv2.waitKey(0)         
        
    if x == ord("d"):
        raiz1(ima1,ima2)
        x=cv2.waitKey(0)  
        
    if x == ord("d"):
        raiz2(ima1,ima2)
        x=cv2.waitKey(0)     
        
    if x == ord("d"):
        potencia1(ima1,ima2)
        x=cv2.waitKey(0)        
                      
    if x == ord("d"):
        potencia2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        potencia3(ima1,ima2)
        x=cv2.waitKey(0) 
        
    if x == ord("d"):
        conjuncion(ima1,ima2)
        x=cv2.waitKey(0)  
        
    if x == ord("d"):
        disyuncion(ima1,ima2)
        x=cv2.waitKey(0)        
        
    if x == ord("d"):
        negacion(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        trasafin(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        escalado(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        rotacion(ima1,ima2)
        x=cv2.waitKey(0)        
        
        cv2.destroyAllWindows()        
    break




'''
#logn metodo 1
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print("logn 1")
logn = np.log(ima1)
print(logn)

cv2.waitKey(0)
cv2.destroyAllWindows()

#raiz meodo 3 
cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
print ('raiz metodo 3')
print (np.sqrt(ima1))
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
cv2.destroyAllWindows()'''
