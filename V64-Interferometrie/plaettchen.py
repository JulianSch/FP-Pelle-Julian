import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp

x1, y1=np.genfromtxt('plaettchendaten.txt' ,unpack=True)
x2, y2=np.genfromtxt('plaettchendaten2.txt' ,unpack=True)
x3, y3=np.genfromtxt('plaettchendaten3.txt' ,unpack=True)
x4, y4=np.genfromtxt('plaettchendaten4.txt' ,unpack=True)
x5, y5=np.genfromtxt('plaettchendaten5.txt' ,unpack=True)
x6, y6=np.genfromtxt('plaettchendaten6.txt' ,unpack=True)
x7, y7=np.genfromtxt('plaettchendaten7.txt' ,unpack=True)
x8, y8=np.genfromtxt('plaettchendaten8.txt' ,unpack=True)
xrad1=(((2*np.pi)/360)*x1)
xrad2=(((2*np.pi)/360)*x2)
xrad3=(((2*np.pi)/360)*x3)
xrad4=(((2*np.pi)/360)*x4)
xrad5=(((2*np.pi)/360)*x5)
xrad6=(((2*np.pi)/360)*x6)
xrad7=(((2*np.pi)/360)*x7)
xrad8=(((2*np.pi)/360)*x8)
#p2=y2-y1
#p1=y1
#p3=y3-y2
#p4=y4-y3
#p5=y5-y4
#p6=y6-y5
#p7=y7-y6
#p8=y8-y7
lambdavac=632.99*10**(-9)
delta=10
T=1*10**(-3)
alpha1 = (y1*lambdavac)/(2*T)
alpha2 = (y2*lambdavac)/(2*T)
alpha3 = (y3*lambdavac)/(2*T)
alpha4 = (y4*lambdavac)/(2*T)
alpha5 = (y5*lambdavac)/(2*T)
alpha6 = (y6*lambdavac)/(2*T)
alpha7 = (y7*lambdavac)/(2*T)
alpha8 = (y8*lambdavac)/(2*T)

#dreiformelbrei
n1=((alpha1**2)+2*(1-np.cos(2*delta*xrad1)*(1-alpha1)))/(2*(1-np.cos(2*delta*xrad1-alpha1)))
#n11=(2*delta*xrad1*T)/(2*delta*xrad1*T-y1*lambdavac)
#n111=(1-(y1*lambdavac)/(T*(2*delta*xrad1)**2))**(-1)

n2=((alpha2**2)+2*(1-np.cos(2*delta*xrad2)*(1-alpha2)))/(2*(1-np.cos(2*delta*xrad2-alpha2)))
#n22=(2*delta*xrad2*T)/(2*delta*xrad2*T-y2*lambdavac)
#n222=(1-(y2*lambdavac)/(T*(2*delta*xrad2)**2))**(-1)

n3=((alpha3**2)+2*(1-np.cos(2*delta*xrad1)*(1-alpha3)))/(2*(1-np.cos(2*delta*xrad3-alpha3)))
#n33=(2*delta*xrad3*T)/(2*delta*xrad3*T-y3*lambdavac)
#n333=(1-(y3*lambdavac)/(T*(2*delta*xrad3)**2))**(-1)

n4=((alpha4**2)+2*(1-np.cos(2*delta*xrad4)*(1-alpha4)))/(2*(1-np.cos(2*delta*xrad4-alpha4)))
#n44=(2*delta*xrad4*T)/(2*delta*xrad4*T-y4*lambdavac)
#n444=(1-(y4*lambdavac)/(T*(2*delta*xrad4)**2))**(-1)

n5=((alpha5**2)+2*(1-np.cos(2*delta*xrad5)*(1-alpha5)))/(2*(1-np.cos(2*delta*xrad5-alpha5)))
#n55=(2*delta*xrad5*T)/(2*delta*xrad5*T-y5*lambdavac)
#n555=(1-(y5*lambdavac)/(T*(2*delta*xrad5)**2))**(-1)

n6=((alpha6**2)+2*(1-np.cos(2*delta*xrad6)*(1-alpha6)))/(2*(1-np.cos(2*delta*xrad6-alpha6)))
#n66=(2*delta*xrad6*T)/(2*delta*xrad6*T-y6*lambdavac)
#n666=(1-(y6*lambdavac)/(T*(2*delta*xrad6)**2))**(-1)

n7=((alpha7**2)+2*(1-np.cos(2*delta*xrad7)*(1-alpha7)))/(2*(1-np.cos(2*delta*xrad7-alpha7)))
#n77=(2*delta*xrad7*T)/(2*delta*xrad7*T-y7*lambdavac)
#n777=(1-(y7*lambdavac)/(T*(2*delta*xrad7)**2))**(-1)

n8=((alpha8**2)+2*(1-np.cos(2*delta*xrad8)*(1-alpha8)))/(2*(1-np.cos(2*delta*xrad8-alpha8)))
#n88=(2*delta*xrad8*T)/(2*delta*xrad8*T-y8*lambdavac)
#n888=(1-(y8*lambdavac)/(T*(2*delta*xrad8)**2))**(-1)

#print(n3)
data = [n1,n2,n3,n4,n5,n6,n7,n8]# n3, n4, n5, n6, n7, n8]
for i in data:
    print("Messung")
    print(np.mean(i))
    print(np.std(i))
    print("----------------------")
#print("--------------------")
#bla=10
#blub=50
#ntest=(2*delta*bla*T)/(2*delta*bla*T-blub*lambdavac)
#print('ntest=')
#print(ntest)
#n9=(1-(lambdavac*y4)/(2*T*x4*2*delta))
#print(np.mean(n9))
