# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
import time
import timeit
from PIL import Image
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg as lin
from scipy.optimize import fsolve
from scipy.linalg import solve
from scipy.linalg import solve_banded
from scipy.special import jn_zeros #prvi parameter je order, drugi št. ničel
from scipy.special import jv #prvi order drugi argument
#from scipy.special import beta
import scipy.special as spec
import scipy.sparse
from scipy.optimize import root
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import complex_ode
from scipy.integrate import simps
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from matplotlib import gridspec
from numba import jit
from scipy import fftpack
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi
pi2 = pi*2

def genKanal(x,a,rand,N):
    if N==0:
        return a
    indices = np.arange(1, N+1)
    return a+np.sum(rand[:,0]*np.cos(indices * x)+rand[:,1]*np.sin(indices*x))
    """
    def aux(i,j):
        if j==0:
            return np.cos((i+1)*x)
        elif j==1:
            return np.sin((i+1)*x)
    return a+np.sum(rand*np.fromfunction(np.vectorize(aux),(N,2)))
    """
def genKanalDer(x,rand,N):
    indices = np.arange(1, N+1)
    return np.sum(-rand[:,0]*indices*np.sin(indices*x) + indices*rand[:,1]*np.cos(indices*x))
    def aux(i,j):
        if j==0:
            return -(i+1)*np.sin((i+1)*x)
        elif j==1:
            return (i+1)*np.cos((i+1)*x)
    return np.sum(rand*np.fromfunction(np.vectorize(aux),(N,2)))
def genRandom(eps,N):
    def aux(i,j):
        return np.random.randn()*eps/(i+1)
    return np.fromfunction(np.vectorize(aux),(N,2))
def genZacetek(v,a,rand,N):
    Kot = 2*pi*np.random.rand()
    Loc = np.random.rand(2)
    while Loc[1]>=genKanal(Loc[0],a,rand,N):
        Loc = np.random.rand(2)
    return np.array([Loc[0],Loc[1],v*np.cos(Kot),v*np.sin(Kot)])
def mestoTrka(predprejsnja,prejsnja,a,rand,N):
    k = predprejsnja[3]/predprejsnja[2]
    n = predprejsnja[1]-k*predprejsnja[0]
    def premica(x):
        return k*x+n
    levi = min((predprejsnja[0],prejsnja[0]))
    desni = max((predprejsnja[0],prejsnja[0]))
    for i in range(2):
        sredina = (levi+desni)/2
        if (premica(sredina)-genKanal(sredina,a,rand,N))*(premica(desni)-genKanal(desni,a,rand,N))<0:
            levi=sredina
        else:
            desni=sredina
    return np.array([sredina,premica(sredina)])  
def dolzina(x,rand,N):
    def integrant(xx):
        return np.sqrt(1+genKanalDer(xx,rand,N)**2)
    return scipy.integrate.quad(integrant,0,x)          
def diffusionJac(t,D,a):
    N = len(t)
    prvi = np.reshape(2*t**a,(N,1))
    drugi = np.reshape(2*D*t**a* np.log(a),(N,1))
    #tretji = np.reshape(np.ones(N),(N,1))
    return np.hstack((prvi,drugi))

def diffusionFit(t,D,a):
    return 2*D*t**a
def simulirajBIS(zac,step,numsteps,a,rand,N,eps):
    #lokacije = np.ones((numsteps//1000+1,3))
    #lokacije = np.ones(numsteps//1000)
    #lokacije[0] = zac[0]
    #lokacije = np.ones((numsteps//1000,3))
    #lokacije[0]=zac[:3]
    lokacije = np.ones((1000,3))
    prejsnja = zac
    predprejsnja = zac
    #lokacije[0]=zac[0]
    #lokacije = np.ones((1000,3))
    counter = 0
    #lokacije2 = []
    for i in range(1,numsteps):
        Upper = genKanal(prejsnja[0],a,rand,N)
        if Upper-prejsnja[1]<0:
            prejsnja[:2] = mestoTrka(predprejsnja,prejsnja,a,rand,N)
            k1 = -1/genKanalDer(prejsnja[0],rand,N)
            k2 = prejsnja[3]/prejsnja[2]
            kot = np.arctan((k2-k1)/(1+k1*k2))
            #lokacije2.append([prejsnja[0],prejsnja[1],prejsnja[2]])
            #lokacije2.append([prejsnja[0],prejsnja[2]])
            rotMatrix = np.array([[np.cos(2*kot),np.sin(2*kot)],[-np.sin(2*kot),np.cos(2*kot)]])
            predprejsnja[2:] = prejsnja[2:]
            prejsnja[2:] = -np.matmul(rotMatrix,prejsnja[2:])#primerno spremeni hitrost       
            lokacije[counter]=prejsnja[:3]
            counter = counter + 1
        elif Upper - prejsnja[1]<eps:
            #lokacije2.append([prejsnja[0],prejsnja[1],prejsnja[2]])
            #lokacije2.append([prejsnja[0],prejsnja[1]])
            k1 = -1/genKanalDer(prejsnja[0],rand,N)
            k2 = prejsnja[3]/prejsnja[2]
            kot = np.arctan((k2-k1)/(1+k1*k2))
            rotMatrix = np.array([[np.cos(2*kot),np.sin(2*kot)],[-np.sin(2*kot),np.cos(2*kot)]])
            predprejsnja[2:] = prejsnja[2:]
            prejsnja[2:] = -np.matmul(rotMatrix,prejsnja[2:])#primerno spremeni hitrost
            lokacije[counter]=prejsnja[:3]
            counter = counter + 1
        elif prejsnja[1]<eps and prejsnja[3]<0:
            #lokacije2.append([prejsnja[0],prejsnja[1],prejsnja[2]])
            #lokacije2.append([prejsnja[0],prejsnja[2]])
            prejsnja[2] = prejsnja[2]
            prejsnja[3] = -prejsnja[3]
            lokacije[counter]=prejsnja[:3]
            counter = counter + 1
        predprejsnja[:2]=prejsnja[:2]
        prejsnja[:2] = prejsnja[:2] + prejsnja[2:]*step
        if counter==1000:
            return np.array(lokacije)
        #if i%1000==0:
            #lokacije2.append([prejsnja[0],prejsnja[1],prejsnja[2]])
            #lokacije[i//1000]=prejsnja[0]
        #lokacije[i]=prejsnja[:2]
    #return lokacije
    print("JA")
    return np.array(lokacije)

def R2(n,observed,predicted):
    mean = 1/n*np.sum(observed)
    tot = np.sum((observed - mean*np.ones(n))**2)
    reg = np.sum((predicted-mean)**2)
    res = np.sum((predicted-observed)**2)
    return 1 - res/tot


if 0:
    #chirikov ekstrapolacija
    epsiloni = [0.00001,0.00005,0.0001,0.0005,0.001]
    delte = np.array([0.006,0.01,0.02,0.09,0.47])
    plt.title("Chirikov - ekstrapolacija z korensko funkcijo")
    plt.plot(epsiloni,delte,"o")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\delta$")
    if 0:
        def funk(x,a,b):
            return a + b*np.sqrt(x)
        def funkJac(x,a,b):
            N = len(x)
            prvi = np.reshape(np.ones(N),(N,1))
            drugi = np.reshape(np.sqrt(x),(N,1))
            return np.hstack((prvi,drugi))
        p = curve_fit(funk,epsiloni,delte,jac=funkJac)[0]
        plt.hlines(0.9,0,0.005)
        plt.text(0.0002,0.95,r"$y=0.9$")
        plt.text(0.0038,2.5,r"$x=0.0045$")
        plt.vlines(0.0045,0,4)
        x = np.linspace(0.00001,0.005,1000)
        plt.plot(x,p[1]*np.sqrt(x) + p[0]*np.ones(1000))
        plt.xlim(0.00001,0.005)
        plt.ylim(0,4)
        vrednosti = np.array([p[1]*np.sqrt(eps) + p[0] for eps in epsiloni])
        #print(np.sum((vrednosti-delte)**2))
        plt.savefig("Chirikovvv.pdf")
    if 0:
        p = np.polyfit(epsiloni,delte,3)
        plt.hlines(0.9,0.00001,0.002)
        plt.text(0.0002,0.95,r"$y=0.9$")
        plt.text(0.00125,2.5,r"$x=0.00124$")
        plt.vlines(0.00124,0,4)
        x = np.linspace(0.00001,0.002,1000)
        plt.plot(x,p[0]*x**3 + p[1]*x**2 + p[2]*x + p[3]*np.ones(1000))
        plt.xlim(0.00001,0.002)
        plt.ylim(0,4)
        vrednosti = np.array([p[0]*eps**3 + p[1]*eps**2 + p[2]*eps + p[3] for eps in epsiloni])
        #print(np.sum((vrednosti-delte)**2))
        plt.savefig("Chirikov.pdf")
    if 1:
        p = np.polyfit(epsiloni,delte,1)
        plt.hlines(0.9,0,0.003)
        plt.text(0.0002,0.95,r"$y=0.9$")
        plt.text(0.0021,1.1,r"$x=0.002$")
        plt.vlines(0.00207,0,1.2)
        x = np.linspace(0.00001,0.003,1000)
        plt.plot(x,p[0]*x + p[1]*np.ones(1000))
        plt.xlim(0.00001,0.003)
        plt.ylim(0,1.2)
        vrednosti = np.array([p[0]*eps + p[1] for eps in epsiloni])
        print(R2(5,delte,vrednosti))
        print(np.sum((vrednosti-delte)**2))
        plt.savefig("Chirikovv.pdf")
if 0:
    #fazni portret za chirikov
    N = 100
    a = 5
    epsiloni = [0.00001,0.00005,0.0001,0.0005, 0.001, 0.002, 0.003]
    fig,ax=plt.subplots()
    zac1 = np.array([np.random.rand()*5,np.random.rand()*2,-0.9,np.sqrt(1-0.9**2)])
    zac2 = np.array([np.random.rand()*5,np.random.rand()*2,0.9,np.sqrt(1-0.9**2)])
    rand = genRandom(1,N)
    barve = ["red","blue"]
    for i in range(len(epsiloni)):
        kurentrand = epsiloni[i]*rand
        lokacije = simulirajBIS(zac1,0.01,2000000,a,kurentrand,N,0.01)
        ax.plot(lokacije[:,0]%pi2,lokacije[:,2],",",color="red")
        lokacije = simulirajBIS(zac2,0.01,2000000,a,kurentrand,N,0.01)
        ax.plot(lokacije[:,0]%pi2,lokacije[:,2],",",color="blue")
        ax.set_ylim(-1,1)
        ax.set_title(r"$\varepsilon = {}$".format(str(epsiloni[i])))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v_x$")
        plt.savefig("Chirikov{}.pdf".format(str(i)))    
if 0:
    #Fazni portret
    N = 100
    n=50
    a = 5
    eps=0.01
    colormap = plt.get_cmap("hsv")
    barve = np.linspace(0,0.95,n)
    fig,ax=plt.subplots()
    vx = np.linspace(-0.9,0.9,n)
    for i in range(1,n+1):
        if i%10==0:
            print(i)
        rand = genRandom(eps,N)
        #zac = genZacetek(1,a,rand,N)
        zac = np.array([np.random.rand()*5,np.random.rand()*2,vx[i-1],np.sqrt(1-vx[i-1]**2)])
        lokacije = simulirajBIS(zac,0.01,2000000,a,rand,N,0.01)
        ax.plot(lokacije[:,0]%pi2,lokacije[:,2],",",color=colormap(barve[i-1]))
    ax.set_title(r"$\varepsilon = 0.01, n=50, N=100, t_t = 1000$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v_x$")
    plt.savefig("Portret01.pdf")
#phase diagram x, vx

def vrednost(x):
    x2 = (x-0.03)/((x-0.03)**2 + (pi/450)**2)
    w02 = 3*10**(-5)*np.sqrt(x2/(x-0.03))
    q1 = pi*w02**2 /(4.05*10**(-7))
    return 1/(pi*4.05*10**(-7))*(q1 + (0.12-x-0.03-x2)**2/q1)
iksi = np.linspace(0.05,0.09,1000)
ipsiloni = [vrednost(x) for x in iksi]
plt.plot(iksi,ipsiloni)
#plt.hlines(9*10**(-10),0.05,0.09)


if 0:
    #difuzija D(eps)
    a = 5
    N=100
    numsteps=1000000
    enke = np.ones(numsteps//1000)
    #Dji = np.ones(9)
    #DjiMax = np.ones(9)
    #DjiMin = np.ones(9)
    Dji = np.array([2072.09, 1418.55, 603.447, 286.845, 162.826, 25.1025, 8.96396, 5.87567, 1.43362,0.7236])
    DjiMax = np.array([2665.88, 1941.6, 844.121, 357.984, 224.192, 34.4191, 11.1984, 8.19081, 1.83226,1.04522])
    DjiMin = np.array([1478.29, 895.512, 362.773, 215.706, 101.461, 15.786, 6.72955, 3.56054, 1.03498,0.402])
    Epsiloni = [0.001,0.0025,0.005,0.0075,0.01,0.025,0.05,0.1,0.75,1]
    #difuzija od eps
    """
    for i in range(9):
        D = []
        for j in range(5):
            print(j)
            hitrosti = []
            for k in range(100):
                rand = genRandom(Epsiloni[i],N)
                zac = genZacetek(1,a,rand,N)
                temp = simulirajBIS(zac,0.01,numsteps,a,rand,N,0.01)
                hitrosti.append((temp-temp[0]*enke)**2)
            hitrosti=np.array(hitrosti)
            x = np.linspace(0,numsteps,numsteps//1000)
            y = np.median(hitrosti,axis=0)
            parametri = curve_fit(diffusionFit,x[800:]/100,y[800:],jac=diffusionJac,method="trf")
            D.append(parametri[0][0])
        povp = sum(D)/5
        odmiki = [abs(d - povp) for d in D]
        napaka = max(odmiki)
        print(povp)
        print(povp+napaka)
        print(povp-napaka)
        print(1/0)
        Dji[i] =povp
        DjiMax[i] =povp+napaka
        DjiMin[i] =povp-napaka
    """
    plt.plot(Epsiloni,Dji,"r")
    plt.plot(Epsiloni,DjiMax,"r--")
    plt.plot(Epsiloni,DjiMin,"r--")
    plt.fill_between(Epsiloni,DjiMin,DjiMax,color=(1,0,0,0.1))
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("D")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.title(r"$D(\varepsilon), N=100$")
    plt.savefig("DodEps2.pdf")
if 0:
    #difuzija odv od N
    a = 5
    eps = 1
    numsteps=1000000
    N = 200
    enke = np.ones(numsteps//1000)
    D = 0
    vx = np.random.rand()
    zac = np.array([np.random.rand(),np.random.rand()*2,vx,np.sqrt(1-vx*vx)])
    for j in range(5):
        hitrosti1 = []
        hitrosti2 = []
        hitrosti3 = []
        for i in range(100):
            if i%10 == 0:
                print(i)
            rand = genRandom(eps,N)
            temp = simulirajBIS(zac,0.01,numsteps,a,rand,N,0.01)
            hitrosti1.append((temp-temp[0]*enke)**2)
            temp2 = simulirajBIS(zac,0.01,numsteps,a,rand[:100],100,0.01)
            hitrosti2.append((temp2-temp2[0]*enke)**2)
            temp3 = simulirajBIS(zac,0.01,numsteps,a,rand[:50],50,0.01)
            hitrosti3.append((temp3-temp3[0]*enke)**2)
        hitrosti1 = np.array(hitrosti1)
        hitrosti2 = np.array(hitrosti2)
        hitrosti3 = np.array(hitrosti3)
        x=np.linspace(0,numsteps,numsteps//1000)
        y1 = np.median(hitrosti1,axis=0)
        y2 = np.median(hitrosti2,axis=0)
        y3 = np.median(hitrosti3,axis=0)
        parametri = curve_fit(diffusionFit,x[800:]/100,y1[800:],jac=diffusionJac,method="trf")
        print("D1 je")
        print(parametri[0])
        parametri = curve_fit(diffusionFit,x[800:]/100,y2[800:],jac=diffusionJac,method="trf")
        print("D2 je")
        print(parametri[0])        
        parametri = curve_fit(diffusionFit,x[800:]/100,y3[800:],jac=diffusionJac,method="trf")
        print("D3 je")
        print(parametri[0])        
if 0:
    #difuzija samo D
    a = 5
    eps = 0.01
    N = 100
    numsteps=1000000
    enke = np.ones(numsteps//1000)
    D = 0
    vx = np.random.rand()
    zac = np.array([np.random.rand(),np.random.rand()*2,vx,np.sqrt(1-vx*vx)])
    print("eps je 0.01, povp. po kanalu:")
    for j in range(5):
        hitrosti = []
        for i in range(100):
            if i%10 == 0:
                print(i)
            rand = genRandom(eps,N)
            temp = simulirajBIS(zac,0.01,numsteps,a,rand,N,0.01)
            hitrosti.append((temp-temp[0]*enke)**2)
        hitrosti = np.array(hitrosti)
        x=np.linspace(0,numsteps,numsteps//1000)
        y = np.median(hitrosti,axis=0)    
        parametri = curve_fit(diffusionFit,x[800:]/100,y[800:],jac=diffusionJac,method="trf")
        print("D je")
        print(parametri[0])
                    
                 
if 0:
    #difuzija
    a = 5
    eps = 1
    N = 100
    fig,ax=plt.subplots()
    numsteps=1000000
    hitrosti = []
    rand = genRandom(eps,N)
    enke = np.ones(numsteps//1000)
    koncne = []
    for i in range(100):
        print(i)
        zac = genZacetek(1,a,rand,N)
        temp = simulirajBIS(zac,0.01,numsteps,a,rand,N,0.01)
        hitrosti.append((temp-temp[0]*enke)**2)
        koncne.append(temp[-1])
    hitrosti = np.array(hitrosti)
    x=np.linspace(0,numsteps,numsteps//1000)
    for i in [25,50,75,100]:
        y = np.median(hitrosti[:i],axis=0)    
        plt.plot(x/100,y,label=r"$n={}$".format(i))
    #plt.plot(x,y)
    plt.legend(loc="best")
    parametri = curve_fit(diffusionFit,x[4000:]/100,y[4000:],jac=diffusionJac,method="trf") 
    plt.plot(x/100,[diffusionFit(i,parametri[0][0],parametri[0][1]) for i in x/100])
    plt.title(r"Difuzija $\varepsilon=1, \sigma^2(t) = 2*{} t^{{{}}}$".format(round(parametri[0][0],4),round(parametri[0][1],2)))
    #plt.xlabel(r"$t$")
    #plt.ylabel(r"$\sigma^2_x$")
    #plt.savefig("difuzija7.pdf")


    

if 0:
    #avtokorelacijska funkcija (za difuzijo)
    fig,ax=plt.subplots()
    numsteps=10000000
    hitrosti = []
    rand = genRandom(1,100)
    #enke = np.ones(numsteps//1000+1)
    koncne = []
    for i in range(50):
        print(i)
        zac = genZacetek(1,5,rand,100)
        temp = simulirajBIS(zac,0.01,numsteps,5,rand,100,0.01)
        hitrosti.append(temp)
    hitrosti = np.array(hitrosti)
    x=np.linspace(0,numsteps,numsteps//1000+1)
    for i in [10,20,30,40,50]:
        #y0 = np.reshape(hitrosti[:i][:,0],(i,1))
        #y = np.sum(np.abs(hitrosti[:i]*y0),axis=0)/i
        y = np.sum(np.abs(hitrosti[:i]),axis=0)/i    
        kor = np.correlate(y,y,mode="same")
        plt.plot(x[:(len(x)//2)]/100,kor[(len(kor)//2):-1]/kor[len(kor)//2],label=r"$n={}$".format(i))
    plt.legend(loc="best")
    plt.title(r"Avtokorelacijska funkcija $\langle x(t) x(0) \rangle$")
    #D = y[0]/2 + np.sum(y[1:])
    plt.savefig("avtokorel2.pdf")

    
if 0:
    #simulacija
    N = 100
    eps = 1
    a = 5
    rand = genRandom(eps,N)
    zac = genZacetek(1,a,rand,N)
    lokacije = simulirajBIS(zac,0.01,2000000,a,rand,N,0.01)
    plt.plot(lokacije[:,0],lokacije[:,1],"r")
    xlim=np.amax(np.abs(lokacije[:,0]))
    x = np.linspace(-xlim,xlim,30000)
    plt.plot(x,np.zeros(30000),"k")
    plt.plot(x,[genKanal(i,a,rand,N) for i in x],"k")
    #plt.title(r"$\varepsilon=1, a=10$")
    #plt.savefig("gibanje6.pdf")
    ##plt.show()
if 0:
    #sim + fazni portret
    N = 100
    eps = 0.01
    a = 5
    rand = genRandom(eps,N)
    zac = genZacetek(1,a,rand,N)
    lokacije = simulirajBIS(zac,0.01,500000,a,rand,N,0.01)
    fig, axes = plt.subplots(2,3,figsize=(19.2,9.6))
    n = len(lokacije)
    nicle = np.zeros(1000)
    deli = [n//100,n//10,n]
    for k in range(3):
        relevantno = lokacije[:deli[k]]
        xmax = np.amax(relevantno[:,0])
        x = np.linspace(-xmax,xmax,1000)
        axes[0,k].plot(x,nicle,"k")
        zid = np.array([genKanal(i,a,rand,N) for i in x])
        axes[0,k].plot(x,zid,"k")
        axes[0,k].set_xlim(-xmax,xmax)
        axes[0,k].set_ylim(0,np.amax(zid))
        axes[0,k].plot(relevantno[:,0],relevantno[:,1],color="purple")
        axes[1,k].plot(relevantno[:,0],relevantno[:,2],".",color="magenta")
        axes[1,k].set_xlim(-xmax,xmax)
        axes[1,k].set_xlabel(r"$x$")
        axes[1,k].set_ylabel(r"$v_x$")
        
    plt.suptitle(r"$a = {}, \varepsilon = {}$".format(a,eps),size=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig("oboje6.pdf")    
if 0:
    #animacija  faznega portreta
    N = 100
    n=50 
    epsiloni = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01] #0.01 do 0.1
    a = 5
    colormap = plt.get_cmap("hsv")
    barve = np.linspace(0,0.95,n)
    vx = np.linspace(-0.9,0.9,n)
    fig, ax = plt.subplots()    
    rezultati = []
    for eps in epsiloni:
        print(eps)
        trenutni = []
        for i in range(1,n+1):
            rand = genRandom(eps,N)
            zac = np.array([np.random.rand()*5,np.random.rand()*2,vx[i-1],np.sqrt(1-vx[i-1]**2)])
            lokacije = simulirajBIS(zac,0.01,2000000,a,rand,N,0.01)
            trenutni.append([lokacije[:,0]%pi2,lokacije[:,2]])
        rezultati.append(trenutni)
    def animiraj(t):
        print(t)
        ax.clear()
        eps = epsiloni[t]
        plt.suptitle(r"$\varepsilon = {}$".format(eps))
        for i in range(1,n+1):
            ax.plot(rezultati[t][i-1][0],rezultati[t][i-1][1],",",color=colormap(barve[i-1]))
    ani = animation.FuncAnimation(fig,animiraj,range(19),interval=500)   
    #plt.show()
    ani.save("Portret.mp4")
    
    
if 0:
    #animacija gibanja biljard + PP
    N = 100
    eps = 1
    a = 10
    rand = genRandom(eps,N)
    #zac = genZacetek(1,a,rand,N)
    iksi = []
    ipsiloni = []
    hitrosti=[]
    for i in range(6):
        zac = np.array([i/20,0.1,0.1,np.sqrt(1-0.1*0.1)])
        lokacije = simulirajBIS(zac,0.01,50000,a,rand,N,0.01) #arej nefiksne dolzine
        iksi.append(lokacije[:,0])
        ipsiloni.append(lokacije[:,1])
        hitrosti.append(lokacije[:,2])
    iksi = np.array(iksi)
    ipsiloni=np.array(ipsiloni)
    hitrosti= np.array(hitrosti)
    #plt.plot(lokacije[:,0],lokacije[:,1],"r")
    xlim=np.amax(np.abs(iksi))
    #hitrostilim=np.amax(np.abs(hitrosti))
    x = np.linspace(-xlim,xlim,30000)
    y = [genKanal(i,a,rand,N) for i in x]
    numiter=len(iksi[0])
    print(numiter)
    nicle = np.zeros(30000)
    #plt.title(r"$\varepsilon=1, a=10$")
    barve=["b","g","r","c","m","y"]
    fig, ax1, ax2 = plt.subplots(2,figsize=(10,10))    
    def animiraj(t):
        print(t)
        ax1.clear()
        ax2.clear()
        ax1.plot(x,nicle,"k")
        ax1.plot(x,y,"k")
        if t!=0:
            for i in range(6):
                ax1.plot(iksi[i][:t],ipsiloni[i][:t],color=barve[i])
        for i in range(6):
            ax1.plot(iksi[i][t],ipsiloni[i][t],".",color=barve[i])
        if t!=numiter-1:
            for i in range(6):
                ax2.plot(iksi[i][:(t+1)],hitrosti[i][:(t+1)],".",color=barve[i])
    ani = animation.FuncAnimation(fig,animiraj,range(numiter),interval=100)   
    #plt.show()
    ani.save("siminpp.mp4")
    print(1/0)   
    
if 0:
    #animacija gibanja samo biljard
    N = 100
    eps = 0.1
    a = 5
    rand = genRandom(eps,N)
    #zac = genZacetek(1,a,rand,N)
    iksi = []
    ipsiloni = []
    zac = np.array([[i/20,0.1,0.1,np.sqrt(1-0.1*0.1)] for i in range(10)])
    barve = plt.get_cmap("gist_rainbow")
    barveI = np.linspace(0,1,10)
    for i in range(10):
        lokacije = simulirajBIS(zac[i],0.01,25000,a,rand,N,0.01)
        iksi.append(lokacije[:,0])
        ipsiloni.append(lokacije[:,1])
    iksi = np.array(iksi)
    ipsiloni=np.array(ipsiloni)
    #plt.plot(lokacije[:,0],lokacije[:,1],"r")
    xlim1=np.amin(iksi)
    xlim2=np.amax(iksi)
    x = np.linspace(xlim1,xlim2,30000)
    y = [genKanal(i,a,rand,N) for i in x]
    numiter=len(iksi[0])
    print(numiter)
    nicle = np.zeros(30000)
    #plt.title(r"$\varepsilon=1, a=10$")
    fig, ax = plt.subplots()    
    def animiraj(t):
        print(t)
        ax.clear()
        ax.plot(x,nicle,"k")
        ax.plot(x,y,"k")
        ax.set_title(r"$\varepsilon=0.1$")
        for i in range(10):
            ax.plot(iksi[i][t-1],ipsiloni[i][t-1],".",color=barve(barveI[i]))
            #if t!=0:
                #ax.plot(iksi[i][:t],ipsiloni[i][:t],"r")
    ani = animation.FuncAnimation(fig,animiraj,range(numiter),interval=10)   
    #plt.show()
    ani.save("GibanjeVec2.mp4")
    print(1/0)   
if 0:
    #izbira Nja
    numSim=100
    AbsOdmik = np.zeros((10,499))
    AbsVelikost = np.zeros((10,499))
    epsiloni = np.linspace(0.1,2,10)
    for eps in range(10):
        print(eps)
        for i in range(numSim):
            rand = genRandom(epsiloni[eps],500)
            AbsVelikostTemp = np.zeros(100)
            for j in range(1,500):
                Temp = rand[j][0]*np.cos((j+1)*np.linspace(0,2*pi,100))+rand[j][1]*np.sin((j+1)*np.linspace(0,2*pi,100))
                AbsOdmik[eps][j-1] = AbsOdmik[eps][j-1] + np.sum(np.abs(Temp))
                AbsVelikostTemp = AbsVelikostTemp + Temp
                AbsVelikost[eps][j-1] = AbsVelikost[eps][j-1] + np.sum(np.abs(AbsVelikostTemp))
        AbsOdmik[eps] = AbsOdmik[eps]/numSim*2*pi/100
        AbsVelikost[eps] = AbsVelikost[eps]/numSim*2*pi/100
    colormap = np.linspace(0.15,0.8,10)
    cmap = plt.get_cmap("hot")
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax1.set_title(r"Geometrija biljarda v odvisnosti od N - $\delta S(N)$")
    ax1.set_ylabel(r"$\delta S$")
    ax1.set_xlabel("N")
    ax2.set_title(r"Geometrija biljarda v odvisnosti od N - $S'(N)$")
    ax2.set_ylabel("S'")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    for i in range(10):
        if i==0:
            ax1.plot(range(1,500),AbsOdmik[i],color=cmap(colormap[i]),label=r"$\varepsilon = 0.1$")
            ax2.plot(range(1,500),AbsVelikost[i],color=cmap(colormap[i]),label=r"$\varepsilon = 0.1$")
            continue
        elif i==9:
            ax1.plot(range(1,500),AbsOdmik[i],color=cmap(colormap[i]),label=r"$\varepsilon = 2$")
            ax2.plot(range(1,500),AbsVelikost[i],color=cmap(colormap[i]),label=r"$\varepsilon = 2$")
            continue
        ax1.plot(range(1,500),AbsOdmik[i],color=cmap(colormap[i]))
        ax2.plot(range(1,500),AbsVelikost[i],color=cmap(colormap[i]))
    ax1.legend(loc="lower center")
    ax2.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig("geom3.pdf")
    

if 0:
    #animacija kanala
    fig, ax = plt.subplots()    
    x = np.linspace(-5,5,1000)
    y = np.zeros(1000)
    stevila = genRandom(0.01,100)
    zidi = np.array([[genKanal(i,4,stevila[:j],j) for i in x] for j in range(101)])
    zgornjameja = np.amax(zidi)
    def animiraj(t):
        print(t)
        ax.clear()
        ax.plot(x,y,"k")
        ax.plot(x,zidi[t],"k")
        ax.set_ylim(0,zgornjameja)
        ax.set_title(r"$n={}, \varepsilon = 0.01$".format(str(t)))
    ani = animation.FuncAnimation(fig,animiraj,range(101),interval=100)   
    #plt.show()
    ani.save("Stena001epst.mp4")
    print(1/0)   
if 0:
    #bolj fensi plot kanala
    eps = 0.1
    N = 100
    a = 0.5
    x = np.linspace(-5,5,1000)
    y = np.zeros(1000)
    stevila = [genRandom(eps,N) for i in range(3)]
    zid = [np.array([genKanal(j,a,stevila[i],N) for j in x]) for i in range(3)]
    zgornjameja = np.amax(zid)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(19.2,4.8))
    plt.suptitle(r"$a = {}, \varepsilon = {}$".format(a,eps),size=24)
    ax1.plot(x, y,"k")
    ax1.plot(x,zid[0],"k")
    ax2.plot(x, y,"k")
    ax2.plot(x,zid[1],"k")
    ax3.plot(x, y,"k")
    ax3.plot(x,zid[2],"k")
    ax1.set_xlim(-5,5)
    ax2.set_xlim(-5,5)
    ax3.set_xlim(-5,5)
    ax1.set_ylim(0,zgornjameja)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("zid5.pdf")
if 0:
    #plot kanala
    eps = 0.0001
    N=100
    a = 5
    x = np.linspace(-5,5,1000)
    y = np.zeros(1000)
    stevila = genRandom(eps,N)
    zid = np.array([genKanal(i,a,stevila,N) for i in x])
    zgornjameja = np.amax(zid)
    plt.ylim(0,zgornjameja)
    plt.xlim(-5,5)
    plt.plot(x,y,"k")
    plt.plot(x,zid,"k")
    plt.title(r"$a = {}, \varepsilon = {}$".format(a,eps))
    plt.tight_layout()
    plt.savefig("zid6.pdf")

def simulirajOLD(zac,step,numsteps,a,rand,N,eps,cd = 5):
    #lokacije = np.ones((numsteps,4))
    #lokacije[0]=zac
    prejsnja = zac
    lokacije2 = []
    zadet=0
    for i in range(1,numsteps):
        Upper = genKanal(prejsnja[0],a,rand,N)
        if Upper - prejsnja[1]<0 or (Upper - prejsnja[1]<eps and zadet==0):
            lokacije2.append([prejsnja[0],prejsnja[2]])
            k1 = -1/genKanalDer(prejsnja[0],rand,N)
            k2 = prejsnja[3]/prejsnja[2]
            kot = np.arctan((k2-k1)/(1+k1*k2))
            rotMatrix = np.array([[np.cos(2*kot),np.sin(2*kot)],[-np.sin(2*kot),np.cos(2*kot)]])
            #print("Zadetek k1 {} k2 {} vekt ({} {})".format(k1,k2,lokacije[i-1][2],lokacije[i-1][3]))
            prejsnja[2:] = -np.matmul(rotMatrix,prejsnja[2:])#primerno spremeni hitrost
            zadet=cd
        elif prejsnja[1]<eps and prejsnja[3]<0:
            lokacije2.append([prejsnja[0],prejsnja[2]])
            prejsnja[2] = prejsnja[2]
            prejsnja[3] = -prejsnja[3]
            #print("tla")
            if zadet!=0:
                zadet-=1
        else:
            if zadet!=0:
                zadet-=1
        prejsnja[:2] = prejsnja[:2] + prejsnja[2:]*step
        #lokacije[i]=prejsnja
    return np.array(lokacije2)
    #return lokacije
    
