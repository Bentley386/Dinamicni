"""
Created on Sat Nov 18 10:46:11 2017

@author: Admin
"""
import timeit
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
from scipy.integrate import complex_ode
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi
g=1

def konvert(mu,nu):
    x = g*np.cosh(mu)*np.cos(nu)
    y = g*np.sinh(mu)*np.sin(nu)
    return (x,y)
def konvert2(k,mu,nu):
    x = ((np.sin(nu)**2 + k)/(np.sinh(mu)**2 - k))**0.5
    odvod = (np.cosh(mu)*np.sin(nu)+np.sinh(mu)*np.cos(nu)*x)/(np.sinh(mu)*np.cos(nu)-np.cosh(mu)*np.sin(nu)*x)
    return (1,odvod)        
def trki(xx,yy,vxx,vyy,n,hocem=False,ar=False): #(mu,nu,pmu,pnu)
    x = xx
    y = yy
    vx = vxx
    vy = vyy
    iksi = [x]
    ipsiloni = [y]
    for i in range(n):
        prvi = (vx**2 /(a**2) + vy**2/(b**2))
        drugi = (2*vx*x/(a**2) + 2*vy*y/(b**2))
        tretji = x**2/(a**2) + y**2/(b**2) - 1
        t = (-drugi + np.sqrt(drugi**2 - 4*prvi*tretji))/(2*prvi)
        x = x+t*vx
        iksi.append(x)
        y = y+t*vy
        ipsiloni.append(y)
        N = np.sqrt((x/(a**2))**2 + (y/(b**2))**2)
        nx = -x/a**2/N
        ny = -y/b**2/N
        hitrost = np.asarray([vx,vy])
        normala = np.asarray([nx,ny])
        novahitrost = hitrost - 2*np.sum(hitrost*normala)*normala
        #vx = -3*vx -(-vx*x/a**2/N -vy*y/b**2/N)*(-2*x/a**2/N)
        #vy = -3*vy -(-vx*x/a**2/N -vy*y/b**2/N)*(-2*y/b**2/N)
        vx = novahitrost[0]
        vy = novahitrost[1]
    return (iksi,ipsiloni)
        
                
#plt.plot((0,0.1),(0,-0.8))
#plt.plot((0,-0.3579),(0,0.933))
if 1:
    a = g*np.cosh(1)
    b = g*np.sinh(1)
    koord=konvert(0.999,pi/2)
    hitrost = konvert2(0.4,0.999,pi/2)
    ha = trki(koord[0],koord[1],hitrost[0],hitrost[1],10000)
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(-a,a)
    ax.set_ylim(-b,b)
    #ax.set_xlim(0.5,1)
    #ax.set_ylim(0.5,1)
    ellipse = Ellipse(xy=(0,0), width=2*a,height=2*b,fc="None",edgecolor="k")
    ax.add_patch(ellipse)
    #plt.plot((0,nx),(0,ny))
    #plt.plot((0,vx),(0,vy))
    plt.plot(ha[0],ha[1])
    ax.set_title(r"$k=1.35$")
    plt.tight_layout()
    plt.show()
    #plt.savefig("grafi/9.pdf")
    
    