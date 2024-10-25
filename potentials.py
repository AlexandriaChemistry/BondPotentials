import math
import numpy as np

AU2INVCM = 219476.0
AU2A  = 0.52917721092


def Kratzer(x, *p):

    r = x

    D, r0 = p[:2]

    R = r - r0

    e = D * (R / r)**2

    return e

def Harmonic(x, *p):

    r = x

    kb, r0 = p[:2]

    e = 0.5*kb*(r - r0)**2

    return e

def Lippincott(x, *p):

    r = x

    D, a, r0 = p[:3]

    R = r - r0

    e = D * (1.0 - np.exp(-a * R**2 / (2 * r) ))

    return e

def Deng_Fan(x, *p):

    r = x

    D, a, r0 = p[:3]

    R = r - r0

    e = D * ( (np.exp(a*r) - np.exp(a*r0)) / (np.exp(a*r) - 1.0) )**2

    return e

def Pseudo_Gaussian(x, *p):

    r = x

    D, a, r0 = p[:3]

    R = r - r0

    e = D * ( 1.0 - (1.0 + (a/2.0) * (1.0 - r0**2/r**2)) * np.exp((a/2.0) * (1.0 - r**2/r0**2)   ))

    return e

def Rydberg(x, *p):

    r = x

    D, a, r0 = p[:3]

    R = r - r0

    e = -D * (1.0 + a * R) * np.exp(-a * R) + D
#    e = D * (1.0 - (1.0 + a * R) * np.exp(-a * R))

    return e

def Varshni(x, *p):

    r = x

    D, a, r0 = p[:3]

    r0 = np.abs(r0) # force r0 > 0.0
    R = r - r0

    e = D * (1.0 - r0 * np.exp(-a * (r**2 - r0**2)) /r)**2

    return e

# Morse
def Morse(x,*p):

    r = x

    D, a, r0 = p[:3]

    R = r - r0

    e = D * (1.0 - np.exp(-a * R))**2
#    e =  D * ( np.exp(-2.0 * a * (r-r0)) - 2.0 * np.exp(-a*(r-r0)) + 1.0)

    return e

def Valence_State(x,*p):

    r = x

    D,a,T,C = p[:4]

    e = (T * np.exp(-a * r) - C) / r + D

    return e

def Rosen_Morse(x,*p):

    r = x

    e = p[0] * np.tanh(r/p[1]) - p[2] * (1.0/ np.cosh(r/p[1]))**2 + p[3]

    return e

def Rosen_Morse_try(x,*p):

    r = x

    e = p[0] * np.tanh(r/p[1]) - p[2] * (1.0/ np.cosh(r/p[1]))**2 + p[3]

    return e

def Linnett(x,*p):

    r = x

    e = p[0]/r**3 - p[1] * np.exp(-p[2] * r) + p[3]

    return e

def Poschl_Teller(x,*p):

    r = x
    csch2 = (1.0 / np.sinh(p[1] * r/2.0) )**2
    sech2 = (1.0 / np.cosh(p[1] * r/2.0)  )**2
    e = p[0] * csch2 - p[2] * sech2 + p[3]

    return e

def Frost_Musulin(x,*p):

    r = x
    e = np.exp(-p[0] * r) * (p[1]/r - p[2]) + p[3] 

    return e

def Levine(x,*p):

    r = x

    e = p[0] * (1.0 - p[1] * np.exp(-p[2]*(r**p[3] - p[1]**p[3]))/r  )**2 

    return e

def Wei_Hua(x,*p):

    r = x

    e1 = np.exp(-p[1]*(r-p[2]))
    e = p[0] * ( (1.0 - e1) / (1.0 - p[3] * e1) )**2 
    return e

def Tietz_I(x,*p):

    r = x

    e = p[0] * ( (r - p[1]) / r )**2 * ( (p[2] + p[3] * r) / (p[4] + p[3] * r)) 

    return e

def Rafi(x,*p):

    r = x
    q = np.array( [ 95000.] )
    pp = np.concatenate( ( q , p))
    
    e = p[0]/r**p[1] + p[2] * r * np.exp(-p[3] * r) + p[4]

    return e

def Noorizadeh(x,*p):

    r = x
    
    e = (p[0]*r**p[1] + p[2] ) / ( 1.0 - np.exp(p[3] * r )) + p[4]

    return e

def Tietz_II(x,*p):

    r = x

    e = p[0] + p[0] * ( ( (p[1]+p[2]) * np.exp(-2.0*p[3]*r) - p[2] * np.exp(-p[3]*r)) / (1.0+p[4]*np.exp(-p[3]*r))**2  )

    return e

def Hulburt_Hirschfelder(x,*p):

    r = x
    
    e = p[4] * ( (1 + p[1]*(r - p[0])**3 + p[2]*(r - p[0])**4) * np.exp(-2*p[3]*(r-p[0])) 
                 - 2 * np.exp(-p[3] * (r - p[0]))) + p[4]

    return e

def Murrell_Sorbie(x,*p):

    r = x
    
    e = -p[0] * (1.0 + p[1]*(r - p[4]) + p[2]*(r-p[4])**2 + p[3]*(r-p[4])**3 )*np.exp(-p[1]*(r-p[4])) + p[0]

    return e

def Sun(x,*p):

    r = x
    
    e = -p[0]*p[1]* (1.0/p[1] + p[2]*(r-p[7]) + p[3]*(r-p[7])**2 + p[4]*(r-p[7])**3 + p[5]*(r-p[7])**4 + p[6]*(r-p[7])**5 ) * np.exp(-p[1]*p[2]*(r-p[7])) + p[0]

    return e

# Lennard-Jones
def Lennard_Jones(x,*p):
    r = x

    eps, rm = p[:2]

    eps = np.abs(eps) # De
    rm = np.abs(rm)
   
    e = eps * ( (rm/r)**12 - 2 * (rm/r)**6) + eps

    return e

def Buckingham(x,*p):
    r = x

    eps, sig, gam = p[:3]

    eps = np.abs(eps) # De
    gam = np.abs(gam)
    
    e = eps * ((6/gam)*np.exp(gam*(1 - (r/sig))) - (sig/r)**6) / (1 - (6/gam)) + eps

    return e

def Wang_Buckingham(x, *p):
    r = x
    
    eps, sig, gam = p[:3]

    gam = np.abs(gam)
    
    V = (2 * eps/ (1 - (3/(gam + 3)))) * (sig**6/(sig**6 + r**6)) * ((3/(gam + 3)) * np.exp(gam * (1 - (r/sig))) - 1) + eps
    
    return(V)

def Cahill(x, *p):

    r = x

    a,b,c,d,e,De = p
    De = np.abs(De)
    
    e = a * (1 - c*r) * np.exp(-b * r) - d / (r**6 + e / r**6) + De
    
    return e

# Xie2005a-PRL_95_263202
def J1(gamma,x):
    """
    """
    r = x
    e = np.exp(-2.0 * gamma * r) * (1.0 + 1.0/r)
    return e

def K1(alpha,beta,r):
    e = np.exp(-alpha * r)  * (1.0 - beta * r *r)/r
    return e

def S0(r):
    e = np.exp(-r) * (1.0 + r + r**2/3.0)
    return e

def Xie2005a(x, *p):
    """
    alpha, beta, gamma, De in atomic units

    output is in invcm
    """
    r = x / AU2A
    #r = x
    alpha, beta, gamma, De = p[:4]

    #De = 39420.1
    
    alpha = np.abs(alpha)
    beta = np.abs(beta)
    gamma = np.abs(gamma)
    De = np.abs(De) * AU2INVCM
    #De = 29.172
    #print "#", alpha, beta, gamma, E0 
    #e = De + AU2INVCM * (J1(gamma,r) + K1(alpha,beta,r)) / (1.0 + S0(r))
    e = De + AU2INVCM * (J1(gamma,r) + K1(alpha,beta,r)) / (1.0 + S0(r))
   
    return e

#
# Tang2003a-JCP_118_4976, 
#

#
# Eqn (2)
#
def f_2n(x,n):
    sum = 0.0
    for k in range(0, 2*n+1):
        sum += x**k / math.factorial(k)

    f = 1.0 - np.exp(-x) * sum
    return(f)
#
# Eqn (1)
#
def Tang2003a(x,*p):
    r = x/AU2A
    C_2n = [None]*(10+1) # create empty array for dispersion coeffs (eqn 1)
    eps, C_2n[6], C_2n[8], C_2n[10], A, b = p[:6]

    eps = np.abs(eps)
    C_2n[6] = np.abs(C_2n[6])
    C_2n[8] = np.abs(C_2n[8])
    C_2n[10] = np.abs(C_2n[10])
    
    nmax = 5
    sum = 0.0

    for n in range(3, nmax+1):
        sum += f_2n(b*r,n) * C_2n[2*n] / r**(2*n)
        
    V = A * np.exp(-b*r) - sum + eps

    return(V*AU2INVCM)
