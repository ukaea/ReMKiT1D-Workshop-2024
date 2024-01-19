import numpy as np

from RMK_support import Node, atan, abs, log, sign

def beta(Ts,ms:float):
    elMass = 9.10938e-31
    if isinstance(Ts,str):
        return  ms/elMass*(Node(Ts))**-1
    return ms/(elMass*Ts)

def betasr(Ts,Tr,ms:float,mr:float):
    return (beta(Ts,ms)**-1 + beta(Tr,mr)**-1)**-1

def alphasr(TsPar,TsPerp,TrPar,TrPerp,ms,mr):
    return betasr(TsPar,TrPar,ms,mr)/betasr(TsPerp,TrPerp,ms,mr)

def phiPositive(X):
    if isinstance(X,Node):
        return atan(abs(X)**0.5)*abs(X)**-0.5
    return np.arctan(np.abs(X)**0.5)*np.abs(X)**-0.5

def phiNegative(X):
    if isinstance(X,Node):
        return log(abs((1+abs(X)**0.5)/(1-abs(X)**0.5)))*0.5*abs(X)**-0.5
    return np.log((1+np.sqrt(np.abs(X)))/(1-np.sqrt(np.abs(X))))/(2*np.sqrt(np.abs(X)))

def stepFunc(X):
    if isinstance(X,Node):
        return 0.5 + 0.5*sign(X)
    return np.heaviside(X,0.5)

def phiExpansionCoeffs(n):
    return [(-1)**k/(2*k+1) for k in range(1,n)]

def phi(X):
    return phiPositive(X)*stepFunc(X) + phiNegative(X)*stepFunc(-X)

def X(alpha):
    return alpha - 1

def K_LMN(alpha,LMN:str,smallX=False):
    if LMN == "200":
        return 2/3 - 2/15*X(alpha) if smallX else X(alpha)**-1*(-1 + (1 + X(alpha))*phi(X(alpha)))
    if LMN == "002":
        return 2/3 - 2/5*X(alpha) if smallX else X(alpha)**-1*2*(1 - phi(X(alpha)))
    raise ValueError("Unknown K_LMN case")