import numpy as np

from RMK_support import Node, atan, abs, log, sign

def beta(Ts:str|float,ms:float) -> Node|float:
    """Function which calculates the inverse thermal velocity used in anisotropic calculations. Normalised to the elctron mass

    Args:
        Ts (str | float): Temperature of species s (could be parallel or perpendicular)
        ms (float): The mass of a particle of species s

    Returns:
        Node|float: Inverse thermal velocity of the particle of species s, normalised to the elctron mass
    """
    elMass = 9.10938e-31
    if isinstance(Ts,str):
        return  ms/elMass*(Node(Ts))**-1
    return ms/(elMass*Ts)

def betasr(Ts:str|float,Tr:str|float,ms:float,mr:float) -> Node|float:
    """Calculates the harmonic mean of the inverse thermal velocities of species s and r
    NOTE that there is no specification of whether the parallel or perpendicular temperature is used, care should be taken to ensure that the same direction is used for both species

    Args:
        Ts (str | float): Temperature of species s (could be parallel or perpendicular)
        Tr (str | float): Temperature of species r (could be parallel or perpendicular)
        ms (float): The mass of a particle of species s
        mr (float): The mass of a particle of species r

    Returns:
        Node|float: The harmonic mean of the inverse thermal velocities of the two species s and r
    """
    return (beta(Ts,ms)**-1 + beta(Tr,mr)**-1)**-1

def alphasr(TsPar:str|float,TsPerp:str|float,TrPar:str|float,TrPerp:str|float,ms:float,mr:float) -> Node|float:
    """Calculates the degree of anisotropy of the two species s and r, through the ratio of the harmonic mean of the inverse thermal velocities of species s and r

    Args:
        TsPar (str | float): Parallel temperature of species s
        TsPerp (str | float): Perpendicular temperature of species s
        TrPar (str | float): Parallel temperature of species r
        TrPerp (str | float): Perpendicular temperature of species r
        ms (float): The mass of a particle of species s
        mr (float): The mass of a particle of species r

    Returns:
        Node|float: The degree of anisotropy of the species s and r
    """
    return betasr(TsPar,TrPar,ms,mr)/betasr(TsPerp,TrPerp,ms,mr)

def X(alpha:Node|float) -> Node|float:
    """Calculates the primary variable X, which the solutions to bi-Maxwellian type integrals of the form K_LMN are functions of

    Args:
        alpha (Node | float): degree of anisotropy of the species s and r

    Returns:
        Node|float: The primary variable X
    """
    return alpha - 1

def phiPositive(X:Node|float) -> Node|float:
    """Calculates the function phi, which the solutions to bi-Maxwellian type integrals of the form K_LMN contain, for positive values of X

    Args:
        X (Node | float): The primary variable X, which the solutions to bi-Maxwellian type integrals of the form K_LMN are functions of

    Returns:
        Node|float: The function phi, for positive values of X
    """
    if isinstance(X,Node):
        return atan(abs(X)**0.5)*abs(X)**-0.5
    return np.arctan(np.abs(X)**0.5)*np.abs(X)**-0.5

def phiNegative(X:Node|float) -> Node|float:
    """Calculates the function phi, which the solutions to bi-Maxwellian type integrals of the form K_LMN contain, for negative values of X

    Args:
        X (Node | float): The primary variable X, which the solutions to bi-Maxwellian type integrals of the form K_LMN are functions of

    Returns:
        Node|float: The function phi, for negative values of X
    """
    if isinstance(X,Node):
        return log(abs((1+abs(X)**0.5)/(1-abs(X)**0.5)))*0.5*abs(X)**-0.5
    return np.log((1+np.sqrt(np.abs(X)))/(1-np.sqrt(np.abs(X))))/(2*np.sqrt(np.abs(X)))

def stepFunc(X:Node|float) -> Node|float:
    """Creates a step function centred at X = 0. Used to combine the positive and negative forms of the function phi, which are each only valid for values of X with the respective sign

    Args:
        X (Node | float): The primary variable X, the sign of which determines the form of the function phi

    Returns: Node|float: Step function centred at X = 0
    """
    if isinstance(X,Node):
        return 0.5 + 0.5*sign(X)
    return np.heaviside(X,0.5)

def phi(X:Node|float) -> Node|float:
    """Calculates the function phi, which the solutions to bi-Maxwellian type integrals of the form K_LMN contain, for all values of X

    Args:
        X (Node | float): The primary variable X, which the solutions to bi-Maxwellian type integrals of the form K_LMN are functions of

    Returns:
        Node|float: The function phi, for all values of X
    """
    return phiPositive(X)*stepFunc(X) + phiNegative(X)*stepFunc(-X)

def K_LMN(alpha:Node|float,LMN:str,smallX:bool=False) -> Node|float|ValueError:
    """Calculates the variables which are solutions to bi-Maxwellian type integrals, see Chodura & Pohl 1971 for more info.

    Args:
        alpha (Node | float): The degree of anisotropy
        LMN (str): The particular bi-Maxwellian type integral being calculated
        smallX (bool, optional): An optional argument to use the taylor expansion of the particular K_LMN variable (defaults to False)

    Raises:
        ValueError: Unknown K_LMN case provided

    Returns:
        Node|float|ValueError: The particular K_LMN variable
    """
    if LMN == "200":
        return 2/3 - 2/15*X(alpha) if smallX else X(alpha)**-1*(-1 + (1 + X(alpha))*phi(X(alpha)))
    if LMN == "002":
        return 2/3 - 2/5*X(alpha) if smallX else X(alpha)**-1*2*(1 - phi(X(alpha)))
    raise ValueError("Unknown K_LMN case")