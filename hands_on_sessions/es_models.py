import RMK_support.simple_containers as sc
import RMK_support.common_models as cm
import RMK_support.sk_normalization as skn
from RMK_support import RKWrapper

import numpy as np

def addESModels(wrapper:RKWrapper,lMax:int,distFunName:str,eFieldName:str,elTempName:str,elDensName:str) -> None:
    """Adds the terms and models needed for the Epperlein-Short problem

    Args:
        wrapper (RKWrapper): Wrapper to add the models to
        lMax (int): Higest present l harmonic
        distFunName (str): Name of the electron distribution function 
        eFieldName (str): Name of the electric field variable
        elTempName (str): Name of the electron temperature variable
        elDensName (str): Name of the electron density variable
    """

    gridObj = wrapper.grid
    
    wrapper.addSpecies("e", 0)
    wrapper.addSpecies("D+", -1, atomicA=2.014, charge=1.0)

    advModel = cm.kinAdvX(modelTag="adv", distFunName=distFunName, gridObj=gridObj)
    wrapper.addModel(advModel)

    cm.addExAdvectionModel(modelTag="E-adv", distFunName=distFunName, eFieldName=eFieldName+"_dual", wrapper=wrapper, dualDistFun=distFunName+"_dual")

    ampMaxModel = sc.CustomModel(modelTag="AM")

    eTerm = cm.ampereMaxwellKineticElTerm(distFunName, eFieldName+"_dual")

    ampMaxModel.addTerm("eTerm", eTerm)

    wrapper.addModel(ampMaxModel)

    cm.addEECollIsotropic(modelTag="e-e0", distFunName=distFunName, elTempVar=elTempName, elDensVar=elDensName, wrapper=wrapper)

    cm.addStationaryIonEIColl(modelTag="e-i_odd",
                            distFunName=distFunName,
                            ionDensVar=elDensName+"_dual",
                            electronDensVar=elDensName+"_dual",
                            electronTempVar=elTempName+"_dual",
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                            wrapper=wrapper)

    cm.addEECollHigherL(modelTag="e-e_odd",
                        distFunName=distFunName,
                        elTempVar=elTempName+"_dual",
                        elDensVar=elDensName+"_dual",
                        wrapper=wrapper,
                        evolvedHarmonics=list(range(2, gridObj.numH()+1, 2)),
                        dualDistFun=distFunName+"_dual")
    
    if lMax > 1:
        cm.addStationaryIonEIColl(modelTag="e-i_even",
                            distFunName=distFunName,
                            ionDensVar=elDensName,
                            electronDensVar=elDensName,
                            electronTempVar=elTempName,
                            ionSpeciesName="D+",
                            evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)),
                            wrapper=wrapper)

        cm.addEECollHigherL(modelTag="e-e_even",
                        distFunName=distFunName,
                        elTempVar=elTempName,
                        elDensVar=elDensName,
                        wrapper=wrapper,
                        evolvedHarmonics=list(range(3, gridObj.numH()+1, 2)))

    


def kappa0(wrapper:RKWrapper) -> float:
    """Helper function to calculate normalized conductivity proportionality constant

    Args:
        wrapper (RKWrapper): Wrapper used in the problem

    Returns:
        float: Normalized kappa_0
    """
    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    epsilon0 = 8.854188e-12  
    
    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge) 

    sqrt2 = np.sqrt(2)

    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16) + (5629/1152 - 529/128)   
    elCondConst = 125*(1+433*sqrt2/360)/(32*delta)

    skNorms = skn.calculateNorms(wrapper.normalization["eVTemperature"],wrapper.normalization["density"],wrapper.normalization["referenceIonZ"])

    lenNorm = skNorms["length"]
    qNorm = skNorms["heatFlux"]

    normalizationConst = wrapper.normalization["eVTemperature"]**3.5/(lenNorm*qNorm)

    return elCondConst*nConstGradT*normalizationConst