import numpy as np

import RMK_support.simple_containers as sc
import RMK_support.common_models as cm
from RMK_support import RKWrapper, Grid
import RMK_support.sk_normalization as skn


def generatorSimpleFluid(**kwargs) -> RKWrapper:
    """Generate a wrapper containing a simple 2-fluid model with effective sources for hands-on session 2.5. Does not set time integration options. 

    The upstream (x=0) boundary is reflective, and outflow Bohm boundary conditions are applied to the downstream/sheath boundary.

    The particle, momentum, and energy equations are solved for the electrons and deuterium ions. The heat flux, friction, and 
    energy exchange terms come from the standard Braginskii closure. The initial temperature and density profiles are determined 
    using a Two Point Model profile defined by the upstream and downstream temperature values, as well as the upstream density value.

    The heating source is uniform over a number of upstream cells. Recycling is set to 100%, so particle sources are not needed. 
    Recycled neutrals are assumed immediately ionized, resulting in a particle source and energy sink with profile shape exp((x-L)/ion_mfp). 
    The energy cost per ionization event (in eV) can be explicitly set. 

    The following kwargs are available:

    hdf5OutputFolder (str): Session output folder in the RMKOutput directory. Defaults to "day_2_5_1".
    mpiProcsX (int): Number of MPI processes in the x direction. Defaults to 8.
    Nx (int): Number of spatial cells. Defaults to 64.
    domainLength (int): Length of grid in x direction in metres. Defaults to 10m.
    Tu (float): Initial upstream temperature in eV. Defaults to 20.
    Td (float): Initial downstream temperature in eV. Defaults to 5.
    nu (float): Initial upstream density in normalized units. Automatically determines the downstream density using a constant pressure 
                assumption. Defaults to 0.8. 
    Nh (int): Number of upstream cells affected by the heating operator. Defaults to 1.
    heatingPower (float): Effective upstream heating in MW/m^2. Defaults to 1.0.
    ionMFP (float): Ion mean free path in metres. Defaults to 0.5m.
    ionCost (float): Energy cost per ionization event in eV. Defaults to 20. 

    Returns:
        RKWrapper: ReMKiT1D wrapper containing initialization data for this run.
    """

    elCharge = 1.60218e-19
    elMass = 9.10938e-31
    amu = 1.6605390666e-27  # atomic mass unit
    ionMass = 2.014*amu  # deuterium mass
    epsilon0 = 8.854188e-12  # vacuum permittivity

    rk = RKWrapper()
    rk.setHDF5Path("./RMKOutput/"+kwargs.get("hdf5OutputFolder","day_2_5_1")+"/")

    numProcsX = kwargs.get("mpiProcsX",8)  # Number of processes in x direction
    numProcsH = 1  # Number of processes in harmonic
    numProcs = numProcsX * numProcsH
    haloWidth = 1  # Halo width in cells

    rk.setMPIData(numProcsX, numProcsH, haloWidth)

    rk.setNormDensity(1.0e19)
    rk.setNormTemperature(10.0)
    rk.setNormRefZ(1.0)

    tempNorm = rk.normalization["eVTemperature"] 
    densNorm = rk.normalization["density"]
    skNorms = skn.calculateNorms(tempNorm,densNorm,1)
    
    timeNorm = skNorms["time"]
    lengthNorm = skNorms["length"]

    Nx = kwargs.get("Nx",64)
    L = kwargs.get("domainLength",10) # Length in meters
    xGridWidths = L/Nx*np.ones(Nx)
    gridObj = Grid(xGridWidths, interpretXGridAsWidths=True, isLengthInMeters=True)

    # Add the grid to the config file
    rk.grid = gridObj

    rk.addSpecies("e", 0, atomicA=elMass/amu, charge=-1.0, associatedVars=["ne", "Ge", "We"])
    rk.addSpecies("D+", -1, atomicA=2.014, charge=1.0, associatedVars=["ni", "Gi", "Wi"])

    electronSpecies = rk.getSpecies("e")
    ionSpecies = rk.getSpecies("D+")

    rk.addCustomDerivation("linExtrapRight",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),ignoreUpperBound=True))

    rk.addCustomDerivation("linExtrapRightLB",sc.boundedExtrapolationDerivation(sc.linExtrapolation(),expectLowerBoundVar=True,ignoreUpperBound=True))

    rk.addCustomDerivation("boundaryFlux",sc.multiplicativeDerivation("linExtrapRight",innerDerivationIndices=[1],outerDerivation="linExtrapRightLB",outerDerivationIndices=[2,3]))

    # Two-point model initialization

    Tu = kwargs.get("Tu",20)/tempNorm #upstream temperature
    Td = kwargs.get("Td",5)/tempNorm #downstream temperature

    T = (Tu**(7/2) - (Tu**(7/2)-Td**(7/2))*gridObj.xGrid/L)**(2/7)

    nu = kwargs.get("nu",0.8) #upstream density

    n = nu*Tu/T

    W = 3*n*T/2
    # Set conserved variables in container

    # Units are not used by ReMKiT1D, but are useful to specify for later plotting
    rk.addVarAndDual("ne", n, units='$10^{19} m^{-3}$', isCommunicated=True)
    rk.addVarAndDual("ni", n, units='$10^{19} m^{-3}$', isCommunicated=True)
    rk.addVarAndDual("Ge", primaryOnDualGrid=True, isCommunicated=True)  # Ge_dual is evolved, and Ge is derived
    rk.addVarAndDual("Gi", primaryOnDualGrid=True, isCommunicated=True)
    rk.addVarAndDual("We", W, units='$10^{20} eV m^{-3}$', isCommunicated=True)
    rk.addVarAndDual("Wi", W, units='$10^{20} eV m^{-3}$', isCommunicated=True)

    # Temperatures
    rk.addVarAndDual("Te", T, isStationary=True, units='$10eV$', isCommunicated=True)
    rk.addVarAndDual("Ti", T, isStationary=True, units='$10eV$', isCommunicated=True)

    # Set heat fluxes

    rk.addVarAndDual("qe", isStationary=True, primaryOnDualGrid=True, isCommunicated=True)
    rk.addVarAndDual("qi", isStationary=True, primaryOnDualGrid=True, isCommunicated=True)

    # Set E field

    rk.addVarAndDual("E", primaryOnDualGrid=True)

    # Set derived fluid quantities

    rk.addVarAndDual("ue", isDerived=True, primaryOnDualGrid=True, derivationRule=sc.derivationRule(
        "flowSpeedFromFlux", ["Ge_dual", "ne_dual"]), isCommunicated=True)
    rk.addVarAndDual("ui", isDerived=True, primaryOnDualGrid=True, derivationRule=sc.derivationRule(
        "flowSpeedFromFlux", ["Gi_dual", "ni_dual"]), isCommunicated=True)
    rk.addVar("cs", isDerived=True, derivationRule=sc.derivationRule("sonicSpeedD+", ["Te", "Ti"]))

    # Set scalar quantities
    rk.addVar("time", isScalar=True, isDerived=True)
    rk.addVar("gammaRight", isScalar=True, isDerived=True, derivationRule=sc.derivationRule(
        "rightElectronGamma", ["Te", "Ti"]), isCommunicated=True, hostScalarProcess=numProcs-numProcsH)

    rk.addVar("cs_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
            ,derivationRule=sc.derivationRule("linExtrapRight",["cs"]))

    rk.addVar("n_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
            ,derivationRule=sc.derivationRule("linExtrapRight",["ne"]))

    rk.addVar("G_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
            ,derivationRule=sc.derivationRule("boundaryFlux",["ni","ui","cs_b"]))

    rk.addVar("u_b",isDerived=True,isScalar=True,isCommunicated=True,hostScalarProcess=numProcs-numProcsH
            ,derivationRule=sc.derivationRule("flowSpeedFromFlux",["G_b","n_b"]))

    ionGamma = 2.5*np.ones([1])  # Scalar variables must be specified as a length 1 numpy array
    rk.addVar("ionGamma", ionGamma, isScalar=True, isDerived=True, outputVar=False)

    # Electron continuity advection

    rk.addModel(cm.staggeredAdvection(modelTag="continuity-ne", 
                                      advectedVar="ne",
                                      fluxVar="Ge_dual",
                                      advectionSpeed="ue",
                                      lowerBoundVar="cs",
                                      rightOutflow=True
                                      )
                )

    # Ion continuity advection

    rk.addModel(cm.staggeredAdvection(modelTag="continuity-ni",
                                      advectedVar="ni",
                                      fluxVar="Gi_dual",
                                      advectionSpeed="ui",
                                      lowerBoundVar="cs",
                                      rightOutflow=True
                                      )
                )

    # Electron pressure grad

    rk.addModel(cm.staggeredPressureGrad(modelTag="pressureGrad-Ge",
                                         fluxVar="Ge_dual",
                                         densityVar="ne",
                                         temperatureVar="Te",
                                         speciesMass=elMass
                                         )
                )

    # Ion pressure grad

    rk.addModel(cm.staggeredPressureGrad(modelTag="pressureGrad-Gi",
                                         fluxVar="Gi_dual",
                                         densityVar="ni",
                                         temperatureVar="Ti",
                                         speciesMass=ionMass
                                         )
                )

    # Electron momentum advection

    rk.addModel(cm.staggeredAdvection(modelTag="advection-Ge",
                                      advectedVar="Ge_dual",
                                      fluxVar="",
                                      advectionSpeed="ue",
                                      staggeredAdvectionSpeed="ue_dual",
                                      lowerBoundVar="cs",
                                      rightOutflow=True,
                                      staggeredAdvectedVar=True
                                      )
                )

    # Ion momentum advection

    rk.addModel(cm.staggeredAdvection(modelTag="advection-Gi",
                                      advectedVar="Gi_dual",
                                      fluxVar="",
                                      advectionSpeed="ui",
                                      staggeredAdvectionSpeed="ui_dual",
                                      lowerBoundVar="cs",
                                      rightOutflow=True,
                                      staggeredAdvectedVar=True
                                      )
                )

    # Ampere-Maxwell E field equation

    rk.addModel(cm.ampereMaxwell(modelTag="ampereMaxwell",
                                 eFieldName="E_dual",
                                 speciesFluxes=["Ge_dual", "Gi_dual"],
                                 species=[electronSpecies, ionSpecies]
                                 )
                )

    # Lorentz force terms

    rk.addModel(cm.lorentzForces(modelTag="lorentzForce",
                                        eFieldName="E_dual",
                                        speciesFluxes=["Ge_dual", "Gi_dual"],
                                        speciesDensities=["ne_dual", "ni_dual"],
                                        species=[electronSpecies, ionSpecies]
                                        )
                )

    # Implicit temperature equations

    rk.addModel(cm.implicitTemperatures(modelTag="implicitTemp",
                                        speciesFluxes=["Ge_dual", "Gi_dual"],
                                        speciesDensities=["ne", "ni"],
                                        speciesEnergies=["We", "Wi"],
                                        speciesTemperatures=["Te", "Ti"],
                                        species=[electronSpecies, ionSpecies],
                                        speciesDensitiesDual=["ne_dual", "ni_dual"]
                                        )
                )

    # Electron energy advection

    # No boundary terms means reflective boundaries => allows all outflow to be governed by sheath heat transmission coefficients
    rk.addModel(cm.staggeredAdvection(modelTag="advection-We",
                                      advectedVar="We",
                                      fluxVar="Ge_dual",
                                      vData=sc.VarData(reqColVars=["We_dual", "ne_dual"],
                                                       reqColPowers=[1.0, -1.0]
                                                       )
                                      )
                )

    # Ion energy advection

    # No boundary terms means reflective boundaries => allows all outflow to be governed by sheath heat transmission coefficients

    rk.addModel(cm.staggeredAdvection(modelTag="advection-Wi",
                                      advectedVar="Wi",
                                      fluxVar="Gi_dual",
                                      vData=sc.VarData(reqColVars=["Wi_dual", "ni_dual"],
                                                       reqColPowers=[1.0, -1.0]
                                                       )
                                      )
                )

    # Electron pressure advection

    rk.addModel(cm.staggeredAdvection(modelTag="advection-pe",
                                      advectedVar="We",
                                      fluxVar="Ge_dual",
                                      vData=sc.VarData(reqColVars=["Te_dual"])
                                      )
                )

    # Ion pressure advection

    # No boundary terms means reflective boundaries => allows all outflow to be governed by sheath heat transmission coefficients

    rk.addModel(cm.staggeredAdvection(modelTag="advection-pi",
                                      advectedVar="Wi",
                                      fluxVar="Gi_dual",
                                      vData=sc.VarData(reqColVars=["Ti_dual"])
                                      )
                )

    # Lorentz force work terms


    rk.addModel(cm.lorentzForceWork(modelTag="lorentzForceWork",
                                                eFieldName="E_dual",
                                                speciesFluxes=["Ge_dual", "Gi_dual"],
                                                speciesEnergies=["We", "Wi"],
                                                species=[electronSpecies, ionSpecies]
                                                )
                )


    # Constants for Braginskii terms
    sqrt2 = np.sqrt(2)

    delta = (1 + 65*sqrt2/32 + 433*sqrt2/288 - 23*sqrt2/16) + (5629/1152 - 529/128)  # A30 in Makarov assuming single ion species and 0 mass ratio

    thermFrictionConst = 25*sqrt2*(1+11*sqrt2/30)/(16*delta)  # A50

    elCondConst = 125*(1+433*sqrt2/360)/(32*delta)
    ionCondConst = 125/32

    # Get the e-i Coulomb Log calculated at normalization values

    # Normalization e-i coulomb log from NRL formulary
    logNorm = skn.logLei(tempNorm,densNorm,1.0)
        
    # Braginskii heat fluxes

    # Adding the model tag to tag list
    modelTag = "braginskiiq"

    # Initializing model
    braginskiiHFModel = sc.CustomModel(modelTag=modelTag)

    # Creating modelbound data properties for e-e and i-i Coulomb logs
    mbData = sc.VarlikeModelboundData()

    mbData.addVariable("logLee", sc.derivationRule("logLee", ["Te_dual", "ne_dual"]))
    mbData.addVariable("logLii", sc.derivationRule("logLiiD+_D+", ["ni_dual", "ni_dual", "Ti_dual", "Ti_dual"]))

    braginskiiHFModel.setModelboundData(mbData.dict())

    # Setting normalization constant calculation
    normConstI = sc.CustomNormConst(multConst=-1.0)

    nConstGradT = 12*np.pi**1.5*epsilon0**2/np.sqrt(elMass*elCharge)  # Comes from e-i collision time

    normConstGradTEl = sc.CustomNormConst(multConst=-nConstGradT*elCondConst,
                                        normNames=["eVTemperature", "length", "heatFlux"], normPowers=[3.5, -1.0, -1.0])
    normConstGradTIon = sc.CustomNormConst(multConst=-nConstGradT*ionCondConst*np.sqrt(elMass/ionMass),
                                        normNames=["eVTemperature", "length", "heatFlux"], normPowers=[3.5, -1.0, -1.0])

    # Variable data

    gradDataEl = sc.VarData(reqRowVars=["Te_dual"], reqRowPowers=[2.5], reqMBRowVars=["logLee"], reqMBRowPowers=[-1.0])
    gradDataIon = sc.VarData(reqRowVars=["Ti_dual"], reqRowPowers=[2.5], reqMBRowVars=["logLii"], reqMBRowPowers=[-1.0])

    # Electrons

    evolvedVar = "qe_dual"

    # Identity term

    identityTermEl = sc.GeneralMatrixTerm(evolvedVar, customNormConst=normConstI, stencilData=sc.diagonalStencil())

    braginskiiHFModel.addTerm("identityTerm_e", identityTermEl)

    # Gradient terms

    implicitVar = "Te"

    gradTermEl = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar,
                                    customNormConst=normConstGradTEl, stencilData=sc.staggeredGradStencil(), varData=gradDataEl)

    braginskiiHFModel.addTerm("bulkGrad_e", gradTermEl)


    # Ions

    evolvedVar = "qi_dual"

    # Identity term

    identityTermIon = sc.GeneralMatrixTerm(evolvedVar, customNormConst=normConstI, stencilData=sc.diagonalStencil())

    braginskiiHFModel.addTerm("identityTerm_i", identityTermIon)

    # Gradient terms

    gradTermIon = sc.GeneralMatrixTerm(evolvedVar, implicitVar="Ti",
                                    customNormConst=normConstGradTIon, stencilData=sc.staggeredGradStencil(), varData=gradDataIon)

    braginskiiHFModel.addTerm("bulkGrad_i", gradTermIon)

    rk.addModel(braginskiiHFModel.dict())

    # Electron heat flux divergence

    # Adding the model tag to tag list
    modelTag = "divq_e"

    # Initializing model
    electronDivQModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constants

    normFlux = sc.CustomNormConst(
        multConst=-1/elCharge, normNames=["heatFlux", "time", "length", "density", "eVTemperature"], normPowers=[1.0, 1.0, -1.0, -1.0, -1.0])
    normBC = sc.CustomNormConst(multConst=-1.0, normNames=["speed", "time", "length"], normPowers=[1.0, 1.0, -1.0])

    vDataBCRight = sc.VarData(reqRowVars=["gammaRight"], reqColVars=["Te"])

    # Bulk flux divergence

    evolvedVar = "We"

    divFluxTerm = sc.GeneralMatrixTerm(evolvedVar, implicitVar="qe_dual",
                                    customNormConst=normFlux, stencilData=sc.staggeredDivStencil())

    electronDivQModel.addTerm("divFlux", divFluxTerm)

    # Add Right boundary term with Bohm condition to outflow (internal energy term)

    rightBCTerm1 = sc.GeneralMatrixTerm(evolvedVar, implicitVar="ne", customNormConst=normBC,
                                    varData=vDataBCRight, stencilData=sc.boundaryStencilDiv("ue", "cs"))

    electronDivQModel.addTerm("rightBCT", rightBCTerm1)

    # Add Right boundary term with Bohm condition to outflow (kinetic energy term)

    vDataBCRightKin = sc.VarData(reqRowVars=["u_b"],reqRowPowers=[2.0])

    rightBCTerm2 = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normBC,
                                    varData=vDataBCRightKin, stencilData=sc.boundaryStencilDiv("ue", "cs"))

    electronDivQModel.addTerm("rightBCU", rightBCTerm2)

    rk.addModel(electronDivQModel.dict())

    # Ion heat flux divergence

    # Adding the model tag to tag list
    modelTag = "divq_i"

    # Initializing model
    ionDivQModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constants

    normFlux = sc.CustomNormConst(
        multConst=-1/elCharge, normNames=["heatFlux", "time", "length", "density", "eVTemperature"], normPowers=[1.0, 1.0, -1.0, -1.0, -1.0])
    normBC = sc.CustomNormConst(multConst=-1.0, normNames=["speed", "time", "length"], normPowers=[1.0, 1.0, -1.0])

    vDataBC = sc.VarData(reqRowVars=["ionGamma"], reqColVars=["Ti"])

    # Bulk flux divergence

    evolvedVar = "Wi"

    divFluxTerm = sc.GeneralMatrixTerm(evolvedVar, implicitVar="qi_dual",
                                    customNormConst=normFlux, stencilData=sc.staggeredDivStencil())

    ionDivQModel.addTerm("divFlux", divFluxTerm)

    # Add Right boundary term with Bohm condition to outflow

    rightBCTerm = sc.GeneralMatrixTerm(evolvedVar, implicitVar="ni", customNormConst=normBC,
                                    varData=vDataBC, stencilData=sc.boundaryStencilDiv("ui", "cs"))

    ionDivQModel.addTerm("rightBCT", rightBCTerm)

    # Add Right boundary term with Bohm condition to outflow (kinetic energy term)

    normBCKin = sc.CustomNormConst(multConst=-ionMass/elMass, normNames=["speed", "time", "length"], normPowers=[1.0, 1.0, -1.0])
    vDataBCRightKin = sc.VarData(reqRowVars=["u_b"],reqRowPowers=[2.0])

    rightBCTerm2 = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normBCKin,
                                    varData=vDataBCRightKin, stencilData=sc.boundaryStencilDiv("ue", "cs"))

    ionDivQModel.addTerm("rightBCU", rightBCTerm2)

    rk.addModel(ionDivQModel.dict())

    # Electron-ion heat exchange terms

    # Adding the model tag to tag list
    modelTag = "eiHeatEx"

    # Initializing model
    eiHeatExModel = sc.CustomModel(modelTag=modelTag)

    # Numerical factor from conversion of ReMKiT1D to Braginskii collision time

    normConst = 4/np.sqrt(np.pi)*elMass/(ionMass*logNorm)

    # Creating modelbound data properties for e-i Coulomb log

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable("logLei", sc.derivationRule("logLeiD+", ["Te", "ne"]))

    eiHeatExModel.setModelboundData(mbData.dict())

    vData = sc.VarData(reqRowVars=["ne", "Te"], reqRowPowers=[2.0, -1.5], reqMBRowVars=["logLei"])

    # Electron terms

    evolvedVar = "We"

    heatExTermAEl = sc.GeneralMatrixTerm(evolvedVar, implicitVar="Te",
                                        customNormConst=-normConst, varData=vData, stencilData=sc.diagonalStencil())

    eiHeatExModel.addTerm("heatExTermA_e", heatExTermAEl)

    heatExTermBEl = sc.GeneralMatrixTerm(evolvedVar, implicitVar="Ti",
                                        customNormConst=normConst, varData=vData, stencilData=sc.diagonalStencil())

    eiHeatExModel.addTerm("heatExTermB_e", heatExTermBEl)

    # Ion terms

    evolvedVar = "Wi"

    implicitVar = "Ti"

    heatExTermAIon = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar,
                                        customNormConst=-normConst, varData=vData, stencilData=sc.diagonalStencil())

    eiHeatExModel.addTerm("heatExTermA_i", heatExTermAIon)

    implicitVar = "Te"

    heatExTermBIon = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar,
                                        customNormConst=normConst, varData=vData, stencilData=sc.diagonalStencil())

    eiHeatExModel.addTerm("heatExTermB_i", heatExTermBIon)

    rk.addModel(eiHeatExModel.dict())

    Nh = kwargs.get("Nh",1)
    Lh = sum(xGridWidths[0:Nh])
    heatingPower = kwargs.get("heatingPower",3.0)/Lh  # in MW/m^3
    energyInjectionRate = heatingPower * 1e6 * timeNorm/(densNorm*elCharge*tempNorm)
    xProfileEnergy = np.zeros(Nx)
    xProfileEnergy[0:Nh] = energyInjectionRate

    # Energy source model

    # Adding the model tag to tag list
    modelTag = "energySource"

    # Initializing model
    energySourceModel = sc.CustomModel(modelTag=modelTag)

    # Electrons

    evolvedVar = "We"
    energySourceTermEl = cm.simpleSourceTerm(evolvedVar=evolvedVar, sourceProfile=xProfileEnergy)

    energySourceModel.addTerm("electronSource", energySourceTermEl)

    # Ions
    evolvedVar = "Wi"
    energySourceTermIon = cm.simpleSourceTerm(evolvedVar=evolvedVar, sourceProfile=xProfileEnergy)

    energySourceModel.addTerm("ionSource", energySourceTermIon)

    rk.addModel(energySourceModel.dict())

    # Electron-ion friction force terms

    # Adding the model tag to tag list
    modelTag = "eiFriction"

    # Initializing model
    eiFrictionModel = sc.CustomModel(modelTag=modelTag)

    # Setting normalization constant calculation
    normConstTel = sc.CustomNormConst(multConst=-elCharge*thermFrictionConst/elMass,
                                    normNames=["eVTemperature", "time", "speed", "length"], normPowers=[1.0, 1.0, -1.0, -1.0])
    normConstTion = sc.CustomNormConst(multConst=elCharge*thermFrictionConst/ionMass,
                                    normNames=["eVTemperature", "time", "speed", "length"], normPowers=[1.0, 1.0, -1.0, -1.0])

    # Numerical factors below come from ReMKiT1D to Braginskii time norm conversion (here the fact that time is normalized to the ei collision time is explicitly used)
    normConstUel = -4/(3*np.sqrt(np.pi)*logNorm)
    normConstUion = 4/(3*np.sqrt(np.pi)*logNorm)*elMass/ionMass

    # Creating modelbound data properties for e-i Coulomb log

    mbData = sc.VarlikeModelboundData()
    mbData.addVariable("logLei", sc.derivationRule("logLeiD+", ["Te_dual", "ne_dual"]))

    eiFrictionModel.setModelboundData(mbData.dict())

    vDataGradT = sc.VarData(reqRowVars=["ne_dual"])
    # Req vars for the R_u terms include implicit conversion of flux to speed
    vDataUEl = sc.VarData(reqRowVars=["ne_dual", "Te_dual"], reqRowPowers=[1.0, -1.5], reqMBRowVars=["logLei"])
    vDataUIon = sc.VarData(reqRowVars=["ne_dual", "Te_dual", "ni_dual"],
                        reqRowPowers=[2.0, -1.5, -1.0], reqMBRowVars=["logLei"])


    # Grad T terms
    implicitVar = "Te"
    evolvedVar = "Ge_dual"

    electronGradTFriction = sc.GeneralMatrixTerm(
        evolvedVar, implicitVar=implicitVar, customNormConst=normConstTel, varData=vDataGradT, stencilData=sc.staggeredGradStencil())

    eiFrictionModel.addTerm("electronGradTFriction", electronGradTFriction)

    evolvedVar = "Gi_dual"

    ionGradTFriction = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normConstTion, varData=vDataGradT, stencilData=sc.staggeredGradStencil(),
                                    )

    eiFrictionModel.addTerm("ionGradTFriction", ionGradTFriction)

    # Electron friction terms
    evolvedVar = "Ge_dual"

    implicitVar = "Ge_dual"

    electronUFrictionA = sc.GeneralMatrixTerm(
        evolvedVar, implicitVar=implicitVar, customNormConst=-normConstUel, varData=vDataUEl, stencilData=sc.diagonalStencil())

    eiFrictionModel.addTerm("eFriction_ue", electronUFrictionA)

    implicitVar = "Gi_dual"

    electronUFrictionB = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normConstUel,
                                            varData=vDataUIon, stencilData=sc.diagonalStencil())

    eiFrictionModel.addTerm("eFriction_ui", electronUFrictionB)

    # Ion friction terms

    evolvedVar = "Gi_dual"

    implicitVar = "Ge_dual"

    ionUFrictionA = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=normConstUion,
                                        varData=vDataUEl, stencilData=sc.diagonalStencil())

    eiFrictionModel.addTerm("iFriction_ue", ionUFrictionA)

    implicitVar = "Gi_dual"

    ionFrictionB = sc.GeneralMatrixTerm(evolvedVar, implicitVar=implicitVar, customNormConst=-normConstUion,
                                        varData=vDataUIon, stencilData=sc.diagonalStencil())

    eiFrictionModel.addTerm("iFriction_ui", ionFrictionB)

    rk.addModel(eiFrictionModel.dict())

    ionLambda = kwargs.get("ionMFP",0.2) #in meters!!!
    xProfileIonization = np.exp(-(L-gridObj.xGrid)/ionLambda)

    xProfileIonization = xProfileIonization/(sum(xProfileIonization*gridObj.xWidths/lengthNorm))

    # Adding the model tag to tag list
    modelTag = "rec"

    # Initializing model
    recModel = sc.CustomModel(modelTag=modelTag)

    recConst = 1.0  # Recycling coef
    xProfileIonization = xProfileIonization * recConst 

    evolvedVar = "ni"
    vData = sc.VarData(reqRowVars=[evolvedVar,"G_b"],reqRowPowers=[-1.0,1.0])   
    recTerm = sc.GeneralMatrixTerm(evolvedVar,spatialProfile=xProfileIonization.tolist(),varData=vData,stencilData=sc.diagonalStencil())
    recModel.addTerm("recyclingTerm_i", recTerm)

    evolvedVar = "ne"
    implicitVar = "ni"
    vData = sc.VarData(reqRowVars=[implicitVar,"G_b"],reqRowPowers=[-1.0,1.0])   
    recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,spatialProfile=xProfileIonization.tolist(),varData=vData,stencilData=sc.diagonalStencil())
    recModel.addTerm("recyclingTerm_e", recTerm)

    ionizationCost = kwargs.get("ionCost",20.0)/tempNorm # Fixed cost per ionization event
    xProfileIonizationEnergy = - xProfileIonization * ionizationCost 

    evolvedVar = "We"
    implicitVar = "ni"
    vData = sc.VarData(reqRowVars=[implicitVar,"G_b"],reqRowPowers=[-1.0,1.0])   
    recTerm = sc.GeneralMatrixTerm(evolvedVar,implicitVar,spatialProfile=xProfileIonizationEnergy.tolist(),varData=vData,stencilData=sc.diagonalStencil())
    recModel.addTerm("recyclingTermEnergy", recTerm)

    rk.addModel(recModel.dict())

    rk.setPETScOptions(cliOpts="-pc_type bjacobi -sub_pc_factor_shift_type nonzero -sub_pc_factor_levels 1 -ksp_gmres_restart 80",kspSolverType="gmres")

    return rk