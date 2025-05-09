import pyomo.environ as pyo


def get_ev_routing_abstract_model():

    m = pyo.AbstractModel()

    # ----  
    # Sets
    # ----

    m.sIntersections = pyo.Set()
    m.sPaths = pyo.Set()
    m.sDeliveryPoints = pyo.Set(within=m.sIntersections)
    m.sChargingStations = pyo.Set(within=m.sIntersections)

    # ----
    # Parameters
    # ----

    # Unindexed parameters
    m.pAccelerationEfficiency = pyo.Param(within=pyo.NonNegativeReals)
    m.pBrakingEfficiency = pyo.Param(within=pyo.NonNegativeReals)
    m.pMinSoC = pyo.Param(within=pyo.NonNegativeReals)
    m.pMaxSoC = pyo.Param(within=pyo.NonNegativeReals)
    m.pStartingSoC = pyo.Param(within=pyo.NonNegativeReals)
    m.pStartingTime = pyo.Param(within=pyo.NonNegativeReals)
    m.pMaxTime = pyo.Param(within=pyo.NonNegativeReals)

    # Parameters indexed by sPaths
    m.pOriginIntersection = pyo.Param(m.sPaths, within=m.sIntersections)
    m.pDestinationIntersection = pyo.Param(m.sPaths, within=m.sIntersections)
    m.pPathLength = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)
    m.pAvgSpeed = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)
    m.pAccelerationBrakingTime = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)
    m.pDistanceAtAvgSpeed = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)
    m.pPowerConsAtAvgSpeed = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)
    m.pKineticEnergy = pyo.Param(m.sPaths, within=pyo.NonNegativeReals)

    # Parameters indexed by sDeliveryPoints
    m.pDeliveryIntersection = pyo.Param(m.sDeliveryPoints, within=m.sIntersections)
    m.pTimeMakingDelivery = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)
    m.pTimeWithoutPenalty = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)
    m.pDelayPenalty = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)

    # Parameters indexed by sChargingStations
    m.pStationIntersection = pyo.Param(m.sChargingStations, within=m.sIntersections)
    m.pChargingPower = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)
    m.pChargingPrice = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)
    m.pMaxChargingTime = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)
    m.pMinChargingTime = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)
    m.pChargerEfficiencyRate = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)

    # ----
    # Variables
    # ----

    # ...       

    # ----
    # Constraints
    # ----

    # ...
    
    # ----
    # Objective function
    # ----

    # ...

    return m
