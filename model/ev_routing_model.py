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
    m.pStartingPoint = pyo.Param(within=m.sIntersections)
    m.pEndingPoint = pyo.Param(within=m.sIntersections)

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

    # Binary variables
    m.v01Charge = pyo.Var(m.sChargingStations, within=pyo.Binary)
    m.v01TravelPath = pyo.Var(m.sPaths, within=pyo.Binary)
    m.v01TravelIntersection = pyo.Var(m.sIntersections, within=pyo.Binary)

    # State of charge
    m.vSoCArrival = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)
    m.vSoCDeparture = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)

    # Operation time
    m.vTimeArrival = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)
    m.vTimeDeparture = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)

    # Time difference (delta)
    m.vTimeCharging = pyo.Var(m.sChargingStations, within=pyo.NonNegativeReals)
    m.vTimeDelay = pyo.Var(m.sDeliveryPoints, within=pyo.NonNegativeReals)

    # ----
    # Constraints
    # ----

    # Navigation constraints

    def c36_travel_intersection(m, intersection):
        return sum(
            m.v01TravelPath[path] for path in m.sPaths
            if m.pOriginIntersection[path] == intersection
        ) == m.v01TravelIntersection[intersection]

    m.c36_travel_intersection = pyo.Constraint(
        m.sIntersections, rule=c36_travel_intersection
    )

    def c37_home_cannot_be_origin(m):
        pass

    # ...
    
    # ----
    # Objective function
    # ----

    # ...

    return m
