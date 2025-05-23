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
    m.v01VisitIntersection = pyo.Var(m.sIntersections, within=pyo.Binary)

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
    # Objective function
    # ----

    # TODO: add the objective function

    # ----
    # Navigation constraints
    # ----

    def c36_visit_intersection_when_origin_of_path(m, intersection):
        return sum(
            m.v01TravelPath[path] for path in m.sPaths
            if m.pOriginIntersection[path] == intersection
        ) == m.v01VisitIntersection[intersection]

    m.c36_visit_intersection_when_origin_of_path = pyo.Constraint(
        m.sIntersections, rule=c36_visit_intersection_when_origin_of_path
    )

    def c37_ending_point_cannot_be_origin(m):
        return sum(
            m.v01TravelPath[path] for path in m.sPaths
            if m.pOriginIntersection[path] == m.pEndingPoint
        ) == 0

    m.c37_ending_point_cannot_be_origin = pyo.Constraint(
        rule=c37_ending_point_cannot_be_origin
    )

    def c38_visit_intersection_when_destination_of_path(m, intersection):
        return sum(
            m.v01TravelPath[path] for path in m.sPaths
            if m.pDestinationIntersection[path] == intersection
        ) == m.v01VisitIntersection[intersection]

    m.c38_visit_intersection_when_destination_of_path = pyo.Constraint(
        m.sIntersections, rule=c38_visit_intersection_when_destination_of_path
    )

    def c39_starting_point_cannot_be_destination(m):
        return sum(
            m.v01TravelPath[path] for path in m.sPaths
            if m.pDestinationIntersection[path] == m.pStartingPoint
        ) == 0

    m.c39_starting_point_cannot_be_destination = pyo.Constraint(
        rule=c39_starting_point_cannot_be_destination
    )

    def c40_visit_starting_point(m):
        return m.v01VisitIntersection[m.pStartingPoint] == 1

    m.c40_visit_starting_point = pyo.Constraint(
        rule=c40_visit_starting_point
    )

    # TODO: confirm whether c40plus is necessary

    def c40plus_visit_ending_point(m):
        return m.v01VisitIntersection[m.pEndingPoint] == 1

    m.c40plus_visit_ending_point = pyo.Constraint(
        rule=c40plus_visit_ending_point
    )

    def c41_visit_delivery_point(m, delivery_point):
        return m.v01VisitIntersection[delivery_point] == 1

    m.c41_visit_delivery_point = pyo.Constraint(
        m.sDeliveryPoints, rule=c41_visit_delivery_point
    )

    # ----
    # Battery constraints
    # ----

    # TODO: add the battery constraints

    # ----
    # Time constraints
    # ----

    # TODO: add the time constraints

    return m
