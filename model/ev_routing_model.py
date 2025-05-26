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
    m.pNumIntersections = pyo.Param(within=pyo.PositiveIntegers)

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

    # Parameter to find path ID given origin and destination nodes
    m.pPath = pyo.Param(m.sIntersections, m.sIntersections, within=m.sPaths)

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
    
    # MTZ variables for subtour elimination
    m.vOrderVisited = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)

    # ----
    # Objective function
    # ----

    # TODO: replace with the actual objective function

    def objective_function(m):
        return sum(m.v01VisitIntersection[intersection] for intersection in m.sIntersections)

    m.Obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    # ----
    # Navigation constraints
    # ----

    def c36_visit_intersection_when_origin_of_path(m, intersection):
        # Every visited node needs to be the origin of a path, except for the ending point
        if intersection == m.pEndingPoint:
            return pyo.Constraint.Skip
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
        # Every visited node needs to be the destination of a path, except for the starting point
        if intersection == m.pStartingPoint:
            return pyo.Constraint.Skip
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

    # cn1: force the EV to end at the ending point
    # It seems this constraint is unnecessary;
    # the EV automatically ends on the ending node with the other constraints

    # def cn1_visit_ending_point(m):
    #     return m.v01VisitIntersection[m.pEndingPoint] == 1
    #
    # m.cn1_visit_ending_point = pyo.Constraint(
    #     rule=cn1_visit_ending_point
    # )

    def c41_visit_delivery_point(m, delivery_point):
        return m.v01VisitIntersection[delivery_point] == 1

    m.c41_visit_delivery_point = pyo.Constraint(
        m.sDeliveryPoints, rule=c41_visit_delivery_point
    )

    # cn2, cn3 and cn4: Miller-Tucker-Zemlin (MTZ) formulation for subtour elimination
    # See [[Issue - Loops in navigation]]
    
    def cn2_mtz_subtour_elimination(m, path):
        origin = m.pOriginIntersection[path]
        destination = m.pDestinationIntersection[path]
        
        # Skip constraint if origin is the starting point (acts as depot in TSP)
        if origin == m.pStartingPoint:
            return pyo.Constraint.Skip
            
        return (
            m.vOrderVisited[origin] - m.vOrderVisited[destination] + 1 
            <= (m.pNumIntersections - 1) * (1 - m.v01TravelPath[path])
        )

    m.cn2_mtz_subtour_elimination = pyo.Constraint(
        m.sPaths, rule=cn2_mtz_subtour_elimination
    )
    
    def cn3_mtz_order_lower_bound(m, intersection):
        if intersection == m.pStartingPoint:
            return m.vOrderVisited[intersection] == 1
        else:
            return m.vOrderVisited[intersection] >= 2

    m.cn3_mtz_order_lower_bound = pyo.Constraint(
        m.sIntersections, rule=cn3_mtz_order_lower_bound
    )

    def cn4_mtz_order_upper_bound(m, intersection):
        return m.vOrderVisited[intersection] <= m.pNumIntersections

    m.cn4_mtz_order_upper_bound = pyo.Constraint(
        m.sIntersections, rule=cn4_mtz_order_upper_bound
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
