import pyomo.environ as pyo


def get_ev_routing_abstract_model(linearize_constraints=False):

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
    # m.pDeliveryIntersection = pyo.Param(m.sDeliveryPoints, within=m.sIntersections)  # Unnecessary, since sDeliveryPoints is a subset of sIntersections
    m.pTimeMakingDelivery = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)
    m.pTimeWithoutPenalty = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)
    m.pDelayPenalty = pyo.Param(m.sDeliveryPoints, within=pyo.NonNegativeReals)

    # Parameters indexed by sChargingStations
    # m.pStationIntersection = pyo.Param(m.sChargingStations, within=m.sIntersections)  # Unnecessary, since sChargingStations is a subset of sIntersections
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
    # -> These are actually not necessary, see [[Update - 2025-05-29]]
    # m.vOrderVisited = pyo.Var(m.sIntersections, within=pyo.NonNegativeReals)

    # Auxiliary variables for linearization
    if linearize_constraints:
        # Auxiliary variables for constraint c43 linearization (SoC energy balance)
        m.vXiSoC = pyo.Var(m.sPaths, within=pyo.NonNegativeReals)
        
        # Auxiliary variables for constraint c46 linearization (time arrival balance)
        m.vZetaTime = pyo.Var(m.sPaths, within=pyo.NonNegativeReals)

    # ----
    # Objective function
    # ----

    def c34_objective_function(m):
        # Charging cost component
        charging_cost = sum(
            m.pChargingPrice[charging_station] *
            m.pChargingPower[charging_station] *
            m.vTimeCharging[charging_station] *
            m.pChargerEfficiencyRate[charging_station]
            for charging_station in m.sChargingStations
        )

        # Delay penalty component
        delay_penalty = sum(
            m.pDelayPenalty[delivery_point] *
            m.vTimeDelay[delivery_point]
            for delivery_point in m.sDeliveryPoints
        )

        return charging_cost + delay_penalty

    m.Obj = pyo.Objective(rule=c34_objective_function, sense=pyo.minimize)

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
    # -> These are actually not necessary, see [[Update - 2025-05-29]]
    
    # def cn2_mtz_subtour_elimination(m, path):
    #     origin = m.pOriginIntersection[path]
    #     destination = m.pDestinationIntersection[path]
    #
    #     # Skip constraint if origin is the starting point (acts as depot in TSP)
    #     if origin == m.pStartingPoint:
    #         return pyo.Constraint.Skip
    #
    #     return (
    #         m.vOrderVisited[origin] - m.vOrderVisited[destination] + 1
    #         <= (m.pNumIntersections - 1) * (1 - m.v01TravelPath[path])
    #     )
    #
    # m.cn2_mtz_subtour_elimination = pyo.Constraint(
    #     m.sPaths, rule=cn2_mtz_subtour_elimination
    # )
    #
    # def cn3_mtz_order_lower_bound(m, intersection):
    #     if intersection == m.pStartingPoint:
    #         return m.vOrderVisited[intersection] == 1
    #     else:
    #         return m.vOrderVisited[intersection] >= 2
    #
    # m.cn3_mtz_order_lower_bound = pyo.Constraint(
    #     m.sIntersections, rule=cn3_mtz_order_lower_bound
    # )
    #
    # def cn4_mtz_order_upper_bound(m, intersection):
    #     return m.vOrderVisited[intersection] <= m.pNumIntersections
    #
    # m.cn4_mtz_order_upper_bound = pyo.Constraint(
    #     m.sIntersections, rule=cn4_mtz_order_upper_bound
    # )

    # ----
    # Battery constraints
    # ----

    # Split constraint (24) into two separate constraints for proper Pyomo handling
    def c24_soc_arrival_lower_bound(m, intersection):
        return m.vSoCArrival[intersection] >= m.v01VisitIntersection[intersection] * m.pMinSoC

    m.c24_soc_arrival_lower_bound = pyo.Constraint(
        m.sIntersections, rule=c24_soc_arrival_lower_bound
    )

    def c24_soc_arrival_upper_bound(m, intersection):
        return m.vSoCArrival[intersection] <= m.v01VisitIntersection[intersection] * m.pMaxSoC
    
    m.c24_soc_arrival_upper_bound = pyo.Constraint(
        m.sIntersections, rule=c24_soc_arrival_upper_bound
    )

    # Impose the same limits on SoC departure
    # This is only necessary in the non-linear model, since the linearized one already has cA6_soc_minus_xi_upper_bound
    # See [[Update - 2025-05-30]]
    if not linearize_constraints:
        def c24_soc_departure_lower_bound(m, intersection):
            return m.vSoCDeparture[intersection] >= m.v01VisitIntersection[intersection] * m.pMinSoC

        m.c24_soc_departure_lower_bound = pyo.Constraint(
            m.sIntersections, rule=c24_soc_departure_lower_bound
        )

        def c24_soc_departure_upper_bound(m, intersection):
            return m.vSoCDeparture[intersection] <= m.v01VisitIntersection[intersection] * m.pMaxSoC
        
        m.c24_soc_departure_upper_bound = pyo.Constraint(
            m.sIntersections, rule=c24_soc_departure_upper_bound
        )

    def c25_soc_departure_charging_station(m, charging_station):
        return (
            m.vSoCDeparture[charging_station] == 
            m.vSoCArrival[charging_station] + 
            m.pChargingPower[charging_station] * m.vTimeCharging[charging_station] * m.pChargerEfficiencyRate[charging_station]
        )

    m.c25_soc_departure_charging_station = pyo.Constraint(
        m.sChargingStations, rule=c25_soc_departure_charging_station
    )

    def c27_soc_departure_non_charging_station(m, intersection):
        return m.vSoCDeparture[intersection] == m.vSoCArrival[intersection]

    m.c27_soc_departure_non_charging_station = pyo.Constraint(
        # Set differences are allowed - See https://pyomo.readthedocs.io/en/6.8.0/pyomo_modeling_components/Sets.html#operations
        m.sIntersections - m.sChargingStations, rule=c27_soc_departure_non_charging_station
    )

    def c42_soc_starting_point(m):
        # Multiplying by the binary variable here is unnecessary due to constraint c40_visit_starting_point
        return m.vSoCArrival[m.pStartingPoint] == m.v01VisitIntersection[m.pStartingPoint] * m.pStartingSoC

    m.c42_soc_starting_point = pyo.Constraint(
        rule=c42_soc_starting_point
    )

    def c43_soc_arrival_energy_balance(m, intersection):
        # Skip for starting point as it's handled by c42
        if intersection == m.pStartingPoint:
            return pyo.Constraint.Skip
        
        if linearize_constraints:
            # Linearized version using auxiliary variables
            return m.vSoCArrival[intersection] == sum(
                m.vXiSoC[path] - m.v01TravelPath[path] * (
                    m.pPowerConsAtAvgSpeed[path] * (m.pDistanceAtAvgSpeed[path] / m.pAvgSpeed[path]) +
                    (1 / m.pAccelerationEfficiency - m.pBrakingEfficiency) * m.pKineticEnergy[path]
                )
                for path in m.sPaths
                if m.pDestinationIntersection[path] == intersection
            )
        else:
            # Original quadratic constraint
            # SoC arrival = SoC departure from origin - power consumption - acceleration energy + braking energy
            return m.vSoCArrival[intersection] == sum(
                m.v01TravelPath[path] * (
                    m.vSoCDeparture[m.pOriginIntersection[path]] -
                    m.pPowerConsAtAvgSpeed[path] * (m.pDistanceAtAvgSpeed[path] / m.pAvgSpeed[path]) -
                    (1 / m.pAccelerationEfficiency - m.pBrakingEfficiency) * m.pKineticEnergy[path]
                )
                for path in m.sPaths
                if m.pDestinationIntersection[path] == intersection
            )

    m.c43_soc_arrival_energy_balance = pyo.Constraint(
        m.sIntersections, rule=c43_soc_arrival_energy_balance
    )

    def c44_soc_ending_point(m):
        # Multiplying by the binary variable here is unnecessary since the ending point is always visited
        # We have to use >= instead of == here, see [[Update - 2025-05-29]]
        return m.vSoCDeparture[m.pEndingPoint] >= m.v01VisitIntersection[m.pEndingPoint] * m.pStartingSoC

    m.c44_soc_ending_point = pyo.Constraint(
        rule=c44_soc_ending_point
    )

    # ----
    # Time constraints
    # ----

    def c45_time_arrival_starting_point(m):
        return m.vTimeArrival[m.pStartingPoint] == m.v01VisitIntersection[m.pStartingPoint] * m.pStartingTime

    m.c45_time_arrival_starting_point = pyo.Constraint(
        rule=c45_time_arrival_starting_point
    )

    def c46_time_arrival_balance(m, intersection):
        # Skip for starting point as it's handled by c45
        if intersection == m.pStartingPoint:
            return pyo.Constraint.Skip
        
        if linearize_constraints:
            # Linearized version using auxiliary variables
            return m.vTimeArrival[intersection] == sum(
                m.vZetaTime[path] + m.v01TravelPath[path] * (
                    (m.pDistanceAtAvgSpeed[path] / m.pAvgSpeed[path]) +
                    m.pAccelerationBrakingTime[path]
                )
                for path in m.sPaths
                if m.pDestinationIntersection[path] == intersection
            )
        else:
            # Original quadratic constraint
            # Time arrival = Time departure from origin + time at average speed + acceleration/braking time
            return m.vTimeArrival[intersection] == sum(
                m.v01TravelPath[path] * (
                    m.vTimeDeparture[m.pOriginIntersection[path]] +
                    (m.pDistanceAtAvgSpeed[path] / m.pAvgSpeed[path]) +
                    m.pAccelerationBrakingTime[path]
                )
                for path in m.sPaths
                if m.pDestinationIntersection[path] == intersection
            )

    m.c46_time_arrival_balance = pyo.Constraint(
        m.sIntersections, rule=c46_time_arrival_balance
    )

    def c47_time_departure_delivery_point(m, delivery_point):
        return m.vTimeDeparture[delivery_point] == m.vTimeArrival[delivery_point] + m.pTimeMakingDelivery[delivery_point]

    m.c47_time_departure_delivery_point = pyo.Constraint(
        m.sDeliveryPoints, rule=c47_time_departure_delivery_point
    )

    def c48_time_departure_charging_station(m, charging_station):
        return m.vTimeDeparture[charging_station] == m.vTimeArrival[charging_station] + m.vTimeCharging[charging_station]

    m.c48_time_departure_charging_station = pyo.Constraint(
        m.sChargingStations, rule=c48_time_departure_charging_station
    )

    # Split constraint (49) into two separate constraints for proper Pyomo handling
    def c49_charging_time_lower_bound(m, charging_station):
        return m.vTimeCharging[charging_station] >= m.v01Charge[charging_station] * m.pMinChargingTime[charging_station]

    m.c49_charging_time_lower_bound = pyo.Constraint(
        m.sChargingStations, rule=c49_charging_time_lower_bound
    )

    def c49_charging_time_upper_bound(m, charging_station):
        return m.vTimeCharging[charging_station] <= m.v01Charge[charging_station] * m.pMaxChargingTime[charging_station]

    m.c49_charging_time_upper_bound = pyo.Constraint(
        m.sChargingStations, rule=c49_charging_time_upper_bound
    )

    def c50_charge_only_when_visiting(m, charging_station):
        return m.v01Charge[charging_station] <= m.v01VisitIntersection[charging_station]

    m.c50_charge_only_when_visiting = pyo.Constraint(
        m.sChargingStations, rule=c50_charge_only_when_visiting
    )

    def c51_time_departure_regular_intersection(m, intersection):
        return m.vTimeDeparture[intersection] == m.vTimeArrival[intersection]

    m.c51_time_departure_regular_intersection = pyo.Constraint(
        m.sIntersections - m.sChargingStations - m.sDeliveryPoints, rule=c51_time_departure_regular_intersection
    )

    def c52_time_arrival_ending_point_limit(m):
        return m.vTimeArrival[m.pEndingPoint] <= m.v01VisitIntersection[m.pEndingPoint] * m.pMaxTime

    m.c52_time_arrival_ending_point_limit = pyo.Constraint(
        rule=c52_time_arrival_ending_point_limit
    )

    def c54_delivery_time_with_delay(m, delivery_point):
        return m.vTimeArrival[delivery_point] <= m.pTimeWithoutPenalty[delivery_point] + m.vTimeDelay[delivery_point]

    m.c54_delivery_time_with_delay = pyo.Constraint(
        m.sDeliveryPoints, rule=c54_delivery_time_with_delay
    )

    # Constraint (55) is not needed because vTimeDelay is non-negative by definition
    # def c55_delay_time_positive(m, delivery_point):
    #     return m.vTimeDelay[delivery_point] >= 0

    # m.c55_delay_time_positive = pyo.Constraint(
    #     m.sDeliveryPoints, rule=c55_delay_time_positive
    # )

    # ----
    # Linearization constraints
    # ----
    
    if linearize_constraints:

        # Linearization constraints for c43 (SoC energy balance)
        # These implement: vXiSoC[path] = v01TravelPath[path] * vSoCDeparture[origin_of_path]
        
        def cA7_xi_lower_bound(m, path):
            # The lower bound is 0 instead of pMinSoC, see [[Update - 2025-05-29]]
            return m.vXiSoC[path] >= 0
        
        m.cA7_xi_lower_bound = pyo.Constraint(
            m.sPaths, rule=cA7_xi_lower_bound
        )
        
        def cA7_xi_upper_bound(m, path):
            return m.vXiSoC[path] <= m.pMaxSoC * m.v01TravelPath[path]
        
        m.cA7_xi_upper_bound = pyo.Constraint(
            m.sPaths, rule=cA7_xi_upper_bound
        )
        
        def cA6_soc_minus_xi_lower_bound(m, path):
            # The lower bound is 0 instead of pMinSoC, see [[Update - 2025-05-29]]
            origin = m.pOriginIntersection[path]
            return m.vSoCDeparture[origin] - m.vXiSoC[path] >= 0
        
        m.cA6_soc_minus_xi_lower_bound = pyo.Constraint(
            m.sPaths, rule=cA6_soc_minus_xi_lower_bound
        )

        def cA6_soc_minus_xi_upper_bound(m, path):
            origin = m.pOriginIntersection[path]
            return m.vSoCDeparture[origin] - m.vXiSoC[path] <= m.pMaxSoC * (1 - m.v01TravelPath[path])

        m.cA6_soc_minus_xi_upper_bound = pyo.Constraint(
            m.sPaths, rule=cA6_soc_minus_xi_upper_bound
        )
        
        # Linearization constraints for c46 (time arrival balance)
        # These implement: vZetaTime[path] = v01TravelPath[path] * vTimeDeparture[origin_of_path]
        
        def cA11_zeta_lower_bound(m, path):
            # The lower bound is 0 instead of pStartingTime, see [[Update - 2025-05-29]]
            return m.vZetaTime[path] >= 0
        
        m.cA11_zeta_lower_bound = pyo.Constraint(
            m.sPaths, rule=cA11_zeta_lower_bound
        )
        
        def cA11_zeta_upper_bound(m, path):
            return m.vZetaTime[path] <= m.pMaxTime * m.v01TravelPath[path]
        
        m.cA11_zeta_upper_bound = pyo.Constraint(
            m.sPaths, rule=cA11_zeta_upper_bound
        )
        
        def cA10_time_minus_zeta_lower_bound(m, path):
            # The lower bound is 0 instead of pStartingTime, see [[Update - 2025-05-29]]
            origin = m.pOriginIntersection[path]
            return m.vTimeDeparture[origin] - m.vZetaTime[path] >= 0
        
        m.cA10_time_minus_zeta_lower_bound = pyo.Constraint(
            m.sPaths, rule=cA10_time_minus_zeta_lower_bound
        )

        def cA10_time_minus_zeta_upper_bound(m, path):
            origin = m.pOriginIntersection[path]
            return m.vTimeDeparture[origin] - m.vZetaTime[path] <= m.pMaxTime * (1 - m.v01TravelPath[path])

        m.cA10_time_minus_zeta_upper_bound = pyo.Constraint(
            m.sPaths, rule=cA10_time_minus_zeta_upper_bound
        )

    return m
