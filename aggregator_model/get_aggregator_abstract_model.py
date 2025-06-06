"""
This script defines the abstract aggregator model with Pyomo
"""

import pyomo.environ as pyo


def get_aggregator_abstract_model():
    """
    Create an abstract Pyomo model for the aggregator optimization problem.
    
    The model maximizes the revenue of the charging stations manager (aggregator).
    Variables: charging fees for each charging station (r^C_i)
    Parameters: hourly electricity prices (C_t), min/max charging fees
    Constraints: minimum and maximum charging fee bounds
    Objective: maximize income from charging fees minus electricity costs
    """

    m = pyo.AbstractModel()

    # ----  
    # Sets
    # ----

    m.sChargingStations = pyo.Set()
    m.sTimePeriods = pyo.Set()

    # ----
    # Parameters
    # ----

    # Parameters indexed by sChargingStations
    m.pMinChargingPrice = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)
    m.pMaxChargingPrice = pyo.Param(m.sChargingStations, within=pyo.NonNegativeReals)

    # Parameters indexed by sTimePeriods
    m.pElectricityCost = pyo.Param(m.sTimePeriods, within=pyo.NonNegativeReals)

    # For now, we use fixed demand values instead of regression model
    # Parameters indexed by sChargingStations and sTimePeriods
    # TODO: remove this parameter (replacing it with the proper regression model)
    m.pDemand = pyo.Param(m.sChargingStations, m.sTimePeriods, within=pyo.NonNegativeReals, default=10.0)

    # ----
    # Variables
    # ----

    # Charging fee for each charging station [$/kWh]
    m.vChargingPrice = pyo.Var(m.sChargingStations, within=pyo.NonNegativeReals)
    
    # Aggregated demand for each charging station at each time period [kWh]
    m.vAggregatedDemand = pyo.Var(m.sChargingStations, m.sTimePeriods, within=pyo.NonNegativeReals)

    # ----
    # Constraints
    # ----

    def c1_min_charging_price(m, station):
        """Minimum charging price constraint"""
        return m.vChargingPrice[station] >= m.pMinChargingPrice[station]

    m.c1_min_charging_price = pyo.Constraint(
        m.sChargingStations, rule=c1_min_charging_price
    )

    def c2_max_charging_price(m, station):
        """Maximum charging price constraint"""
        return m.vChargingPrice[station] <= m.pMaxChargingPrice[station]

    m.c2_max_charging_price = pyo.Constraint(
        m.sChargingStations, rule=c2_max_charging_price
    )

    def c3_set_aggregated_demand(m, station, time_period):
        """Set aggregated demand equal to fixed demand parameter (for now)"""
        # TODO: replace this with a proper constraint using the regression model
        return m.vAggregatedDemand[station, time_period] == m.pDemand[station, time_period]

    m.c3_set_aggregated_demand = pyo.Constraint(
        m.sChargingStations, m.sTimePeriods, rule=c3_set_aggregated_demand
    )

    # ----
    # Objective function
    # ----

    def objective_function(m):
        """
        Maximize revenue: income from charging fees minus electricity costs
        Income: charging fee * aggregated demand
        Cost: electricity cost * aggregated demand
        """
        # Income from charging fees
        income = sum(
            m.vChargingPrice[station] * m.vAggregatedDemand[station, time_period]
            for station in m.sChargingStations
            for time_period in m.sTimePeriods
        )
        
        # Electricity costs
        cost = sum(
            m.pElectricityCost[time_period] * m.vAggregatedDemand[station, time_period]
            for station in m.sChargingStations
            for time_period in m.sTimePeriods
        )
        
        return income - cost

    m.Obj = pyo.Objective(rule=objective_function, sense=pyo.maximize)

    return m
