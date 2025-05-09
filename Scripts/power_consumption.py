import numpy as np

def calculate_power_consumption(speed, path_type=None):
    """
    Calculate the power consumption of an electric vehicle based on speed.
    
    Parameters:
    -----------
    speed : float
        Average speed on path p in m/s
    path_type : str, optional
        Type of path (if different speeds are needed based on path type)
        
    Returns:
    --------
    float
        Power consumption in kW
    """
    # Constants based on the provided values
    A_v = 3.06  # Frontal area [m²]
    C_v_DR = 0.31  # Drag coefficient
    C_v_IT = 0.075  # Rolling resistance coefficient
    P_v_A = 2.0  # Ancillary power losses [kW]
    P_v_S = 0.475  # Stopping power losses [kW]
    W_v = 21778.2  # Weight [N]
    alpha_v = 0.00096  # Drivetrain coefficient
    beta_v = 0.193  # Drivetrain coefficient
    gamma_v = 18.21  # Drivetrain coefficient
    rho = 1.225  # Air density [kg/m³]
    
    # Calculate power consumption using the formula:
    # P_{v,p} = (1/2)*C_v^{DR}*rho*A_v*V_p^3 + alpha_v*V_p^3 + beta_v*V_p^2 + gamma_v*V_p + P_v^S + W_v*C_v^{IT}*V_p + P_v^A
    
    term1_w = 0.5 * C_v_DR * rho * A_v * speed**3  # 1 * kg/m^3 * m^2 * m^3/s^3 = kg*m^2/s^3 = N*m/s = W
    term1_kw = term1_w / 1000  # kW

    term2 = alpha_v * speed**3  # m^3/s^3  ???
    term3 = beta_v * speed**2  # m^2/s^2  ???
    term4 = gamma_v * speed  # m/s  ???

    term5_kw = P_v_S  # kW 

    term6 = W_v * C_v_IT * speed  # N * 1 * m/s = N*m/s = W
    term6_kw = term6 / 1000  # kW

    term7_kw = P_v_A  # kW
 
    power_kw = term1_kw + term2 + term3 + term4 + term5_kw + term6_kw + term7_kw
    return power_kw


if __name__ == "__main__":
    
    speeds_kmh = [30, 50, 80]
    speeds_ms = [s / 3.6 for s in speeds_kmh]
    
    print("Power consumption at different speeds:")
    print("Speed (km/h) | Speed (m/s) | Power Consumption (kW)")
    print("---------------------------------------------------")
    for kmh, ms in zip(speeds_kmh, speeds_ms):
        power = calculate_power_consumption(ms)
        print(f"{kmh:11.1f} | {ms:10.2f} | {power:20.2f}")
    
