from model import get_routing_map_data, get_ev_routing_abstract_model
import pyomo.environ as pyo


def main():
    # Define the file path
    file_path = "../data/37-intersection map.xlsx"
    
    # Get data from Excel file
    print(f"Loading data from {file_path}...")
    input_data = get_routing_map_data(file_path, ev=1)
    
    # Get the abstract model
    print("Creating abstract model...")
    abstract_model = get_ev_routing_abstract_model()
    
    # Create a concrete instance using the data
    print("Creating concrete model instance...")
    concrete_model = abstract_model.create_instance(input_data)
    
    # Basic model information
    print("\nModel Information:")
    print(f"Number of intersections: {len(concrete_model.sIntersections)}")
    print(f"Number of paths: {len(concrete_model.sPaths)}")
    print(f"Number of delivery points: {len(concrete_model.sDeliveryPoints)}")
    print(f"Number of charging stations: {len(concrete_model.sChargingStations)}")
    
    print("\nConcrete model created successfully!")
    return concrete_model


if __name__ == "__main__":
    try:
        model = main()
        print("Model creation completed successfully.")
    except Exception as e:
        print(f"Error creating model: {e}") 