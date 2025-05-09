from model import get_routing_map_data


if __name__ == "__main__":
    input_data = get_routing_map_data("./data/37-intersection map.xlsx")
    print(input_data)
