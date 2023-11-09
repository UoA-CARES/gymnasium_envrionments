

def get_map_location(map_idx):
    # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/map_constants.asm
    map_locations = {
        0: "Pallet Town",
        1: "Viridian City",
        2: "Pewter City",
        3: "Cerulean City",
        12: "Route 1",
        13: "Route 2",
        14: "Route 3",
        15: "Route 4",
        33: "Route 22",
        37: "Red house first",
        38: "Red house second",
        39: "Blues house",
        40: "oaks lab",
        41: "Pokémon Center (Viridian City)",
        42: "Poké Mart (Viridian City)",
        43: "School (Viridian City)",
        44: "House 1 (Viridian City)",
        47: "Gate (Viridian City/Pewter City) (Route 2)",
        49: "Gate (Route 2)",
        50: "Gate (Route 2/Viridian Forest) (Route 2)",
        51: "viridian forest",
        52: "Pewter Museum (floor 1)",
        53: "Pewter Museum (floor 2)",
        54: "Pokémon Gym (Pewter City)",
        55: "House with disobedient Nidoran♂ (Pewter City)",
        56: "Poké Mart (Pewter City)",
        57: "House with two Trainers (Pewter City)",
        58: "Pokémon Center (Pewter City)",
        59: "Mt. Moon (Route 3 entrance)",
        60: "Mt. Moon",
        61: "Mt. Moon",
        68: "Pokémon Center (Route 4)",
        193: "Badges check gate (Route 22)"
    }
    if map_idx in map_locations.keys():
        return map_locations[map_idx]
    else:
        return "Unknown Location"
