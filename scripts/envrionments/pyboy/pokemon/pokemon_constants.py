# https://github.com/pret/pokered/tree/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants

import json
from pathlib import Path

here = Path(__file__).parent


def load_dict(path):
    with open(path, "r", encoding="utf-8") as fp:
        info = json.load(fp)
        data = {}
        for k, v in info.items():
            data[int(k)] = v
        return data


pokemon = load_dict(f"{here}/pokemon_constants.json")


def get_pokemon(pokemon_id):
    if pokemon_id in pokemon:
        return pokemon[pokemon_id]
    return "Unknown Pokemon"


types = load_dict(f"{here}/type_constants.json")


def get_type(type_id):
    if type_id in types:
        return types[type_id]
    return "Unknown Type"


def get_status(status_id):
    status = {}
    if status_id in status:
        return status[status_id]
    return "Unknown Status"


map_locations = load_dict(f"{here}/map_constants.json")


def get_map_location(map_idx):
    # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/map_constants.asm
    if map_idx in map_locations:
        return map_locations[map_idx]
    return "Unknown Location"


def main():
    file_path = f"{Path.home()}/cares_rl_configs/pokemon/map_constants.asm"
    save_path = file_path.replace("asm", "json")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line for line in lines if "mapconst" in line]
        lines = lines[1:]

        data = {}
        for line in lines:
            line = line.split(" ")
            value = line[1]
            key = int(f"0x{line[len(line)-1].replace('$', '')}", base=16)
            data[key] = value

    print(data)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
