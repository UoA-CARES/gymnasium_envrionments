from typing import List, Dict
from pyboy.utils import WindowEvent
from util.configurations import GymEnvironmentConfig
from envrionments.pyboy.Pyboy import Pyboy

import envrionments.pyboy.pokemon.pokemon_constants as pkc


class Pokemon(Pyboy):
    def __init__(self, config: GymEnvironmentConfig) -> None:
        super().__init__(config, "PokemonRed.gb", "has_pokedex.state")

        self.valid_actions: List[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_button: List[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

    def _stats_to_state(self, game_stats: Dict[str, any]) -> List[any]:
        state: List[any] = []
        return state

    def _generate_game_stats(self) -> Dict[str, any]:
        return {
            "location": self._get_location(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            "type_id": self._read_party_type(),
            "type": [pkc.get_type(id) for id in self._read_party_type()],
            "hp": self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            "caught_pokemon": self._read_caught_pokemon_count(),
            "seen_pokemon": self._read_seen_pokemon_count(),
            "money": self._read_money(),
            "events": self._read_events(),
        }

    def _reward_stats_to_reward(self, reward_stats: Dict[str, any]) -> int:
        reward_total = 0
        for _, reward in reward_stats.items():
            reward_total += reward
        return reward_total

    def _calculate_reward_stats(self, new_state: Dict[str, any]) -> Dict[str, int]:
        return {
            "caught_reward": self._caught_reward(new_state),
            "seen_reward": self._seen_reward(new_state),
            "health_reward": self._health_reward(new_state),
            "xp_reward": self._xp_reward(new_state),
            "levels_reward": self._levels_reward(new_state),
            "badges_reward": self._badges_reward(new_state),
            "money_reward": self._money_reward(new_state),
            "event_reward": self._event_reward(new_state),
        }

    def _caught_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["caught_pokemon"] - self.prior_game_stats["caught_pokemon"]

    def _seen_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["seen_pokemon"] - self.prior_game_stats["seen_pokemon"]

    def _health_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["hp"]["current"]) - sum(
            self.prior_game_stats["hp"]["current"]
        )

    def _xp_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])

    def _levels_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["levels"]) - sum(self.prior_game_stats["levels"])

    def _badges_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["badges"] - self.prior_game_stats["badges"]

    def _money_reward(self, new_state: Dict[str, any]) -> int:
        return new_state["money"] - self.prior_game_stats["money"]

    def _event_reward(self, new_state: Dict[str, any]) -> int:
        return sum(new_state["events"]) - sum(self.prior_game_stats["events"])

    def _check_if_done(self, game_stats: Dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return True if self.prior_game_stats["badges"] > 0 else False

    def _get_location(self) -> Dict[str, any]:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {
            "x": x_pos,
            "y": y_pos,
            "map_id": map_n,
            "map": pkc.get_map_location(map_n),
        }

    def _get_party_size(self) -> int:
        return self._read_m(0xD163)

    def _get_badge_count(self) -> int:
        return self._bit_count(self._read_m(0xD356))

    def _read_party_id(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/pokemon_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def _read_party_type(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/type_constants.asm
        return [
            self._read_m(addr)
            for addr in [
                0xD170,
                0xD171,
                0xD19C,
                0xD19D,
                0xD1C8,
                0xD1C9,
                0xD1F4,
                0xD1F5,
                0xD220,
                0xD221,
                0xD24C,
                0xD24D,
            ]
        ]

    def _read_party_level(self) -> List[int]:
        return [
            self._read_m(addr)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]

    def _read_party_status(self) -> List[int]:
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/status_constants.asm
        return [
            self._read_m(addr)
            for addr in [0xD16F, 0xD19B, 0xD1C7, 0xD1F3, 0xD21F, 0xD24B]
        ]

    def _read_party_hp(self) -> Dict[str, List[int]]:
        hp = [
            self.read_hp(addr)
            for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ]
        max_hp = [
            self.read_hp(addr)
            for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ]
        return {"current": hp, "max": max_hp}

    def _read_party_xp(self) -> List[List[int]]:
        return [
            self._read_triple(addr)
            for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
        ]

    def read_hp(self, start: int) -> int:
        return 256 * self._read_m(start) + self._read_m(start + 1)

    def _read_caught_pokemon_count(self) -> int:
        return sum([self._bit_count(self._read_m(i)) for i in range(0xD2F7, 0xD30A)])

    def _read_seen_pokemon_count(self) -> int:
        return sum([self._bit_count(self._read_m(i)) for i in range(0xD30A, 0xD31D)])

    def _read_money(self) -> int:
        return (
            100 * 100 * self._read_bcd(self._read_m(0xD347))
            + 100 * self._read_bcd(self._read_m(0xD348))
            + self._read_bcd(self._read_m(0xD349))
        )

    def _read_events(self) -> List[int]:
        event_flags_start = 0xD747
        event_flags_end = 0xD886
        # museum_ticket = (0xD754, 0)
        # base_event_flags = 13
        return [
            self._bit_count(self._read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ]
