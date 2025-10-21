"""
Author: Lee Violet Ong
Date: 15/10/25 (v1.2)

Environment that resets to a single node but with varying noise portfolios.
"""

# Directory setup
import os
import rl_corrective_gym

file_dir = os.path.dirname(rl_corrective_gym.nominal_trajectory.__file__)

from functools import cached_property
import copy
import io

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from PIL import Image
from daceypy import DA, array

from rl_corrective_gym.gym_env_setup.space_env_config import SpaceEnvironmentConfig
from rl_corrective_gym.RK78 import RK78
from environments.gym_environment import GymEnvironment

# CONSTANTS
AU = 1.49597870691e8  # km
DAY = 86400


class CorrectiveTransferEnvironment(gym.Env, GymEnvironment):
    def __init__(
        self,
        config: SpaceEnvironmentConfig,
    ):
        super().__init__(config)

        traj_filename: str = config.traj_filename
        impulse_filename: str = config.impulse_filename

        # define universal parameters
        self.sun_mu: float = 1.32712440018e11
        self.ve: float = np.sqrt(self.sun_mu / AU)  # orbital velocity of earth

        # define required information from SCP data
        # [pos (km), vel (km/s), m (kg)]
        self.nominal_traj: np.ndarray = pd.read_csv(
            os.path.join(file_dir, traj_filename)
        ).to_numpy()

        self.num_timesteps: int = len(self.nominal_traj) - 1  # no-dim
        self.max_m: float = self.nominal_traj[0, -1]  # kg

        # applied impulse [vel (km/s)]
        self.nominal_imp: np.ndarray = pd.read_csv(
            os.path.join(file_dir, impulse_filename)
        ).to_numpy()

        # task config (doi: 10.1016/j.actaastro.2023.10.018)
        self.tof: float = config.tof  # in days
        self.timestep: float = self.tof / self.num_timesteps * DAY  # in seconds

        # dynamics uncertainties config (in km, km/s)
        # TODO: investigate the impact of pos/vel to the final deviation
        self.dyn_pos_sd: float = config.dyn_pos_sd
        self.dyn_vel_sd: float = config.dyn_vel_sd

        # thruster config
        self.max_thrust: float = config.max_thrust  # N, kg km/s^2
        self.exhaust_vel: float = config.exhaust_vel  # km/s
        self.max_corr: float = config.max_corr  # km/s

        # reward function config
        self.dyn_rew: int = config.dyn_rew
        self.effort_rew: int = config.effort_rew

        # init state is the first state by default (no noise)
        self.chosen_timestamp: int = self.num_timesteps - 1
        self.nom_imp: np.ndarray = self.nominal_imp[self.chosen_timestamp, :]

        # the following are variables that will get UPDATED
        self.state: np.ndarray = self.nominal_traj[self.chosen_timestamp, :]
        self.vmax: float = self._get_vmax()
        self.noise: np.ndarray = np.array([0] * 7)

        # define the spaces ie. all possible range of obs and action
        # [nom pos, nom vel, nom m, nom imp]
        nom_constraints: np.ndarray = np.array(
            [AU, AU, AU, self.ve, self.ve, self.ve, 0.0, self.ve, self.ve, self.ve]
        )
        # [act pos, act vel, act m]
        act_constraints: np.ndarray = np.array(
            [AU, AU, AU, self.ve, self.ve, self.ve, 0.0]
        )

        self.observation_space: spaces.Box = spaces.Box(
            low=-2 * np.concatenate((nom_constraints, act_constraints)),
            high=2 * np.concatenate((nom_constraints, act_constraints)),
            dtype=np.float64,
        )

        self.action_config: int = config.action_config
        self._init_action_space()

        # logging purposes
        self._init_logs()

    @cached_property
    def max_action_value(self) -> float:
        return self.action_space.high[0]

    @cached_property
    def min_action_value(self) -> float:
        return self.action_space.low[0]

    @cached_property
    def observation_space_shape(self) -> int:
        return self.observation_space.shape[0]

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.action_space, spaces.Box):
            action_num = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            action_num = self.action_space.n
        else:
            raise ValueError(f"Unhandled action space type: {type(self.action_space)}")
        return action_num

    def sample_action(self) -> int:
        return self.action_space.sample()

    def set_seed(self, seed: int) -> None:
        super().reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.action_space.seed(seed)
        # important for timestep replicability
        np.random.seed(seed)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        """
        Frame refers to the plot of the desired trajectory as well as the currentn
        guid and noguid trajectories.

        Called for collection of state images in the record class. Called twice:
        - after reset (shows the chosen state)
        - after step (show the whole propagation results)
        """
        dpi: int = 100  # dots per inches
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.plot(
            self.nominal_traj[:, 0], self.nominal_traj[:, 1], self.nominal_traj[:, 2]
        )
        ax.plot(
            self.gui_log_pos[:, 0], self.gui_log_pos[:, 1], self.gui_log_pos[:, 2], "r"
        )
        ax.plot(
            self.nogui_log_pos[:, 0],
            self.nogui_log_pos[:, 1],
            self.nogui_log_pos[:, 2],
            "g",
        )

        # # convert the plot into numpy
        buf: io.BytesIO = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)

        frame: np.ndarray = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        fig.clear()

        return frame

    def get_overlay_info(self) -> dict:
        return {}

    def reset(self, *, training: bool = True) -> np.ndarray:
        super().reset()
        self._init_logs()
        self._init_state()

        return self.state

    def step(self, action) -> tuple:

        if self.action_config == 0:
            total_imp: np.ndarray = action
        elif self.action_config == 1:
            total_imp: np.ndarray = self.nom_imp + action
        else:
            A: np.ndarray = np.reshape(action, (6, 3))
            total_imp: np.ndarray = self._pseudo_optimal_control(A) + self.nom_imp

        ximp: np.ndarray = self.nominal_imp[-1, :]
        xf: np.ndarray = copy.deepcopy(self.nominal_traj[-1, :])  # xf+, inc nom imp
        xf[3:6] -= ximp  # xf-, w/o nom imp

        gui_xf: np.ndarray = self._propagate(True, total_imp)  # gui_xf-
        ngui_xf: np.ndarray = self._propagate(False)  # ngui_xf-
        gui_err: np.ndarray = gui_xf - xf
        ngui_err: np.ndarray = ngui_xf - xf

        rewards = self._reward_function(
            self.vmax, total_imp, gui_err[0:6], ngui_err[0:6]
        )

        # terminal state, reward, done, truncated, info
        info: dict = {
            "reward_dyn": rewards["dynamics"],
            "reward_effort": rewards["effort"],
            "reward_misc": rewards["misc"],
            "timestep": self.chosen_timestamp,
            "noise": self.noise,
            "vmax": self.vmax,
            "action": action,
            "gui_err": gui_err,
            "ngui_err": ngui_err,
        }

        self.state = np.concatenate((xf, ximp, gui_xf))
        return self.state, rewards["total"], True, False, info

    # =================== HELPER FUNCTIONS ========================
    def _init_action_space(self):
        """
        Initialises the action space. Currently has three variants:
            1. Total impulse applied to the perturbed state (replaces the nominal)
            2. Corrective impulse applied to the perturbed state (on top of the nominal)
            3. Gain matrix (6x3) used to for the least squared solution
        """
        if self.action_config == 0 or self.action_config == 1:
            self.action_space: spaces.Box = spaces.Box(
                low=np.array(3 * [-self.vmax]),
                high=np.array(3 * [self.vmax]),
                dtype=np.float64,
            )
        else:
            # TODO: investigate threshold to deviations for better representation here
            self.action_space: spaces.Box = spaces.Box(
                low=np.concatenate((np.array(9 * [-AU]), np.array(9 * [-self.ve]))),
                high= np.concatenate((np.array(9 * [AU]), np.array(9 * [self.ve]))),
                dtype=np.float64,
            )

    def _reward_function(
        self,
        vmax: float,
        total_imp: np.ndarray,
        gui_err: np.ndarray,
        ngui_err: np.ndarray,
    ) -> dict:
        reward_dyn: float = self._reward_dynamics(gui_err, ngui_err, vmax)
        reward_effort: float = self._reward_effort(total_imp)
        reward_misc: float = self._reward_misc(total_imp, vmax)

        total_reward: float = reward_dyn + reward_effort + reward_misc

        return {
            "total": total_reward,
            "dynamics": reward_dyn,
            "effort": reward_effort,
            "misc": reward_misc,
        }

    def _reward_dynamics(
        self, gui_err: np.ndarray, ngui_err: np.ndarray, vmax: float
    ) -> float:
        """
        Computes the reward associated to the state deviation.

        Arguments:
        - guid_err: 6-dim deviation vector of guided xf (w/ corrective imp) from desired xf
        - no_guid_err: 6-dim deviation vector of non-guided xf from desired xf
        """

        reward: float = 0.0

        gui_norm: float = np.linalg.norm(gui_err)
        ngui_norm: float = np.linalg.norm(ngui_err)

        if self.dyn_rew == 0:
            # corresponds to reward function 1
            err_prop: float = gui_norm / ngui_norm
            p: float = 1  # regularisation variable for steepness/sensitivity
            reward = (1 - err_prop**p) / (1 + err_prop**p)

        elif self.dyn_rew == 1:
            # corresponds to reward function 2
            tol_2: float = 1e6
            reward = tol_2 / (tol_2 + gui_norm) - 1

        elif self.dyn_rew == 2:
            # corresponds to reward function 3
            ngui_perr: float = np.linalg.norm(ngui_err[0:3])
            prew: float = min(np.linalg.norm(gui_err[0:3]), ngui_perr) / ngui_perr
            vrew: float = min(np.linalg.norm(gui_err[3:6]), vmax) / vmax

            reward = (3 / (1 + prew + vrew)) - 2
        else:
            assert False, "No such dynamics reward function"

        return reward

    def _reward_effort(self, total_imp: np.ndarray) -> float:
        """
        Computes the control effort reward. (Optional i.e. can be NONE if config = 0)

        Arguments:
        - control_imp: the control impulse chosen
        """

        reward: float = 0.0

        if self.effort_rew == 0:  # NONE
            pass

        elif self.effort_rew == 1:
            # corresponds to reward function 6
            total_norm: float = np.linalg.norm(total_imp)

            unit_dir: np.ndarray = total_imp / total_norm
            nom_norm: float = np.dot(self.nom_imp, unit_dir)
            corr_norm: float = total_norm - nom_norm

            reward = -corr_norm / self.max_corr

        else:
            assert False, "No such control effort reward function"

        return reward

    def _reward_misc(self, total_imp: np.ndarray, vmax: float) -> float:
        """
        Computes the miscs rewards i.e. must haves for constraints.
        Current content: control penalty.
        TODO: bitmask when content increases

        Arguments:
        - control_imp: the control impulse chosen
        - vmax: norm of the total possible imp at the timestep
        """

        reward: float = 0.0
        over_imp: float = np.linalg.norm(total_imp) - vmax

        if over_imp > 0:
            reward = 1 / (1 + over_imp) - 1

        return reward

    def _mass_update(self, m0: float, impulse: np.ndarray) -> float:
        """
        Implements the Tsiolkovsky Rocket Equation for the mass update.

        Arguments:
        - m0: the current mass at t before impulse (kg)
        - impulse: the total impulse vector at t (km/s)
        """
        return m0 * np.exp(-np.linalg.norm(impulse) / self.exhaust_vel)

    def _get_vmax(self) -> float:
        """
        Computes the norm of the maximum impulse, using the Tsiolvosky equation.
        """
        m0: float = self.state[-1]  # kg
        return self.exhaust_vel * np.log(
            (m0 * self.exhaust_vel)
            / (m0 * self.exhaust_vel - self.max_thrust * self.timestep)
        )  # km/s

    def _propagate(
        self, is_guid: bool, action: np.ndarray = [0.0, 0.0, 0.0]
    ) -> np.ndarray:
        """
        Propagates the chosen global state to the terminal timestep.
        Returns the terminal state.
        """

        total_impulse: np.ndarray = copy.deepcopy(
            self.nominal_imp[self.chosen_timestamp]
        )  # km/s
        pos: np.ndarray = copy.deepcopy(self.state[10:13])  # km
        vel: np.ndarray = copy.deepcopy(self.state[13:16])  # km/s (w/o imp)
        m: float = copy.deepcopy(self.state[-1])  # kg

        if is_guid:
            total_impulse = action

        vel += total_impulse

        # logging
        self._update_logs(is_guid, pos, vel, m)

        m = self._mass_update(m, total_impulse)
        pos, vel = np.array(
            pk.propagate_lagrangian(
                r0=pos,
                v0=vel,
                tof=self.timestep,
                mu=self.sun_mu,
            )
        )

        for i in range(self.chosen_timestamp + 1, self.num_timesteps + 1):
            nominal_impulse = self.nominal_imp[i]
            vel += nominal_impulse

            self._update_logs(is_guid, pos, vel, m)

            if i == self.num_timesteps:
                vel -= nominal_impulse
                break

            m = self._mass_update(m, nominal_impulse)
            pos, vel = np.array(
                pk.propagate_lagrangian(
                    r0=pos,
                    v0=vel,
                    tof=self.timestep,
                    mu=self.sun_mu,
                )
            )

        # return a dictionary of terminal state
        return np.concatenate((pos, vel, [m]))

    def _update_logs(self, is_guid: bool, pos: np.ndarray, vel: np.ndarray, m: float):

        if is_guid:
            self.gui_log_pos = np.append(self.gui_log_pos, [pos], axis=0)
            self.gui_log_vel = np.append(self.gui_log_vel, [vel], axis=0)
            self.gui_log_m = np.append(self.gui_log_m, m)
        else:
            self.nogui_log_pos = np.append(self.nogui_log_pos, [pos], axis=0)
            self.nogui_log_vel = np.append(self.nogui_log_vel, [vel], axis=0)
            self.nogui_log_m = np.append(self.nogui_log_m, m)

    def _init_logs(self):
        self.gui_log_pos = self.nominal_traj[0 : self.chosen_timestamp, 0:3]
        self.gui_log_vel = self.nominal_traj[0 : self.chosen_timestamp, 3:6]
        self.gui_log_m = self.nominal_traj[0 : self.chosen_timestamp, -1]

        self.nogui_log_pos = self.nominal_traj[0 : self.chosen_timestamp, 0:3]
        self.nogui_log_vel = self.nominal_traj[0 : self.chosen_timestamp, 3:6]
        self.nogui_log_m = self.nominal_traj[0 : self.chosen_timestamp, -1]

    def _init_state(self):
        """
        Initialises the global state ie. choses the timestep and perturbation applied.
        """
        # NOTE: current implementation looks at resetting at the second last node
        chosen_state: np.ndarray = copy.deepcopy(
            self.nominal_traj[self.chosen_timestamp, :]
        )  # inc. imp
        chosen_state[3:6] -= self.nom_imp

        # covariance matrix set up
        pos_var: float = self.dyn_pos_sd**2
        vel_var: float = self.dyn_vel_sd**2
        cov: np.ndarray = np.diag(
            [pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]
        )
        mean: np.ndarray = np.array([0] * 6)

        # choose the gaussian noise for the chosen state
        self.noise = np.concatenate(
            (np.random.multivariate_normal(mean, cov), np.array([0]))
        )
        self.state = np.concatenate(
            (chosen_state, self.nom_imp, chosen_state + self.noise)
        )

    # ================== DEBUGGNG FUNCTIONS ========================
    def _dynamics(self, t, x: array, params) -> array:
        # Keplerian 2-body equations of motions
        # Makes use of daceypy array struct
        pos: array = x[0:3]
        vel: array = x[3:6]

        pos_norm: float = pos.vnorm()
        v_dot: array = -(self.sun_mu / pos_norm**3) * pos

        return vel.concat(v_dot)

    def _stm_pert(self):
        """
        This function utilises DA to obtain the STM (first-order).
        """
        DA.init(1, 6)

        tf: float = (self.num_timesteps - self.chosen_timestamp) * self.timestep
        # define the DA variables - in this case, the variables
        # are the EOM variables themselves
        chosen_state: np.ndarray = self.nominal_traj[self.chosen_timestamp, :][0:6]
        x0: array = array(chosen_state + [DA(1), DA(2), DA(3), DA(4), DA(5), DA(6)])

        with DA.cache_manager():
            xf_DA = RK78(x0, 0.0, tf, self._dynamics, None)

        return xf_DA.linear()

    def _optimal_control(self) -> np.ndarray:
        """
        Computes the least squares solution for the optimal control to reduce
        the deviation:

        delta(v) = -(A_T @ A)^(-1) @ A_T @ x
        where x is the error to correct
        """
        full_phi: np.ndarray = self._stm_pert()
        # A is the second half of the STM (6x3), impact of vel dev
        A: np.ndarray = full_phi[:, 3:6]
        print(A)
        A_T: np.ndarray = np.transpose(A)

        return -(np.linalg.inv(A_T @ A) @ A_T) @ self.noise[0:6] - self.noise[3:6]

    def _pseudo_optimal_control(self, A: np.ndarray) -> np.ndarray:
        """
        Computes the least squares solution for the optimal control to reduce
        the deviation:

        delta(v) = -(A_T @ A)^(-1) @ A_T @ x
        where x is the error to correct
        """
        A_T: np.ndarray = np.transpose(A)

        try:
            opt_imp: np.ndarray = -(np.linalg.inv(A_T @ A) @ A_T) @ self.noise[0:6] - self.noise[3:6]
        except np.linalg.LinAlgError:
            opt_imp: np.ndarray = np.array([0.0, 0.0, 0.0])

        return opt_imp
