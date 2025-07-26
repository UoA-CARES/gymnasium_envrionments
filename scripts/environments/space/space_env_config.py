from util.configurations import GymEnvironmentConfig


class SpaceEnvironmentConfig(GymEnvironmentConfig):
    traj_filename: str
    impulse_filename: str

    tof: float
    max_thrust: float
    exhaust_vel: float

    dyn_pos_sd: float = 1.0
    dyn_vel_sd: float = 0.05
    dyn_m_sd: float = 1.0
