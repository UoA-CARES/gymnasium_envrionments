from util.configurations import GymEnvironmentConfig
from envrionments.Pokemon import Pokemon

if __name__ == '__main__':
    args = {
        'gym' : 'pokemon',
        'task' : 'pokemon',
    }
    config = GymEnvironmentConfig(**args)
    env = Pokemon(config)

    env.step(0)

    env.step(1)

    env.step(2)

    env.step(3)
