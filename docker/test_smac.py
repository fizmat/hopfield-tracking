from ConfigSpace import ConfigurationSpace, Configuration
from smac import HyperparameterOptimizationFacade, Scenario


def evaluate(config: Configuration, seed: int = 0) -> float:
    return config['x'] ** 2 + 10 * config['y'] ** 2


def test_smac():
    scenario = Scenario(ConfigurationSpace({'x': (-1., 1.), 'y': (-1., 1.)}),
                        deterministic=True, n_trials=100)
    smac = HyperparameterOptimizationFacade(scenario, evaluate)
    best_config = smac.optimize()
    print(best_config)


if __name__ == '__main__':
    test_smac()
