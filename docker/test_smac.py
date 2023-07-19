from ConfigSpace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

def evaluate(config):
    return config['x']**2 + 10*config['y']**2

def main():
    scenario = Scenario({
            'run_obj': 'quality',
            'runcount-limit': 10,
            'cs': ConfigurationSpace({'x': (-1., 1.), 'y': (-1., 1.)})
        })
    optimizer = SMAC4BB(scenario=scenario, tae_runner=evaluate)
    best_config = optimizer.optimize()
    print(best_config)

if __name__ == '__main__':
    main()