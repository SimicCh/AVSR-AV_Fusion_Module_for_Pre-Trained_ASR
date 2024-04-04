import yaml
import sys
from utils.utils import test


if __name__ == "__main__":

    config_fn = sys.argv[1]

    with open(config_fn, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    test(config)
