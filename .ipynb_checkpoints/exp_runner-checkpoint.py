import sys, getopt, yaml
from Experiment import Experiment 

def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'c:o:')
    except getopt.GetoptError:
        print('invalid args')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-c':
            print(f'config file: {arg}')
            config_file = arg
        if opt == '-o':
            print(f'output directory: {arg}')
            output_dir = arg
            
    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
    except Exception:
        print(f'{config_file} not a valid config file')
        sys.exit(2)
        
    experiment = Experiment(config, output_dir)
    print(experiment)
    sys.exit(0)
        

    

if __name__ == "__main__":
    main(sys.argv[1:])

