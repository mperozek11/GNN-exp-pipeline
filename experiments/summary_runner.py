import sys, getopt
from SetSummarizer import SetSummarizer


def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'hd:')
    except getopt.GetoptError:
        print('invalid args')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print(f'\nThis tool generates config.yml files for every permutation of a top-level config file')
            print('\nRequired flags: \n -d results directory to summarize. This directory should include "result.yaml" files to be summarized\n')
            sys.exit(0)
        if opt == '-d':
            print(f'result directory: {arg}')
            RESULT_DIR = arg
        
    summarizer = SetSummarizer(RESULT_DIR=RESULT_DIR)
    exp_summary = summarizer.summarize(RESULT_DIR)
    summarizer.visualize()
    sys.exit(0)
        

    

if __name__ == "__main__":
    main(sys.argv[1:])

