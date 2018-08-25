from __future__ import print_function
from data_analysis_base import AnalysisBaseClass
import sys
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute results in Snowboard paper from pretrained model.'
    )
    parser.add_argument(
        '--experiments',
        dest='experiments',
        type=str,
        action='append',
        required=True,
        help='List of experiments to run. Options: table_1_main, table_2_main' \
            'table_3_main, figure_3_main, or all for all experiments'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args() 

def get_ids(fname):
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# --------------------------- Create Splits ---------------------------------- #

# Get caption results
caption_paths = []
base_dir = '../final_captions_eccv2018/'
baseline_ft = ('Baseline-FT', base_dir + 'baseline_ft.json')
caption_paths.append(baseline_ft)
uw = ('UpWeight', base_dir + 'upweight.json')
caption_paths.append(uw)
balanced = ('Balanced', base_dir + 'balanced.json')
caption_paths.append(balanced)
confident =  ('Equalizer w/o ACL', base_dir + 'confident.json')
caption_paths.append(confident)
acl = ('Equalizer w/o Confident', base_dir + 'confusion.json')
caption_paths.append(acl) 
equalizer = ('Equalizer', base_dir + 'equalizer.json')
caption_paths.append(equalizer)

# Crete analysis tools
analysis_computer = AnalysisBaseClass(caption_paths)

# Get dataset splits; shopping refers to 'Men Also Like Shopping'
shopping_dev_split = analysis_computer.get_shopping_split()
shopping_dev_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_dev_split
)
shopping_test_split = analysis_computer.get_shopping_split(
    fpath='../data/test.data'
)
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_test_split
)

balanced_dev = get_ids('balanced_split/val_man.txt') + \
    get_ids('balanced_split/val_woman.txt') 

balanced_test = get_ids('balanced_split/test_man.txt') + \
    get_ids('balanced_split/test_woman.txt') 

datasets = [
    ('COCO Bias (Val)',shopping_dev_split_ids),
    ('COCO Bias (Test)',shopping_test_split_ids),
    ('COCO Balanced (Val)',balanced_dev),
    ('COCO Balanced (Test)',balanced_test)
]

# ------------------------- Helpers to Run Experiments ----------------------- #

def table_1_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        print('Model\tError\tRatio Difference')
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print(model_results['all_incorrect'], end='\t')
            print(model_results['gt_ratio'] - model_results['ratio'])

def table_2_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        print(
            'Model\tWomen Correct\tWomen Incorrect\tWomen Other\t' \
            'Men Correct\tMen Incorrect\tMen Other'
        )
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print(model_results['female_correct'], end='\t')
            print(model_results['female_incorrect'], end='\t')
            print(model_results['female_other'], end='\t')
            print(model_results['male_correct'], end='\t')
            print(model_results['male_incorrect'], end='\t')
            print(model_results['male_other'])

def table_3_main():
    print("TODO")

def figure_3_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        analysis_computer.retrieve_accuracy_with_confidence(id_list)

def table_1_supp():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        analysis_computer.biased_objects(caption_path, id_list)

def table_2_supp():
    print("TODO")

experiment_functions = OrderedDict([
    ('table_1_main',table_1_main),
    ('table_2_main',table_2_main),
    ('table_3_main',table_3_main),
    ('figure_3_main',figure_3_main),
    ('table_1_supp',table_1_supp),
])
experiment_names = experiment_functions.keys()

if __name__ == '__main__':
    args = parse_args()
    if args.experiments[0] == "all":
        args.experiments = experiment_names
    for experiment in args.experiments:
        if experiment not in experiment_names:
            raise ValueError('please use name in: '+' '.join(experiment_names))
        print("###################### "+experiment+" ######################")
        experiment_functions[experiment]()
