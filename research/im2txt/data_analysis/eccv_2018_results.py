from __future__ import print_function
from data_analysis_base import AnalysisBaseClass
import sys
import argparse
from collections import OrderedDict
import numpy as np

# ------------------------- Helpers to Run Experiments ----------------------- #

def table_1_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        print('Model\tError\tRatio Difference')
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print("{0:.2f}".format(model_results['all_incorrect']*100),end='\t')
            print("{0:.2f}".format(
                model_results['gt_ratio']-model_results['ratio']
            ))

def table_2_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        print(
            'Model\tWomen Correct\tWomen Incorrect\tWomen Other\t' \
            'Men Correct\tMen Incorrect\tMen Other\tOutcome Divergence'
        )
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print("{0:.2f}".format(model_results['female_correct']*100),end='\t')
            print("{0:.2f}".format(model_results['female_incorrect']*100),end='\t')
            print("{0:.2f}".format(model_results['female_other']*100),end='\t')
            print("{0:.2f}".format(model_results['male_correct']*100),end='\t')
            print("{0:.2f}".format(model_results['male_incorrect']*100),end='\t')
            print("{0:.2f}".format(model_results['male_other']*100), end='\t')
            print("{0:.3f}".format(model_results['outcome_divergence']))

def table_3_main():
    print('=============== Grad-Cam ')
    for split_name, img_paths in datasets_pointing:
        print('---------------------- %s ----------------------' % split_name)
        img_path_w = img_paths[0]
        img_path_m = img_paths[1]
        if 'val' in img_path_w: save_path = save_path_gradcam % 'val'
        elif 'test' in img_path_w: save_path = save_path_gradcam % 'test'
        else: assert(False)
        all_results = analysis_computer.pointing(img_path_w, img_path_m, vocab_file, save_path, checkpoint_path, 'gradcam')
        print('Model\tWoman\tMan\tAll')
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print("{0:.2f}".format(model_results['woman']*100),end='\t')
            print("{0:.2f}".format(model_results['man']*100),end='\t')
            print("{0:.2f}".format(model_results['all']*100))

    print('=============== Saliency ')
    for split_name, img_paths in datasets_pointing:
        print('---------------------- %s ----------------------' % split_name)
        img_path_w = img_paths[0]
        img_path_m = img_paths[1]
        if 'val' in img_path_w: save_path = save_path_saliency % 'val'
        elif 'test' in img_path_w: save_path = save_path_saliency % 'test'
        else: assert(False)
        all_results = analysis_computer.pointing(img_path_w, img_path_m, vocab_file, save_path, checkpoint_path, 'saliency')
        print('Model\tWoman\tMan\tAll')
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print("{0:.2f}".format(model_results['woman']*100),end='\t')
            print("{0:.2f}".format(model_results['man']*100),end='\t')
            print("{0:.2f}".format(model_results['all']*100))


def figure_3_main():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_res = analysis_computer.retrieve_accuracy_with_confidence(id_list)
        for model, model_results in all_res.iteritems():
            print('model name: %s' % model)
            print('confidence\taccuracy')
            for i, result_i in enumerate(model_results):
                print(i+1, end='\t')
                print("{0:.2f}".format(result_i*100))

def table_1_supp():
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        results = analysis_computer.biased_objects(
            './data/captions_only_valtrain2014.json', id_list
        )
        for word, word_results in results.iteritems():
            print ('For word: %s' % word)
            gt_ratio = word_results.pop('gt_ratio')
            print ("Model name\tError\tDelta Ratio") 
	    for model, model_results in word_results.iteritems():
                print(model, end='\t')
                print("{0:.3f}".format(model_results['error']),end='\t')
                print("{0:.3f}".format(
                    np.abs(gt_ratio-model_results['delta_ratio'])
                ))
                print ("\n")

def table_2_supp():
    # Get caption results
    caption_paths = []
    base_dir = './final_captions_eccv2018/'
    baseline_ft = ('Baseline-FT', base_dir + 'baseline_ft.json')
    caption_paths.append(baseline_ft)
    equalizer = ('Equalizer', base_dir + 'equalizer.json')
    caption_paths.append(equalizer)
    equalizer_w_sets = ('Equalizer w/ Sets', base_dir + 'equalizer_w_sets.json')
    caption_paths.append(equalizer_w_sets)
    # Create analysis tools
    analysis_computer = AnalysisBaseClass(caption_paths)
    for split_name, id_list in datasets:
        print('---------------------- %s ----------------------' % split_name)
        all_results = analysis_computer.accuracy(id_list)
        print('Model\tError\tRatio Difference')
        for model, model_results in all_results.iteritems():
            print(model, end='\t')
            print("{0:.2f}".format(model_results['all_incorrect']*100),end='\t')
            print("{0:.2f}".format(
                model_results['gt_ratio']-model_results['ratio']
            ))

# --------------------------- Parse Arguments ---------------------------------- #

experiment_functions = OrderedDict([
    ('table_1_main',table_1_main),
    ('table_2_main',table_2_main),
    ('table_3_main',table_3_main),
    ('figure_3_main',figure_3_main),
    ('table_1_supp',table_1_supp),
    ('table_2_supp',table_2_supp)
])
experiment_names = experiment_functions.keys()

# Get caption results
caption_paths = []
base_dir = './final_captions_eccv2018/'
baseline_ft = ('Baseline-FT', base_dir + 'baseline_ft.json')
caption_paths.append(baseline_ft)
balanced = ('Balanced', base_dir + 'balanced.json')
caption_paths.append(balanced)
uw = ('UpWeight', base_dir + 'upweight.json')
caption_paths.append(uw)
confident =  ('Equalizer-w/o-ACL', base_dir + 'confidence.json')
caption_paths.append(confident)
acl = ('Equalizer-w/o-Confident', base_dir + 'confusion.json')
caption_paths.append(acl) 
equalizer = ('Equalizer', base_dir + 'equalizer.json')
caption_paths.append(equalizer)

all_model_names = [caption_path[0] for caption_path in caption_paths]

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
        help='List of experiments to run. Options: '+' '.join(experiment_names)
    )
    parser.add_argument(
        '--models',
        dest='models',
        type=str,
        nargs='+',
        default= all_model_names,
        help='List of models to test. Options: '+' '.join(all_model_names)
    )
    parser.add_argument(
        '--splits',
        dest='splits',
        type=str,
        nargs='+',
        default= ['val', 'test'],
        help='List of splits. Options: val, test'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args() 
    
args = parse_args()

caption_paths = [caption_path for caption_path in caption_paths \
                 if caption_path[0] in args.models]
# --------------------------- Create Splits ---------------------------------- #

def get_ids(fname):
    """Fetch ids from 'Men Also Like Shopping' split files"""
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# Create analysis tools
analysis_computer = AnalysisBaseClass(caption_paths)

# Get dataset splits; shopping refers to 'Men Also Like Shopping'
shopping_dev_split = analysis_computer.get_shopping_split()
shopping_dev_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_dev_split
)
shopping_test_split = analysis_computer.get_shopping_split(
    fpath='./data/bias_splits/test.data'
)
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_test_split
)

balanced_dev = get_ids('./data/balanced_split/val_man.txt') + \
    get_ids('./data/balanced_split/val_woman.txt') 

balanced_test = get_ids('./data/balanced_split/test_man.txt') + \
    get_ids('./data/balanced_split/test_woman.txt') 

datasets = []
if 'val' in args.splits:
    datasets.extend([
        ('COCO Bias (Val)',shopping_dev_split_ids),
        ('COCO Balanced (Val)',balanced_dev)
    ])
if 'test' in args.splits:
    datasets.extend([
        ('COCO Bias (Test)',shopping_test_split_ids),
        ('COCO Balanced (Test)',balanced_test)
    ])
# For pointing
datasets_pointing = []
if 'val' in args.splits:
    datasets_pointing.append(
        ('COCO Balanced (Val)', ['./data/balanced_split/val_woman.txt', './data/balanced_split/val_man.txt']))
if 'test' in args.splits:
    datasets_pointing.append(
        ('COCO Balanced (Test)', ['./data/balanced_split/test_woman.txt', './data/balanced_split/test_man.txt']))

vocab_file = './data/word_counts.txt'
save_path_gradcam = './results_gradcam_%s_gt/' # %s -- val test 
# save_path_gradcam = './results_gradcam_%s_pred/' # %s -- val test
save_path_saliency = './results_saliency_%s_gt/' # %s -- val test
checkpoint_path = "./final_weights_eccv2018/%s/train/model.ckpt-1500000" # %s -- model name


if __name__ == '__main__':
    args = parse_args()
    if args.experiments[0] == "all":
        args.experiments = experiment_names
    elif args.experiments[0] == "all-but-pointing":
        experiment_names.remove('table_3_main')
        args.experiments = experiment_names
    for experiment in args.experiments:
        if experiment not in experiment_names:
            raise ValueError('please use name in: '+' '.join(experiment_names))
        print('\n\nRESULTS: '+experiment)
        experiment_functions[experiment]()
