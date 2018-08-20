from __future__ import print_function
from data_analysis_base import AnalysisBaseClass, caption_paths

def get_ids(fname):
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# Crete analysis tools
analysis_computer = AnalysisBaseClass(caption_paths)

# Get datasets, shopping refers to 'Men Also Like Shopping'
shopping_dev_split = analysis_computer.get_shopping_split()
shopping_dev_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_dev_split
)
shopping_test_split = analysis_computer.get_shopping_split(
    fpath='/data1/caption_bias/models/research/im2txt/'
    'im2txt/data/raw-data/reducingbias/data/COCO/test.data'
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

for split_name, id_list in datasets:
    print('---------------------- %s ----------------------' % split_name)
    all_results = analysis_computer.accuracy(id_list)
    print('Model\tError\tRatio Difference')
    for model, model_results in all_results.iteritems():
        print(model, end='\t')
        print(model_results['all_incorrect'], end='\t')
        print(model_results['gt_ratio'] - model_results['ratio'])

