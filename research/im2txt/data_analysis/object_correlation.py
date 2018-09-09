from data_analysis_base import AnalysisBaseClass

def get_ids(fname):
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# Crete analysis tools
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
analysis_computer = AnalysisBaseClass(caption_paths)

# Get datasets
shopping_dev_split = analysis_computer.get_shopping_split()
shopping_dev_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_dev_split
)
shopping_test_split = analysis_computer.get_shopping_split(
    fpath='../data/bias_splits/test.data'
)
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_test_split
)

print('dev')
analysis_computer.bias_amplification_objects_stats(gt_captions, shopping_dev_split_ids)
print('test')
analysis_computer.bias_amplification_objects_stats('../data/captions_val2014.json', shopping_test_split_ids)
