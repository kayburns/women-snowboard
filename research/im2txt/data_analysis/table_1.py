from data_analysis_base import AnalysisBaseClass, caption_paths

def get_ids(fname):
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# Crete analysis tools
analysis_computer = AnalysisBaseClass(caption_paths)

# Get datasets
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


print "----------- MSCOCO Biased -----------"

print('dev')
analysis_computer.accuracy(shopping_dev_split_ids)
print('test')
analysis_computer.accuracy(shopping_test_split_ids)

print "----------- MSCOCO Balanced -----------"

print('dev')
analysis_computer.accuracy(balanced_dev)

print('test')
analysis_computer.accuracy(balanced_test)

print "----------- MSCOCO Confident -----------"

print('dev')
conf_caps = set(AnalysisBaseClass.get_confident_split())
conf_val_set = set(shopping_dev_split_ids) & conf_caps
conf_test_set = set(shopping_test_split_ids) & conf_caps
analysis_computer.accuracy(
    filter_imgs=conf_val_set
)
print('test')
analysis_computer.accuracy(
    filter_imgs=conf_test_set
)

