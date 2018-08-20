from data_analysis_base import AnalysisBaseClass, caption_paths

def get_ids(fname):
    with open(fname, 'r') as f:
        return [int(img_id.split()[0]) for img_id in f.readlines()]

# Crete analysis tools
analysis_computer = AnalysisBaseClass(caption_paths)

# Get datasets
shopping_test_split = analysis_computer.get_shopping_split(
    fpath='../data/test.data'
)
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(
    shopping_test_split
)
analysis_computer.biased_objects('../data/captions_val2014.json', 
                                 shopping_test_split_ids)

