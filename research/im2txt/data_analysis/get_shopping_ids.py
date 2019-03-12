from __future__ import print_function
from data_analysis_base import AnalysisBaseClass

caption_paths = []

analysis_computer = AnalysisBaseClass(caption_paths)

for split in ['dev', 'test', 'train']:

    shopping_split = analysis_computer.get_shopping_split(
        fpath='./data/bias_splits/%s.data' %split
    )
    shopping_split_ids = analysis_computer.convert_filenames_to_ids(
        shopping_split
    )
    write_file = open('./data/bias_splits/%s.ids.txt' %split, 'w')
    for id in shopping_split_ids:
        write_file.writelines("%d\n" %id)
    write_file.close()
