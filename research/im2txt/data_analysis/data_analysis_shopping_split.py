from data_analysis_base import AnalysisBaseClass, caption_paths

analysis_computer = AnalysisBaseClass(caption_paths)
shopping_dev_split = analysis_computer.get_shopping_split()
shopping_dev_split_ids = analysis_computer.convert_filenames_to_ids(shopping_dev_split)
shopping_test_split = analysis_computer.get_shopping_split(fpath='/data1/caption_bias/models/research/im2txt/im2txt/data/raw-data/reducingbias/data/COCO/test.data')
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(shopping_test_split)

print "\n\n\n Accuracy \n\n\n"
print('dev')
analysis_computer.accuracy(shopping_dev_split_ids)
print('test')
analysis_computer.accuracy(shopping_test_split_ids)
"""
print "\n\n\n TPR/FPR \n\n\n"

print('dev')
analysis_computer.retrieve_accuracy_with_confidence(shopping_dev_split_ids)

print('test')
analysis_computer.retrieve_accuracy_with_confidence(shopping_test_split_ids) 
"""

