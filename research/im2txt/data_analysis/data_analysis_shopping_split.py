from data_analysis_base import AnalysisBaseClass, caption_paths

analysis_computer = AnalysisBaseClass(caption_paths)
shopping_test_split = analysis_computer.get_shopping_split()
shopping_test_split_ids = analysis_computer.convert_filenames_to_ids(shopping_test_split)

print "\n\n\n Accuracy \n\n\n"

analysis_computer.accuracy(shopping_test_split_ids)

print "\n\n\n TPR/FPR \n\n\n"

analysis_computer.retrieve_accuracy_with_confidence(shopping_test_split_ids)
