'''
Basic data analysis: Can be used to get error and ratio for different sets of captions.
'''

import sys
import json
from data_analysis_common import *
import nltk
import pdb
from bias_detection import *

img_2_anno_dict = create_dict_from_list(pickle.load(open(target_train)))
img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_test))))
img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_val))))

img_2_anno_dict_simple = {}
for key, value in img_2_anno_dict.iteritems():
    id = int(key.split('_')[-1].split('.jpg')[0])
    img_2_anno_dict_simple[id] = {}
    img_2_anno_dict_simple[id]['male'] = value[0]
    img_2_anno_dict_simple[id]['female'] = int(not value[0])  
    assert int(not value[0]) == value[1]
bias_ids = img_2_anno_dict_simple.keys()

def set_predictions(predictions, anno_ids):
    anno_ids = set(anno_ids)
    prediction_subset = []
    for prediction in predictions:
        if prediction['image_id'] in anno_ids:
            prediction_subset.append(prediction)
    pred_images = accuracy(prediction_subset, man_word_list_synonyms, woman_word_list_synonyms, img_2_anno_dict_simple)

confident_man = open(base_dir + '/data2/anja/xai/captions/val-confident-man-500new.txt').readlines()
confident_man = [int(c.strip()) for c in confident_man]
confident_woman = open(base_dir + '/data2/anja/xai/captions/val-confident-woman-500new.txt').readlines()
confident_woman = [int(c.strip()) for c in confident_woman]
confident_ims = set(confident_man + confident_woman) & set(bias_ids)

for caption_path in caption_paths:
    print "Model name: %s" %caption_path[0]
    predictions = json.load(open(caption_path[1]))

    set_predictions(predictions, confident_ims)
    pdb.set_trace() 

