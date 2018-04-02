'''
Code to run eval for sentences which include specific nouns.
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

def bias_ratios(predictions, anno_ids):
#    predictions = json.load(open(caption_file))
    anno_ids = set(anno_ids)
    prediction_subset = []
    for prediction in predictions:
        if prediction['image_id'] in anno_ids:
            prediction_subset.append(prediction)
    #count man words and woman words
    man_count = 0.
    woman_count = 0.
    other_count = 0.
    for prediction in prediction_subset:
        sentence_words = nltk.word_tokenize(prediction['caption'].lower())
        pred_man = is_gendered(sentence_words, 'man', man_word_list_synonyms, woman_word_list_synonyms)
        pred_woman = is_gendered(sentence_words, 'woman', man_word_list_synonyms, woman_word_list_synonyms)
        if pred_man & pred_woman:
            other_count += 1
        elif pred_man:
            man_count += 1
        elif pred_woman:
            woman_count += 1
        else:
            other_count += 1
    print "Counts:"
    print "Man: %d/%d (%f)." %(man_count, len(prediction_subset), man_count/len(prediction_subset))
    print "Woman: %d/%d (%f)." %(woman_count, len(prediction_subset), woman_count/len(prediction_subset))
    print "Other: %d/%d (%f)." %(other_count, len(prediction_subset), other_count/len(prediction_subset))
    print "%d\t%d\t%d\t%d" % (man_count, woman_count, other_count, len(prediction_subset))

def bias_word_ratios(predictions, anno_ids, word):
#    predictions = json.load(open(caption_file))
    anno_ids = set(anno_ids)
    prediction_subset = []
    for prediction in predictions:
        if prediction['image_id'] in anno_ids:
            prediction_subset.append(prediction)
    #count man words and woman words
    man_word_count = 0.
    woman_word_count = 0.
    man_count = 0.
    woman_count = 0.
    for prediction in prediction_subset:
        sentence_words = nltk.word_tokenize(prediction['caption'].lower())
        pred_man = is_gendered(sentence_words, 'man', man_word_list_synonyms, woman_word_list_synonyms)
        pred_woman = is_gendered(sentence_words, 'woman', man_word_list_synonyms, woman_word_list_synonyms)
        word_bool = word in sentence_words
        if pred_man & pred_woman:
            pass
        elif pred_man:
            man_count += 1
            if word_bool:
                man_word_count += 1
        elif pred_woman:
            woman_count += 1
            if word_bool:
                woman_word_count += 1

    print "Percent co-occurence gender and word:"
    percent_coocurrence_man = man_word_count/man_count
    percent_coocurrence_woman = woman_word_count/woman_count
    print "Man: %d/%d (%f)." %(man_word_count, man_count, percent_coocurrence_man)
    print "Woman: %d/%d (%f)." %(woman_word_count, woman_count, percent_coocurrence_woman)
    print "%f\t%f" % (percent_coocurrence_man, percent_coocurrence_woman)

def bias_predictions(predictions, anno_ids):
#    predictions = json.load(open(caption_file))
    anno_ids = set(anno_ids)
    prediction_subset = []
    for prediction in predictions:
        if prediction['image_id'] in anno_ids:
            prediction_subset.append(prediction)
    pred_images = accuracy(prediction_subset, man_word_list_synonyms, woman_word_list_synonyms, img_2_anno_dict_simple)

bias_words = ['umbrella', 'kitchen', 'cell', 'table', 'food', 'skateboard', 'baseball', 'tie', 'motorcycle', 'snowboard']
          
for bias_word in bias_words:
   print "Bias word is: %s" %bias_word
   for caption_path in caption_paths:
       anno_ids = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/intersection_%s_person.txt' %bias_word).readlines()
       print "Model name: %s" %caption_path[0]
       anno_ids = [int(a.strip()) for a in anno_ids]
       predictions = json.load(open(caption_path[1]))
       #bias_ratios(predictions, anno_ids)
       bias_word_ratios(predictions, anno_ids, bias_word)
       pdb.set_trace() 
