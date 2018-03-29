import sys
import json
from data_analysis_common import *
import nltk
import pdb

bias_ids = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/bias-set.txt').readlines()
bias_ids = set([int(b.strip()) for b in bias_ids])
# find ids in confident subset
confident_subset_ids = []
man_confident_ids = '/home/lisaanne/lev/data2/anja/xai/captions/val-confident-man-500.txt'
woman_confident_ids = '/home/lisaanne/lev/data2/anja/xai/captions/val-confident-woman-500.txt'

with open(woman_confident_ids) as f:
   confident_subset_ids = f.readlines()
with open(man_confident_ids) as f:
   confident_subset_ids += f.readlines()

bias_ids = [int(img_id) for img_id in confident_subset_ids]

def bias_ratios(predictions):
    prediction_subset = []
    for prediction in predictions:
        if prediction['image_id'] in bias_ids:
            prediction_subset.append(prediction)
    other_type_both = 0.
    other_type_person = 0.
    other_type_catchall = 0.
    total_other = 0.
    for prediction in prediction_subset:
        sentence_words = nltk.word_tokenize(prediction['caption'].lower())
        pred_man = is_gendered(sentence_words, 'man', man_word_list_synonyms, woman_word_list_synonyms)
        pred_woman = is_gendered(sentence_words, 'woman', man_word_list_synonyms, woman_word_list_synonyms)
        if pred_man & pred_woman:
            other_type_both += 1
            total_other += 1
        elif (not pred_man) & (not pred_woman):
            total_other += 1
            if len(set(sentence_words) & set(['person', 'people', 'player', 'child'])) > 0:
                other_type_person += 1
            else:
                other_type_catchall += 1
    print "Other type both man and woman: %f" %(other_type_both/total_other)  
    print "Other type gender neutral: %f" %(other_type_person/total_other)  
    print "Other type catch all: %f" %(other_type_catchall/total_other)  
    print "%f\t%f\t%f" %(other_type_both/total_other, other_type_person/total_other, other_type_catchall/total_other)
 
for caption_path in caption_paths:
    print "Model name: %s" %caption_path[0]
    predictions = json.load(open(caption_path[1]))
    bias_ratios(predictions)
    pdb.set_trace() 

