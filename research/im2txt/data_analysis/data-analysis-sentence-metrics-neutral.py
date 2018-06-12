import sys
import json
from data_analysis_common import *
import nltk
import pdb
from bias_detection import *
sys.path.append('/data1/caption_bias/models/research/im2txt/coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os

#base_dir = '/home/lisaanne/lev/'
base_dir = ''
#create person set gt

gt_path = base_dir + '/data1/caption_bias/models/research/im2txt/coco-caption/annotations/captions_val2014.json'

gt_path_person = gt_path.replace('.json', '.person.json')

gendered_words = set(man_word_list_synonyms + woman_word_list_synonyms)

if not os.path.exists(gt_path_person):
    gt_caps = json.load(open(gt_path))
    annotations = gt_caps['annotations']
    for cap in annotations:
        words = nltk.word_tokenize(cap['caption'].lower())
        words = ['person' if word in gendered_words else word for word in words] 
        cap['caption'] = ' '.join(words)
    gt_caps['annotations'] = annotations 
    with open(gt_path_person, 'w') as outfile:
        json.dump(gt_caps, outfile)

coco = COCO(gt_path_person)


target_set = 'biased' #options are biased, balanced, confident

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

if target_set == 'biased': 
    image_set = bias_ids
elif target_set == 'balanced':
    confident_man = open(base_dir + '/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-man.txt').readlines()       
    confident_man = [int(c.strip()) for c in confident_man]
    confident_woman = open(base_dir + '/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-woman.txt').readlines()       
    confident_woman = [int(c.strip()) for c in confident_woman]
    confident_ims = set(confident_man + confident_woman) & set(bias_ids)
    image_set = confident_ims
elif target_set == 'confident':    
    confident_man = open(base_dir + '/data2/anja/xai/captions/val-confident-man-500new.txt').readlines()
    confident_man = [int(c.strip()) for c in confident_man]
    confident_woman = open(base_dir + '/data2/anja/xai/captions/val-confident-woman-500new.txt').readlines()
    confident_woman = [int(c.strip()) for c in confident_woman]
    confident_ims = set(confident_man + confident_woman) & set(bias_ids)
    image_set = confident_ims
else: 
    print "Invalid set specified"   
 
for caption_path in caption_paths:

    generation_coco = coco.loadRes(caption_path[1])
    coco_evaluator = COCOEvalCap(coco, generation_coco)
    coco_evaluator.params['image_id'] = list(set(image_set) & set(generation_coco.getImgIds())) 
    coco_evaluator.evaluate()

    predicted_caps = json.load(open(caption_path[1]))

    for cap in predicted_caps:
        words = nltk.word_tokenize(cap['caption'].lower())
        words = ['person' if word in gendered_words else word for word in words] 
        cap['caption'] = ' '.join(words)
        if len(set(words) & gendered_words) > 0: pdb.set_trace()
    
    """
    person_caps = 'tmp/person_caps.json'
    with open(person_caps, 'w') as outfile:
        json.dump(predicted_caps, outfile)
    generation_coco = coco.loadRes(person_caps)
    coco_evaluator = COCOEvalCap(coco, generation_coco)
    coco_evaluator.params['image_id'] = list(set(image_set) & set(generation_coco.getImgIds()))
    coco_evaluator.evaluate()
    """
    print "Model name: %s" %caption_path[0]
    pdb.set_trace() 


