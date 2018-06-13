'''
Gets sentence metric over MSCOCO test set and biased set.
'''
from IPython import embed
import sys
import json
from data_analysis_common import *
import nltk
import pdb
from bias_detection import *
from pycocotools.coco import COCO
sys.path.append('/data2/kaylee/caption_bias/models/research/im2txt/coco-caption')
from pycocoevalcap.eval import COCOEvalCap
import os
from data_analysis_base import AnalysisBaseClass, caption_paths

#base_dir = '/home/lisaanne/lev/'
base_dir = ''
#create person set gt

gt_path = base_dir + '/data1/caption_bias/models/research/im2txt/coco-caption/annotations/captions_val2014.json'
coco = COCO(gt_path)

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

for caption_path in caption_paths:
    print "Model name: %s" %caption_path[0]

    print "##############Sentence metrics over MSCOCO Set:#######################"
    predicted_captions = json.load(open(caption_path[1]))
    
    generation_coco = coco.loadRes(caption_path[1])
    coco_evaluator = COCOEvalCap(coco, generation_coco)
    # coco_evaluator.evaluate()

    print "##############Sentence metrics over Biased Set:#######################"
    
    
    #Sentence metrics over biased captions
    bias_captions = []
    for caption in predicted_captions:
        if caption['image_id'] in bias_ids:
            bias_captions.append(caption)
            
    bias_save_file = caption_path[1].replace('.json', '_bias.json')
    if not os.path.exists(bias_save_file):
    
        with open(bias_save_file, 'w') as outfile:
            json.dump(bias_captions, outfile)
    
    generation_coco = coco.loadRes(bias_save_file)
    coco_evaluator = COCOEvalCap(coco, generation_coco)
    
    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_evaluator.params['image_id'] = generation_coco.getImgIds()
    embed()
    coco_evaluator.evaluate()

    print "Model name: %s" %caption_path[0]
    print "##############Sentence metrics over Shopping Test:#######################"
    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    shopping_test_split = AnalysisBaseClass.get_shopping_split()
    coco_evaluator.params['image_id'] = AnalysisBaseClass.convert_filenames_to_ids(shopping_test_split)
    coco_evaluator.evaluate()
    

