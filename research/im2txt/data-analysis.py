import sys
sys.path.append('/data1/caption_bias/models/research/im2txt/coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import numpy as np
from bias_detection import *
import json
import os
import nltk

#json file for annotations for entire set

#save_file = '/home/lisaanne/lev/data2/caption-bias/result_jsons/baseline.json'
#save_file = '/home/lisaanne/lev/data2/kaylee/caption_bias/models/research/im2txt/captions/ft_incep_captions_225k.json'
#save_file = '/home/lisaanne/lev/data2/kaylee/caption_bias/models/research/im2txt/captions/captions_blocked_10_fine_tune_incep_225k_iters.json'
#save_file = '/home/lisaanne/lev/data2/caption-bias/result_jsons/confusionI_subtract_ce_blockLoss_fresh.json'
#save_file = '/home/lisaanne/lev/data2/caption-bias/result_jsons/blocked_loss_ce_loss_confusion_margin_2.json'
save_file = '/home/lisaanne/lev/data2/caption-bias/result_jsons/balance_man_woman_ft_inception.json'
predicted_captions = json.load(open(save_file))

coco = COCO('coco/annotations/captions_val2014.json')
generation_coco = coco.loadRes(save_file)
coco_evaluator = COCOEvalCap(coco, generation_coco)
print "Evaluation over the entire MSCOCO set:"
coco_evaluator.evaluate()

print "Evaluation over the biased set:"

#First create json with unbiased data.  

#From Kaylee's code

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

bias_captions = []
for caption in predicted_captions:
    if caption['image_id'] in bias_ids:
        bias_captions.append(caption)
        
bias_save_file = save_file.replace('.json', '_bias.json')
if not os.path.exists(bias_save_file):

    with open(bias_save_file, 'w') as outfile:
        json.dump(bias_captions, outfile)

generation_coco = coco.loadRes(bias_save_file)
coco_evaluator = COCOEvalCap(coco, generation_coco)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
coco_evaluator.params['image_id'] = generation_coco.getImgIds()

print "Evaluation over the biased MSCOCO set:"
coco_evaluator.evaluate()

#Get F1 scores for man/woman + some eval preprocessing

man_word_list_synonyms = ['boy', 'brother', 'dad', 'husband', 'man', 'groom', 'male', 'guy', 'men']
woman_word_list_synonyms = ['girl', 'sister', 'mom', 'wife', 'woman', 'bride', 'female', 'lady', 'women']

def is_gendered(words, gender_type='woman', man_word_list=['man'], woman_word_list=['woman']):
    m = False
    f = False
    check_m = (gender_type == 'man')
    check_f = (gender_type == 'woman')
    if len(set(words) & set(man_word_list)) > 0:
        m = True
    if len(set(words) & set(woman_word_list)) > 0:
        f = True
    if m & f:
        return False
    elif m & check_m:
        return True
    elif f & check_f:
        return True
    else:
        return False


def accuracy(predicted, man_word_list=['man'], woman_word_list=['woman']):
    f_tp = 0.
    f_fp = 0.
    f_tn = 0.
    f_other = 0.
    f_total = 0.
    
    
    m_tp = 0.
    m_fp = 0.
    m_tn = 0.
    m_other = 0.
    m_total = 0.
    
    male_pred_female = []
    female_pred_male = []
    male_pred_male = []
    female_pred_female = []
    male_pred_other = []
    female_pred_other = []
    
    for prediction in predicted:
        image_id = prediction['image_id']
        male = img_2_anno_dict_simple[image_id]['male']
        female = img_2_anno_dict_simple[image_id]['female']
        #pred_male = classified_as_man(prediction['caption'].split(' '))
        #pred_female = classified_as_woman(prediction['caption'].split(' '))
        sentence_words = nltk.word_tokenize(prediction['caption'].lower())
        pred_male = is_gendered(sentence_words, 'man', man_word_list, woman_word_list)
        pred_female = is_gendered(sentence_words, 'woman', man_word_list, woman_word_list)

        if (female & pred_female):
            f_tp += 1
            female_pred_female.append(prediction)
        if (male & pred_male):
            m_tp += 1
            male_pred_male.append(prediction)
        if (male & pred_female):
            f_fp += 1
            male_pred_female.append(prediction)
        if (female & pred_male):
            m_fp += 1
            female_pred_male.append(prediction)
        if ((not female) & (not pred_female)):
            f_tn += 1
        if ((not male) & (not pred_male)):
            m_tn += 1
        pred_other = (not pred_male) & (not pred_female)
        if (female & pred_other):
            f_other += 1
            female_pred_other.append(prediction)
        if (male & pred_other):
            m_other += 1
            male_pred_other.append(prediction)
        if female:
            f_total += 1
        if male:
            m_total += 1
    
    print "Of female images:"
    print "Man predicted %f percent." %(m_fp/f_total)
    print "Woman predicted %f percent." %(f_tp/f_total)
    print "Other predicted %f percent." %(f_other/f_total)
    
    print "Of male images:"
    print "Man predicted %f percent." %(m_tp/m_total)
    print "Woman predicted %f percent." %(f_fp/m_total)
    print "Other predicted %f percent." %(m_other/m_total)

    pred_images = {}
    pred_images['male_pred_male'] = male_pred_male
    pred_images['female_pred_female'] = female_pred_female
    pred_images['female_pred_male'] = female_pred_male
    pred_images['male_pred_female'] = male_pred_female
    pred_images['male_pred_other'] = male_pred_other
    pred_images['female_pred_other'] = female_pred_other
    
    return pred_images

predicted_all = json.load(open(save_file))
predicted = [p for p in predicted_all if p['image_id'] in img_2_anno_dict_simple]

#pred_images = accuracy(predicted, man_word_list_synonyms, woman_word_list_synonyms)
pred_images = accuracy(predicted)

#over confident set

confident_man = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-man.txt').readlines()
confident_man = [int(c.strip()) for c in confident_man]
confident_woman = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-woman.txt').readlines()
confident_woman = [int(c.strip()) for c in confident_woman]
confident_ims = set(confident_man + confident_woman)
predicted_confident = [p for p in predicted if p['image_id'] in confident_ims]
print "\nOver confident set"
tmp = accuracy(predicted_confident)

#over NOT confident set
print "\nOver not confident set"
predicted_not_confident = [p for p in predicted if p['image_id'] not in confident_ims]
tmp = accuracy(predicted_not_confident)


#Analysis of "other" sentences
print "\nAnalysis of 'other' sentences"
other_sentences = pred_images['male_pred_other'] + pred_images['female_pred_other']

#gender predictions of "other" sentences
tmp = accuracy(other_sentences, man_word_list_synonyms, woman_word_list_synonyms)

other_sentences = tmp['male_pred_other'] + tmp['female_pred_other']

mention_both = 0
gender_neutral = 0
others = 0

for sentence in other_sentences:
    words = nltk.word_tokenize(sentence['caption'].lower())
    if len(set(words) & set(man_word_list_synonyms+woman_word_list_synonyms)) > 0:
        mention_both += 1
    elif len(set(words) & set(['person', 'people', 'player', 'child'])) > 0:
        gender_neutral += 1
    else:
        others += 1

print "Mention both men and women: %d/%d" %(mention_both, len(other_sentences))
print "Mention gender neutral: %d/%d" %(gender_neutral, len(other_sentences))
print "Don't mention common person word: %d/%d" %(others, len(other_sentences))
