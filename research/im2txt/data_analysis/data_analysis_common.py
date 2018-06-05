'''
Common code shared across data analysis scripts.
'''

import json
import nltk

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
    if m & check_m:
        return True
    elif f & check_f:
        return True
    else:
        return False

def format_gt_captions(gt_file):
    gt = json.load(open(gt_file))
    gt_caps = []
    for annotation in gt['annotations']:
        gt_caps.append({'image_id': annotation['image_id'], 'caption': annotation['caption']})
    return gt_caps

def accuracy(predicted, man_word_list=['man'], woman_word_list=['woman'], img_2_anno_dict_simple=None):
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
        #if image_id not in confident_subset_ids:
        #   continue
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
    print "%f	%f	%f" % (m_fp/f_total, f_tp/f_total, f_other/f_total)
    
    print "Of male images:"
    print "Man predicted %f percent." %(m_tp/m_total)
    print "Woman predicted %f percent." %(f_fp/m_total)
    print "Other predicted %f percent." %(m_other/m_total)
    print "%f	%f	%f"% (m_tp/m_total, f_fp/m_total, m_other/m_total)

    print "Of total:"
    print "Correct %f percent." %((m_tp+f_tp)/(m_total+f_total))
    print "Incorect %f percent." %((m_fp+f_fp)/(m_total+f_total))
    print "Other predicted %f percent." %((f_other+m_other)/(m_total+f_total))
    print "%f	%f	%f"% ((m_tp+f_tp)/(m_total+f_total), (m_fp+f_fp)/(m_total+f_total), (m_other+f_other)/(f_total+m_total))

    print "ratio", float(f_tp + f_fp)/(m_tp + m_fp)
    pred_images = {}
    pred_images['male_pred_male'] = male_pred_male
    pred_images['female_pred_female'] = female_pred_female
    pred_images['female_pred_male'] = female_pred_male
    pred_images['male_pred_female'] = male_pred_female
    pred_images['male_pred_other'] = male_pred_other
    pred_images['female_pred_other'] = female_pred_other
    
    return pred_images

#caption paths
#all models
caption_paths = []
base_dir = ''
#base_dir = '/home/lisaanne/lev/'
#normal_training = ('normal training', base_dir + '/data1/caption_bias/models/research/im2txt/val_cap.json')
#caption_paths.append(normal_training)
# acl_nosum_ce = ('ACL (10) - no sum (CE)', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/train_blocked_ce.json')
# caption_paths.append(acl_nosum_ce)
# full_gender_set = ('full gender set', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/equalizer_all_gender_words.json')
# caption_paths.append(full_gender_set)
#baseline_ft_inception = ('baseline ft inception', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/ft_incep_captions_500k_bias_split.json')
#caption_paths.append(baseline_ft_inception)
#uw = ('uw 10x', base_dir + '/data2/caption-bias/result_jsons/LW10_ft-inception-fresh.json')
#caption_paths.append(uw)
#balanced = ('balanced', base_dir + '/data2/caption-bias/result_jsons/balance_man_woman_ft_inception.json')
#caption_paths.append(balanced)
#acl = ('acl', base_dir + '/data2/caption-bias/result_jsons/blocked_loss_w10_ft_incep_no_sum.json')
#caption_paths.append(acl)
#acl_conq = ('ACL Con-Q', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_loss_500k_iters.json')
#caption_paths.append(acl_conq)
#acl_conq_uw = ('ACL Con-Q UW', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/confusiont_quotient_UW.json')
#caption_paths.append(acl_conq_uw)
#acl_uw = ('ACL UW', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/train_blocked_U_10.json')
#caption_paths.append(acl_uw)
#acl_uw_ce = ('ACL UW CE', base_dir + '/data2/caption-bias/result_jsons/blocked_ce_LW10_ft-inception-fresh-iter1.500k.json')
#caption_paths.append(acl_uw_ce)
#quotient = ('quotient', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_no_blocked_caps.json')
#caption_paths.append(quotient)
#quotient_uw = ('quotient UW', base_dir +'/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_UW_10_500k_caps.json')
#caption_paths.append(quotient_uw)
#pytorch_model = ('pytorch_model', '/home/lisaanne/projects/sentence-generation/results/output.45950.ft-all-set.loss-acl10.ce-blocked.json')
#caption_paths.append(pytorch_model)
# uw_man5_woman15 = ('uw_man5_woman15', base_dir + '/data2/caption-bias/result_jsons/uw-man5-woman15_ft-inception-fresh.json')
# caption_paths.append(uw_man5_woman15)

#caption_paths = []
#base_dir = '/home/lisaanne/lev/'
#baseline_ft_inception = ('baseline ft inception', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/ft_incep_captions_500k_bias_split.json')
#caption_paths.append(baseline_ft_inception)
#uw = ('uw 10x', base_dir + '/data2/caption-bias/result_jsons/LW10_ft-inception-fresh.json')
#caption_paths.append(uw)
#balanced = ('balanced', base_dir + '/data2/caption-bias/result_jsons/balance_man_woman_ft_inception.json')
#caption_paths.append(balanced)
#acl = ('acl', base_dir + '/data2/caption-bias/result_jsons/blocked_loss_w10_ft_incep_no_sum.json')
#caption_paths.append(acl)
# acl_conq = ('ACL Con-Q', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_loss_500k_iters.json')
# caption_paths.append(acl_conq)

#quotient = ('quotient', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_no_blocked_caps.json')
#caption_paths.append(quotient)

# rebuttal captions
equalizer = ('equalizer', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/rebuttal_captions/equalizer_retest.json')
caption_paths.append(equalizer)

all_gender_words = ('equalizer trained with larger set of gender words', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/rebuttal_captions/equalizer_all_gender_words.json')
caption_paths.append(all_gender_words)

pairs = ('equalizer loss with coco images without people', base_dir+'/data2/kaylee/caption_bias/models/research/im2txt/rebuttal_captions/selective_pairs.json')
caption_paths.append(pairs)
