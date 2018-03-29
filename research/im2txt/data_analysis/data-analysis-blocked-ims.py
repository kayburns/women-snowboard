import json
from data_analysis_common import *
import pdb

caption_paths = []
base_dir = '/home/lisaanne/lev/'
normal_training = ('normal training', base_dir + '/data2/caption-bias/result_jsons/normal-training-blocked-ims.json')
caption_paths.append(normal_training)
acl_nosum_ce = ('ACL (10) - no sum (CE)', base_dir + '/data2/caption-bias/result_jsons/acl-ce-blocked-ims.json')
caption_paths.append(acl_nosum_ce)
baseline_ft_inception = ('baseline ft inception', base_dir + '/data2/caption-bias/result_jsons/train_fine_tune_incep_bias_split-blocked-ims.json')
caption_paths.append(baseline_ft_inception)
uw = ('uw 10x', base_dir + '/data2/caption-bias/result_jsons/LW10_ft-inception-fresh-blocked-ims.json')
caption_paths.append(uw)
balanced = ('balanced', base_dir + '/data2/caption-bias/result_jsons/balanced-blocked-ims.json')
caption_paths.append(balanced)
acl = ('acl', base_dir + '/data2/caption-bias/result_jsons/blocked_loss_w10_ft_incep_no_sum-blocked-ims.json')
caption_paths.append(acl)
acl_conq = ('ACL Con-Q', base_dir + '/data2/caption-bias/result_jsons/quotient-confusion-blocked-ims.json')
caption_paths.append(acl_conq)
acl_conq_uw = ('ACL Con-Q UW', base_dir + '/data2/caption-bias/result_jsons/quotient_confusion_LW_10_gender-blocked-ims.json')
caption_paths.append(acl_conq_uw)
acl_uw = ('ACL UW', base_dir + '/data2/caption-bias/result_jsons/quotient-blocked-ims.json')
caption_paths.append(acl_uw)
acl_uw_ce = ('ACL UW CE', base_dir + '/data2/caption-bias/result_jsons/quotient-only-blocked-ims.json')
caption_paths.append(acl_uw_ce)
quotient = ('quotient', base_dir + '/data2/caption-bias/result_jsons/quotient-only-blocked-ims.json')
caption_paths.append(quotient)
#quotient_uw = ('quotient UW', base_dir +'/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_UW_10_500k_caps.json')
#caption_paths.append(quotient_uw)


def blocked_ratios(predictions):
    man_count = 0.
    woman_count = 0.
    other_count = 0.

    for prediction in predictions:
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
            other_count+= 1

    print "Persents:"
    print "Man: %d/%d (%f)." %(man_count, len(predictions), man_count/len(predictions))
    print "Woman: %d/%d (%f)." %(woman_count, len(predictions), woman_count/len(predictions))
    print "Other: %d/%d (%f)." %(other_count, len(predictions), other_count/len(predictions))
    print "%f\t%f\t%f" % (man_count/len(predictions), woman_count/len(predictions), other_count/len(predictions))


for caption_path in caption_paths:
    print "Model name: %s" %caption_path[0]
    predictions = json.load(open(caption_path[1]))
    blocked_ratios(predictions)
    pdb.set_trace()
