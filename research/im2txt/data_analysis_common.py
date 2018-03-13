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

def format_gt_captions(gt_file):
    gt = json.load(open(gt_file))
    gt_caps = []
    for annotation in gt['annotations']:
        gt_caps.append({'image_id': annotation['image_id'], 'caption': annotation['caption']})
    return gt_caps

#caption paths

caption_paths = []
base_dir = '/home/lisaanne/lev/'
normal_training = ('normal training', base_dir + '/data1/caption_bias/models/research/im2txt/val_cap.json')
caption_paths.append(normal_training)
acl_nosum_ce = ('ACL (10) - no sum (CE)', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/train_blocked_ce.json')
caption_paths.append(acl_nosum_ce)
baseline_ft_inception = ('baseline ft inception', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/ft_incep_captions_500k_bias_split.json')
caption_paths.append(baseline_ft_inception)
uw = ('uw 10x', base_dir + '/data2/caption-bias/result_jsons/LW10_ft-inception-fresh.json')
caption_paths.append(uw)
balanced = ('balanced', base_dir + '/data2/caption-bias/result_jsons/balance_man_woman_ft_inception.json')
caption_paths.append(balanced)
acl = ('acl', base_dir + '/data2/caption-bias/result_jsons/blocked_loss_w10_ft_incep_no_sum.json')
caption_paths.append(acl)
acl_conq = ('ACL Con-Q', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_loss_500k_iters.json')
caption_paths.append(acl_conq)
acl_conq_uw = ('ACL Con-Q UW', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/confusiont_quotient_UW.json')
caption_paths.append(acl_conq_uw)
acl_uw = ('ACL UW', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/train_blocked_U_10.json')
caption_paths.append(acl_uw)
acl_uw_ce = ('ACL UW CE', base_dir + '/data2/caption-bias/result_jsons/blocked_ce_LW10_ft-inception-fresh-iter1.500k.json')
caption_paths.append(acl_uw_ce)
quotient = ('quotient', base_dir + '/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_no_blocked_caps.json')
caption_paths.append(quotient)
quotient_uw = ('quotient UW', base_dir +'/data2/kaylee/caption_bias/models/research/im2txt/captions/quotient_UW_10_500k_caps.json')
caption_paths.append(quotient_uw)







