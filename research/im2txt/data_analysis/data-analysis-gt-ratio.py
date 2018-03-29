from data_analysis_common import *
from bias_detection import *

caps = format_gt_captions('/data/lisaanne/coco/annotations/captions_val2014.json')

img_2_anno_dict = create_dict_from_list(pickle.load(open(target_train)))
img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_test))))
img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_val))))

caption_ids = [cap['image_id'] for cap in caps]

img_2_anno_dict_simple = {}
for key, value in img_2_anno_dict.iteritems():
    id = int(key.split('_')[-1].split('.jpg')[0])
    if id in caption_ids:
        img_2_anno_dict_simple[id] = {}
        img_2_anno_dict_simple[id]['male'] = value[0]
        img_2_anno_dict_simple[id]['female'] = int(not value[0])  
        assert int(not value[0]) == value[1]
bias_ids = img_2_anno_dict_simple.keys()


predicted = [p for p in caps if p['image_id'] in img_2_anno_dict_simple]
import pdb; pdb.set_trace()
pred_images = accuracy(predicted, man_word_list_synonyms, woman_word_list_synonyms, img_2_anno_dict_simple)

confident_subset_ids = []
#man_confident_ids = '/home/lisaanne/lev/data2/anja/xai/captions/val-confident-man-500.txt'
#woman_confident_ids = '/home/lisaanne/lev/data2/anja/xai/captions/val-confident-woman-500.txt'

print("ONLY EVALUATING ON CONFIDENT SUBSET")
confident_man = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-man.txt').readlines()
confident_man = [int(c.strip()) for c in confident_man]
confident_woman = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/val-confident-woman.txt').readlines()
confident_woman = [int(c.strip()) for c in confident_woman]
confident_ims = set(confident_man + confident_woman)

predicted_confident = [p for p in predicted if p['image_id'] in confident_ims]

pred_images = accuracy(predicted_confident, man_word_list_synonyms, woman_word_list_synonyms, img_2_anno_dict_simple)


