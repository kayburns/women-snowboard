import json
import nltk
import pdb

#coco_dir = '/data/lisaanne/coco/annotations/captions_%s2014.json' 
coco_dir = coco_dir = '/data1/caption_bias/models/research/im2txt/im2txt/data/raw-data/annotations/captions_%s2014.json'

man_word_list = ['boy', 'brother', 'dad', 'husband', 'man', 'groom', 'male', 'guy']
woman_word_list = ['girl', 'sister', 'mom', 'wife', 'woman', 'bride', 'female', 'lady']

for split in ['val']:#['train', 'val']:
    data = json.load(open(coco_dir %split))

    id_to_annotations = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in id_to_annotations:
            id_to_annotations[image_id] = []
        id_to_annotations[image_id].append(annotation)

    confident_man = []
    confident_woman = []
    for image_id in id_to_annotations.keys():
        captions = [annotation['caption'] for annotation in id_to_annotations[image_id]]
        man_bool = 0
        woman_bool = 0
        for caption in captions:
            words = nltk.word_tokenize(caption)
            man_words = len(set(man_word_list) & set(words)) > 0
            woman_words = len(set(woman_word_list) & set(words)) > 0
            man_bool += man_words
            woman_bool += woman_words

        if (man_bool > 0) & (woman_bool > 0):
            man_bool = False
            woman_bool = False

        man_bool = man_bool >= 5
        woman_bool = woman_bool >= 5
        if not (man_bool & woman_bool):
            if man_bool:
                confident_man.append(image_id)
                a = 1
            if woman_bool:
                confident_woman.append(image_id)

    write_txt_file = '/data2/caption-bias/dataset-splits/%s-all-confident-man.txt' %split
    write_txt = open(write_txt_file, 'w')
    for image_id in confident_man:
        write_txt.writelines('%s\n' %image_id)
    write_txt_file = '/data2/caption-bias/dataset-splits/%s-all-confident-woman.txt' %split
    write_txt = open(write_txt_file, 'w')
    for image_id in confident_woman:
        write_txt.writelines('%s\n' %image_id)  
