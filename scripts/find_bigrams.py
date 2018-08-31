import json
import nltk
from nltk.stem import *
import sys
from nltk.corpus import wordnet as wn
import pickle as pkl
import pdb

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
verbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}

delete_list = {'he', 'a'}

coco_dir = '/data/lisaanne/coco/annotations/captions_%s2014.json' 

lemma = nltk.wordnet.WordNetLemmatizer()

man_word_list = ['boy', 'brother', 'dad', 'husband', 'man', 'groom', 'male', 'guy', 'father']
woman_word_list = ['girl', 'sister', 'mom', 'wife', 'woman', 'bride', 'female', 'lady', 'mother']

class myDict(object):  #I feel like there has to be a python function that does this

    def __init__(self):
        self.dict = {}
        self.image_id_dict = {}

    def add(self, item, image_id):
        if item not in self.dict:
            self.dict[item] = 0.
            self.image_id_dict[item] = []
        if not image_id in set(self.image_id_dict[item]):
            self.dict[item] += 1.
            self.image_id_dict[item].append(image_id)

    def __call__(self, item):
        return self.dict[item]

    def normalize(self, norm_factor):
        for key in self.dict.keys():
            self.dict[key] = self.dict[key]/float(norm_factor)

man_bigram_words = myDict()
woman_bigram_words = myDict()
word_counts = myDict()

num_man_sentences = 0.
num_woman_sentences = 0.

bias_set = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/bias-set.txt').readlines()
bias_set = set([int(b.strip()) for b in bias_set])

for split in ['val']:
    data = json.load(open(coco_dir %split))

    for i, annotation in enumerate(data['annotations']):
        sys.stdout.write('\r%d/%d' %(i, len(data['annotations'])))
        if annotation['image_id'] in bias_set:
            caption = annotation['caption']
            words = nltk.word_tokenize(caption.lower())
            for word in words: word_counts.add(word, annotation['image_id'])
            if len((set(man_word_list) | set(woman_word_list)) & set(words)) > 0: #this will include sentences with both man and woman, but okay bc should add to ratio the same?
                lemmas = [lemma.lemmatize(word) for word in words]
                man_sentence = len(set(man_word_list) & set(words)) > 0
                woman_sentence = len(set(woman_word_list) & set(words)) > 0
                if man_sentence and (not woman_sentence):  
                    for word in words: man_bigram_words.add(word, annotation['image_id'])
                if woman_sentence and (not man_sentence):  
                    for word in words: woman_bigram_words.add(word, annotation['image_id'])
                num_man_sentences += 1
                num_woman_sentences += 1
#normalize counts by number of sentences
man_bigram_words.normalize(num_man_sentences)
woman_bigram_words.normalize(num_woman_sentences)
all_words = set(man_bigram_words.dict.keys()) & set(woman_bigram_words.dict.keys())
all_words = all_words - set(man_word_list)
all_words = all_words - set(woman_word_list)

threshold = 250 
words_gt_threshold = []
for key in word_counts.dict.keys():
    if word_counts(key) > threshold:
        words_gt_threshold.append(key)

all_words = all_words & set(words_gt_threshold)
all_words = all_words & (nouns | verbs)
all_words -= delete_list

all_word_ratio = {} 
for word in all_words:
    all_word_ratio[word] = man_bigram_words(word)/(woman_bigram_words(word) + man_bigram_words(word))

#look at 50 most common words?

sorted_keys = sorted(all_word_ratio, key=all_word_ratio.get)

save_dict = {}

#``hand selected'' words to consider

bias_words = ['umbrella', 'kitchen', 'cell', 'table', 'food', 'skateboard', 'baseball', 'tie', 'motorcycle', 'snowboard']


for word in bias_words:
    common_words = set(woman_bigram_words.image_id_dict[word]) & set(man_bigram_words.image_id_dict[word])
    woman_bigram_words.image_id_dict[word] = list(set(woman_bigram_words.image_id_dict[word]) - common_words)
    man_bigram_words.image_id_dict[word] = list(set(man_bigram_words.image_id_dict[word]) - common_words)

import pdb; pdb.set_trace()
for word in bias_words: print len(set(man_bigram_words.image_id_dict[word]) & set(woman_bigram_words.image_id_dict[word]))

print 'man'
for word in bias_words: print '%s %d' %(word, len(man_bigram_words.image_id_dict[word]))
print 'woman'
for word in bias_words: print '%s %d' %(word, len(woman_bigram_words.image_id_dict[word]))


for word in bias_words:
    write_txt = open('/home/lisaanne/lev/data2/caption-bias/dataset-splits/intersection_%s_person.txt' %word, 'w')
    assert len(set(man_bigram_words.image_id_dict[word]) & set(woman_bigram_words.image_id_dict[word])) == 0
    for image_id in man_bigram_words.image_id_dict[word] + woman_bigram_words.image_id_dict[word]:
        write_txt.writelines('%s\n' %image_id)
    write_txt.close()
pdb.set_trace()

#for key in sorted_keys:
#    save_dict[key] = all_word_ratio[key]
#pkl.dump(save_dict, open('/home/lisaanne/lev/data2/caption-bias/data/application_val.p', 'wb'))
