import pdb
import json

#txt_file = '/home/lisaanne/lev/data1/caption_bias/models/research/im2txt/val_captions_fine_tune_2.txt'
txt_file = '/home/lisaanne/lev/data1/caption_bias/models/research/im2txt/val_captions.txt'

lines = open(txt_file).readlines()

ims = []
sentences = []

for i, line in enumerate(lines):
    if 'Captions for image' in line:
        ims.append(line)
        sentences.append(lines[i+1])

ims = [int(im.split('_')[-1].split('.jpg')[0]) for im in ims]
sentences = [sentence[5:].split(' .')[0] for sentence in sentences]

predicted_caps = []
for im, sentence in zip(ims, sentences):
    predicted_caps.append({'image_id': im, 'caption': sentence})
pdb.set_trace()
save_file = '/home/lisaanne/lev/data2/caption-bias/result_jsons/baseline.json'

pdb.set_trace()
with open(save_file, 'w') as outfile:
    json.dump(predicted_caps, outfile)
