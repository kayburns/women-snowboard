'''
Helper code to process men also like shopping annotations and some data analysis.
'''

import pickle
import csv
from collections import defaultdict

###########################################
############       Utils       ############
###########################################

def classified_as_woman(caption):
    """
    Returns true if caption classified agent
    as woman.

    Only returns true if woman present in
    word and not man.
    """
    return ('woman' in caption) and ('man' not in caption)

def classified_as_man(caption):
    """
    Returns true if caption classified agent
    as man.

    Only returns true if man present in
    word and not woman.
    """
    return ('man' in caption) and ('woman' not in caption)

def parse_filter_files(filename):
    """
    Parses a file with an image name
    on each line.

    Returns list of file names.
    """
    with open(filename) as f:
        img_names = f.readlines()
    return [img_name.strip('\n') for img_name in img_names]

def dict_to_csv(filename, dictionary, columns=None):
    """
    Writes each key value pair of a dictionary to a 
    row in CSV file named filename. Optional argument
    columns allows you to specify column names.
    """
    with open(filename,'wb+') as f:
        w = csv.writer(f)
        if columns != None:
            w.writerow(columns)
        w.writerows(dictionary.items())

def img_name_to_img_id(img_name):
    return int(img_name.split('.')[0].split('_')[2])
        
def caption_file_to_dict(file, keys='img_name'):
    """
    Converts a file with captions (as output my
    im2txt model) into a dictionary indexed by
    image name or image id (depending on the
    argument to keys).
    :input:
    filename -- file name of file with captions
    keys -- 'img_name' for dictionary indexed by 
        string of image name. 'img_id' for dictionary
        indexed by integer representing image id
    """
    caption_dict = {}
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if keys == 'img_id':
                    img_key = img_name_to_img_id(img_name)
                elif keys == 'img_name':
                    img_key = img_name
                else:
                    raise ValueError("key must be 'img_id' or 'img_name'")
                first_caption = f.readline().split()
                caption_dict[img_key] = first_caption
            line = f.readline()

    return caption_dict

###########################################
############     Analysis      ############
###########################################

def get_image_names_from_reducing_bias(data_path):
    """
    Returns a list of image names
    contained in the data set defined by
    data_path.
    """
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file)
    img_names = []
    for img in data:
        img_names.append(img['img'])
        # later, if need ids
        # img_ids.append(int(img['img'].split('.')[0].split('_')[2]))
    return img_names

def compute_man_female_per_object_322(samples):
    """
    :input: list of dictionaries, one for
        each image.
    :returns: counts of [male, female] occurences
        per object
    taken from
    https://github.com/uclanlp/reducingbias
    inference_debias.py
    """
    count = dict()
    for i in range(80):
        count[i] = [0,0]
    for sample in samples:
        sample = sample['annotation']
        if sample[0] == 1: # man
            objs = sample[2:162]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][0] += 1
        else: # woman
            objs = sample[162:]
            for j in range(80):
                if objs[2*j] == 1:
                    count[j][1] += 1
    return count

def create_dict_from_list(samples):
    """
    :input: list structured as follows
        [{'annotation': [1, 0, ...], 
          'img': 'COCO_ ...'}]  //note from lisa: gt from reducing bias paper
    :output: dictionary structured as follows
        {'COCO_ ...': [1, 0, ...]}
    """
    img2anno = {}
    for sample in samples:
        img = sample['img']
        img2anno[img] = sample['annotation']
    return img2anno

def create_dict_of_ids_from_list(samples):
    """
    :input: list structured as follows
        [{'annotation': [1, 0, ...], 
          'img': 'COCO_ ...'}]
    :output: dictionary structured as follows
        {'COCO_ ...': [1, 0, ...]}
    """
    img2anno = {}
    for sample in samples:
        img = int(sample['img'].split('.')[0].split('_')[2])
        img2anno[img] = sample['annotation']
    return img2anno

def male_female_error(file, img_2_anno_dict):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        img_2_anno_dict -- dictionary that maps
            images to annotations

    :return: tuple,
        (% women misclassified as men,
        % men misclassified as women).
    """
    img_name_list = list(img_2_anno_dict.keys())

    total_man, total_woman, mis_man, mis_woman = 0, 0, 0, 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_name_list:
                    first_caption = f.readline().split()
                    if img_2_anno_dict[img_name][0] == 1: # male
                        total_man += 1
                        if ('woman' in first_caption) and ('man' not in first_caption):
                            # print("man", first_caption)
                            mis_man += 1
                    else: # female 
                        total_woman += 1
                        if ('man' in first_caption) and ('woman' not in first_caption):
                            # print("woman", first_caption)
                            mis_woman += 1
            line = f.readline()

    return (mis_man/float(total_man), mis_woman/float(total_woman))

def male_female_count(file):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
    :return: tuple,
        (# of occurences of 'man',
        # of occurences of 'woman').
    """

    total_man, total_woman = 0, 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                first_caption = f.readline().split()
                if classified_as_woman(first_caption):
                    total_woman += 1
                if classified_as_man(first_caption): 
                    total_man += 1
                    print (first_caption)
            line = f.readline()

    return (total_man, total_woman)

def male_female_count_filter(file, filter_list):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        filter_list -- list of file names to accept
    :return: tuple,
        (# of occurences of 'man',
        # of occurences of 'woman').
    """

    total_man, total_woman = 0, 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in filter_list:
                    first_caption = f.readline().split()
                    if classified_as_woman(first_caption):
                        total_woman += 1
                    if classified_as_man(first_caption): 
                        total_man += 1
            line = f.readline()

    return (total_man, total_woman)

def compute_male_female_per_object_from_predictions(file, img_2_anno_dict, object_mapping):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        images_with_male_female -- list of images tested
            in reducing bias paper
        words_of_interest -- dictionary from id to
            object of interest
    :returns: counts of [male, female] occurences
        per object
        {"toilet": [108, 45],
         "teddy_bear": [89, 156],
         ...
        }
    """
    # set up dictionary with counts
    img_list = list(img_2_anno_dict.keys())
    correct = defaultdict(lambda: [0, 0])
    incorrect = defaultdict(lambda: [0, 0])
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_list:
                    first_caption = f.readline().split()
                    if img_2_anno_dict[img_name][0] == 1: # man
                        objs = img_2_anno_dict[img_name][2:162]
                        for j in range(80):
                            if objs[2*j] == 1:
                                if classified_as_man(first_caption):
                                    correct[object_mapping[j]][0] += 1
                                elif classified_as_woman(first_caption):
                                    incorrect[object_mapping[j]][0] += 1
                    else: # woman
                        objs = img_2_anno_dict[img_name][162:]
                        for j in range(80):
                            if objs[2*j] == 1:
                                if classified_as_woman(first_caption):
                                    correct[object_mapping[j]][1] += 1
                                elif classified_as_man(first_caption):
                                    incorrect[object_mapping[j]][1] += 1
            line = f.readline()
    return (correct, incorrect)

def compute_male_female_per_object_from_predictions_filter(file, img_2_anno_dict, object_mapping, filter_list):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        images_with_male_female -- list of images tested
            in reducing bias paper
        words_of_interest -- dictionary from id to
            object of interest
    :returns: counts of [male, female] occurences
        per object
        {"toilet": [108, 45],
         "teddy_bear": [89, 156],
         ...
        }
    """
    # set up dictionary with counts
    img_names = set(img_2_anno_dict.keys())
    img_name_set = set.intersection(img_names, set(filter_list))
    correct = defaultdict(lambda: [0, 0])
    incorrect = defaultdict(lambda: [0, 0])
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_name_set:
                    first_caption = f.readline().split()
                    if img_2_anno_dict[img_name][0] == 1: # man
                        objs = img_2_anno_dict[img_name][2:162]
                        for j in range(80):
                            if objs[2*j] == 1:
                                if classified_as_man(first_caption):
                                    correct[object_mapping[j]][0] += 1
                                elif classified_as_woman(first_caption):
                                    incorrect[object_mapping[j]][0] += 1
                    else: # woman
                        objs = img_2_anno_dict[img_name][162:]
                        for j in range(80):
                            if objs[2*j] == 1:
                                if classified_as_woman(first_caption):
                                    correct[object_mapping[j]][1] += 1
                                elif classified_as_man(first_caption):
                                    incorrect[object_mapping[j]][1] += 1
            line = f.readline()
    return (correct, incorrect)

def get_percent_captions_without_man_woman(file, img_2_anno_dict):
    """
    Computes the percent of captions that don't mention
    only man or woman in file. 
    :input:
    file -- name of file with captions
    img_2_anno_dict -- dictionary that maps
        images to annotations
    """
    img_list = list(img_2_anno_dict.keys())

    total, unclassified = 0, 0
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                img_name = line[3].strip(':')                              
                if img_name in img_list:                                   
                    first_caption = f.readline().split()                   
                    total += 1                                                       
                    if classified_as_man(first_caption) or classified_as_woman(first_caption):
                        unclassified += 1                   
            line = f.readline()
    return 1 - (unclassified / float(total))

def get_percent_captions_without_man_woman_filter(file, img_2_anno_dict, filter_list=[]):
    """
    Computes the percent of captions that don't mention
    only man or woman in file. 
    :input:
    file -- name of file with captions
    img_2_anno_dict -- dictionary that maps
        images to annotations
    """
    img_names = set(img_2_anno_dict.keys())
    img_name_set = set.intersection(img_names, set(filter_list))

    total, unclassified = 0, 0
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                img_name = line[3].strip(':')
                if img_name in img_name_set:
                    first_caption = f.readline().split()
                    total += 1
                    if classified_as_man(first_caption) or classified_as_woman(first_caption):
                        unclassified += 1
            line = f.readline()
    return 1 - (unclassified / float(total))

def precision(file, img_2_anno_dict):
    """
    Computes captions precision in classifying
    male/female.
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        img_2_anno_dict -- dictionary that maps
            images to annotations
    :return: tuple of (precision_for_men, precision_for_women)
        precision = |{images correctly id gender}| / |{number of images with gender}|
    """
    img_list = list(img_2_anno_dict.keys())

    total_woman = 0
    correct_woman = 0
    total_man = 0
    correct_man = 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_list:
                    first_caption = f.readline().split()

                    if classified_as_man(first_caption):
                        total_man += 1
                        if img_2_anno_dict[img_name][0] == 1: # male
                            correct_man += 1
                    if classified_as_woman(first_caption):
                        total_woman += 1
                        if img_2_anno_dict[img_name][0] == 0: # female
                            correct_woman += 1
            line = f.readline()

    return (correct_man, total_man, correct_woman, total_woman)
    # return (correct_man/float(total_man), correct_woman/float(total_woman))

def precision_filter(file, img_2_anno_dict, filter_list=[]):
    """
    Computes captions precision in classifying
    male/female.
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        img_2_anno_dict -- dictionary that maps
            images to annotations
        filter_list -- list of image names to accept
    :return: tuple of (precision_for_men, precision_for_women)
        precision = |{images correctly id gender}| / |{number of images labeled with gender}|
    """
    img_names = set(img_2_anno_dict.keys())
    img_name_set = set.intersection(img_names, set(filter_list))

    total_woman = 0
    correct_woman = 0
    total_man = 0
    correct_man = 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_name_set:
                    first_caption = f.readline().split()

                    if classified_as_man(first_caption):
                        total_man += 1
                        # print("man", first_caption)
                        if img_2_anno_dict[img_name][0] == 1: # male
                            correct_man += 1
                    if classified_as_woman(first_caption):
                        total_woman += 1
                        # print("woman", first_caption)
                        if img_2_anno_dict[img_name][0] == 0: # female
                            correct_woman += 1
            line = f.readline()

    return (correct_man, total_man, correct_woman, total_woman)
    # return (correct_man/float(total_man), correct_woman/float(total_woman))

def recall(file, img_2_anno_dict):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        img_2_anno_dict -- dictionary that maps
            images to annotations

    :return: tuple, (recall man, recall woman).
        recall = |{number of X classified correctly}| / |{total number of X}|
    """
    img_name_list = set(img_2_anno_dict.keys())

    total_man, total_woman, correct_man, correct_woman = 0, 0, 0, 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_name_list:
                    first_caption = f.readline().split()
                    if img_2_anno_dict[img_name][0] == 1: # male
                        total_man += 1
                        if classified_as_man(first_caption):
                            correct_man += 1
                        # else:
                            # print(img_name, "had a man but caption was: ", first_caption)
                    else: # female 
                        total_woman += 1
                        if classified_as_woman(first_caption):
                            correct_woman += 1
                        # else:
                            # print(img_name, " had a woman but caption was: ", first_caption)
            line = f.readline()
    return (correct_man, total_man, correct_woman, total_woman)
    # return (correct_man/float(total_man), correct_woman/float(total_woman))

def recall_filter(file, img_2_anno_dict, filter_list=[]):
    """
    :input:
        file -- file name with caption data (format
            is default from tensorflow im2txt model)
        img_2_anno_dict -- dictionary that maps
            images to annotations
        filter_file -- file with list of images to consider

    :return: tuple, (recall man, recall woman).
    """
    img_names = set(img_2_anno_dict.keys())
    img_name_set = set.intersection(img_names, set(filter_list))

    total_man, total_woman, correct_man, correct_woman = 0, 0, 0, 0

    with open(file) as f:
        line = f.readline()
        while line:
            line = line.split()
            if line[0] == 'Captions':
                # only use the first caption
                img_name = line[3].strip(':')
                if img_name in img_name_set:
                    first_caption = f.readline().split()
                    if img_2_anno_dict[img_name][0] == 1: # male
                        total_man += 1
                        if classified_as_man(first_caption):
                            correct_man += 1
                        else:
                            print(img_name, " had a man but caption was: ", first_caption)
                    else: # female 
                        total_woman += 1
                        if classified_as_woman(first_caption):
                            correct_woman += 1
                        else:
                            print(img_name, " had a woman but caption was: ", first_caption)
            line = f.readline()

    return (correct_man, total_man, correct_woman, total_woman)
    # return (correct_man/float(total_man), correct_woman/float(total_woman))

# test_caption_file = 'test_captions.txt'
# train_caption_file = 'train_captions.txt'
# val_caption_file = 'val_captions.txt'
target_train = '../im2txt/data/raw-data/reducingbias/data/COCO/train.data'
target_val = '../im2txt/data/raw-data/reducingbias/data/COCO/dev.data'
target_test = '../im2txt/data/raw-data/reducingbias/data/COCO/test.data'
val_filter_file = '/data1/caption_bias/models/research/im2txt/im2txt/data/val_dataset.txt'
test_filter_file = '/data1/caption_bias/models/research/im2txt/im2txt/data/test_dataset.txt'
val_caption_file = 'val_captions_fine_tune_2.txt'
train_caption_file = 'train_captions_fine_tune_2.txt'

id2object = {0: 'toilet', 1: 'teddy_bear', 2: 'sports_ball', 3: 'bicycle', 4: 'kite', 5: 'skis', 6: 'tennis_racket', 7: 'donut', 8: 'snowboard', 9: 'sandwich', 10: 'motorcycle', 11: 'oven', 12: 'keyboard', 13: 'scissors', 14: 'chair', 15: 'couch', 16: 'mouse', 17: 'clock', 18: 'boat', 19: 'apple', 20: 'sheep', 21: 'horse', 22: 'giraffe', 23: 'person', 24: 'tv', 25: 'stop_sign', 26: 'toaster', 27: 'bowl', 28: 'microwave', 29: 'bench', 30: 'fire_hydrant', 31: 'book', 32: 'elephant', 33: 'orange', 34: 'tie', 35: 'banana', 36: 'knife', 37: 'pizza', 38: 'fork', 39: 'hair_drier', 40: 'frisbee', 41: 'umbrella', 42: 'bottle', 43: 'bus', 44: 'zebra', 45: 'bear', 46: 'vase', 47: 'toothbrush', 48: 'spoon', 49: 'train', 50: 'airplane', 51: 'potted_plant', 52: 'handbag', 53: 'cell_phone', 54: 'traffic_light', 55: 'bird', 56: 'broccoli', 57: 'refrigerator', 58: 'laptop', 59: 'remote', 60: 'surfboard', 61: 'cow', 62: 'dining_table', 63: 'hot_dog', 64: 'car', 65: 'cup', 66: 'skateboard', 67: 'dog', 68: 'bed', 69: 'cat', 70: 'baseball_glove', 71: 'carrot', 72: 'truck', 73: 'parking_meter', 74: 'suitcase', 75: 'cake', 76: 'wine_glass', 77: 'baseball_bat', 78: 'backpack', 79: 'sink'}


"""Preprocessing."""
#images_with_male_female = get_image_names_from_reducing_bias(target_train)
#img_2_anno_dict = create_dict_from_list(pickle.load(open(target_train)))
#img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_test))))
#img_2_anno_dict.update(create_dict_from_list(pickle.load(open(target_val))))

"""Accuracy scores."""
# accuracy_scores_test = male_female_error(test_caption_file, img_2_anno_dict)
# accuracy_scores_train = male_female_error(train_caption_file, img_2_anno_dict)
# accuracy_scores_val = male_female_error(val_caption_file, img_2_anno_dict)

"""Gender Counts."""
# counts_test = male_female_count_filter(val_caption_file, parse_filter_files(test_filter_file))
# counts_train = male_female_count(train_caption_file)
# counts_val = male_female_count_filter(val_caption_file, parse_filter_files(val_filter_file))

# print ("test, train, val", counts_test, counts_train, counts_val)

"""Scores per object."""
# target_occurence_counts = compute_man_female_per_object_322(pickle.load(open(comparison_file)))
# counts_per_object_train = compute_male_female_per_object_from_predictions(train_caption_file, images_with_male_female, id2object)
# counts_per_object = compute_male_female_per_object_from_predictions()
# counts_per_object_train = compute_male_female_per_object_from_predictions(train_caption_file, img_2_anno_dict, id2object)
# counts_per_object_val = compute_male_female_per_object_from_predictions_filter(val_caption_file, img_2_anno_dict, id2object, parse_filter_files(val_filter_file))
# counts_per_object_test = compute_male_female_per_object_from_predictions_filter(val_caption_file, img_2_anno_dict, id2object, parse_filter_files(test_filter_file))

"""Percent without man woman."""
# no_mw_train = get_percent_captions_without_man_woman(train_caption_file, img_2_anno_dict)
# no_mw_val = get_percent_captions_without_man_woman_filter(val_caption_file, img_2_anno_dict, parse_filter_files(val_filter_file))
# no_mw_test = get_percent_captions_without_man_woman_filter(val_caption_file, img_2_anno_dict, parse_filter_files(test_filter_file))

# print ("percent captions without man/woman in train", no_mw_train)
# print ("percent captions without man/woman in val", no_mw_val)
# print ("percent captions without man/woman in test", no_mw_test)

"""Precision and recall."""
#precision_test = precision_filter(val_caption_file, img_2_anno_dict, parse_filter_files(test_filter_file))
#precision_val = precision_filter(val_caption_file, img_2_anno_dict, parse_filter_files(val_filter_file))
#precision_train = precision(train_caption_file, img_2_anno_dict)

#recall_test = recall_filter(val_caption_file, img_2_anno_dict, parse_filter_files(test_filter_file))
#recall_val = recall_filter(val_caption_file, img_2_anno_dict, parse_filter_files(val_filter_file))
#recall_train = recall(train_caption_file, img_2_anno_dict)


#print("precision_test", precision_test)
#print("precision_val", precision_val)
#print("precision_train", precision_train)

#print("recall_test", recall_test)
#print("recall_val", recall_val)
#print("recall_train", recall_train)

# dict_to_csv("analysis/training_fine_tune_correct_counts_per_obj.csv", counts_per_object_train[0], columns=["Object", "[Male, Female]"])
# dict_to_csv("analysis/training_fine_tune_incorrect_counts_per_obj.csv", counts_per_object_train[1], columns=["Object", "[Male as Female, Female as Male]"])
# dict_to_csv("analysis/val_fine_tune_correct_counts_per_obj.csv", counts_per_object_val[0], columns=["Object", "[Male, Female]"])
# dict_to_csv("analysis/val_fine_tune_incorrect_counts_per_obj.csv", counts_per_object_val[1], columns=["Object", "[Male as Female, Female as Male]"])
# dict_to_csv("analysis/test_fine_tune_correct_counts_per_obj.csv", counts_per_object_test[0], columns=["Object", "[Male, Female]"])
# dict_to_csv("analysis/test_fine_tune_incorrect_counts_per_obj.csv", counts_per_object_test[1], columns=["Object", "[Male as Female, Female as Male]"])
