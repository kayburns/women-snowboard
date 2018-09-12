"""Common code shared across data analysis scripts."""
import os, sys
import json, pickle, collections
import nltk
from pattern.en import singularize
import numpy as np

class AnalysisBaseClass:

    """
    Each AnalysisBaseClass instance contains a set
    of captions on which inference is performed. All
    analysis methods that work at on a single caption
    path are stored as static methods, shared by all
    instances.
    """
    
    man_word_list_synonyms = ['boy', 'brother', 'dad', 'husband', 'man', \
        'groom', 'male','guy', 'men', 'males', 'boys', 'guys', 'dads', 'dude', \
        'policeman', 'policemen', 'father', 'son', 'fireman', 'actor', \
        'gentleman', 'boyfriend', 'mans', 'his', 'obama', 'actors']
    woman_word_list_synonyms = ['girl', 'sister', 'mom', 'wife', 'woman', \
        'bride', 'female', 'lady', 'women', 'girls', 'ladies', 'females', \
        'moms', 'actress', 'nun', 'girlfriend', 'her', 'she']
    person_word_list_synonyms = ['person', 'child', 'kid', 'teenager', \
        'someone', 'player', 'rider', 'people', 'crowd', 'skiier', \
        'snowboarder', 'surfer', 'children', 'hipster', 'bikder', 'bikers', \
        'skiiers', 'snowboarders', 'surfers', 'chef', 'chefs', 'family', \
        'skateboarder', 'skateboarders', 'adult', 'adults', 'baby', 'babies', \
        'skiers', 'skier', 'diver', 'divers', 'bicycler', 'bicyclists', \
        'friends', 'kids', 'hikers', 'hiker', 'student', 'students', \
        'teenagers', 'riders', 'shopper', 'cyclist', 'officer', 'pedestrians', \
        'teen', 'worker', 'passenger', 'passengers', 'cook', 'cooks', \
        'officers', 'persons', 'workers', 'pedestrian', 'employee', \
        'employees', 'driver', 'cyclists', 'skater', 'skaters', 'toddler', \
        'fighter', 'patrol', 'patrols', 'cop', 'couple', 'server', 'carrier', \
        'player', 'players', 'motorcyclist', 'motorcyclists', 'youngsters', \
        'carpenter', 'owner', 'owners', 'individual', 'bicyclists', \
        'bicyclist', 'group', 'boarder', 'boarders', 'boater', 'painter', \
        'artist', 'citizen', 'youths', 'staff', 'biker', 'technician', 'hand', \
        'baker', 'fans', 'they', 'manager', 'plumber', 'hands', \
        'team', 'teams','performer', 'performers', 'couples', 'rollerblader']
    anno_dir = './data/'
    target_train = os.path.join(anno_dir, 'bias_splits/train.data')
    target_val = os.path.join(anno_dir, 'bias_splits/dev.data')
    target_test = os.path.join(anno_dir, 'bias_splits/test.data')

    def __init__(self, caption_paths):
        """
        caption_paths -- paths to captions that need to be analyzed
        """
        self.caption_paths = caption_paths
        AnalysisBaseClass.caption_paths = caption_paths
        # create annotation dictionary and simplified anno dict
        img_2_anno_dict = AnalysisBaseClass.create_dict_from_list(
            pickle.load(open(AnalysisBaseClass.target_train, 'rb'))
        )
        img_2_anno_dict.update(
            AnalysisBaseClass.create_dict_from_list(
                pickle.load(open(AnalysisBaseClass.target_test, 'rb'))
            )
        )
        img_2_anno_dict.update(
            AnalysisBaseClass.create_dict_from_list(
                pickle.load(open(AnalysisBaseClass.target_val, 'rb'))
            )
        )
        img_2_anno_dict_simple = AnalysisBaseClass.simplify_anno_dict(
            img_2_anno_dict
        )
        self.img_2_anno_dict = img_2_anno_dict
        self.img_2_anno_dict_simple = img_2_anno_dict_simple

    #import ipdb; ipdb.set_trace()
    gt_path = './data/captions_only_valtrain2014.json'
    gt_caps_list = json.load(open(gt_path))['annotations']
    gt_caps = collections.defaultdict(list)
    for cap in gt_caps_list:
        gt_caps[int(cap['image_id'])].append(cap['caption'])

    def accuracy(self, filter_imgs=None):
        """
        Returns a dictionary mapping model name to result dictionary. For each
        model, computes accuracy breakdown per class and across classes. Also
        computes the ratio of man to woman.
        """
        all_results = {}
        for caption_path in self.caption_paths:
            predictions = json.load(open(caption_path[1]))
            model_results, _ = AnalysisBaseClass.accuracy_per_model(
                predictions,
                AnalysisBaseClass.man_word_list_synonyms,
                AnalysisBaseClass.woman_word_list_synonyms,
                filter_imgs,
                self.img_2_anno_dict_simple
            )
            all_results[caption_path[0]] = model_results
        return all_results
   
    def retrieve_accuracy_with_confidence(self, filter_imgs=None):
        """
        Calculate the recall for the correct label over different levels of
        confidence for all caption paths.

        When the number of ground truth captions for an image meets the
        specified confidence threshold, the correct label is the gender.
        Otherwise, correct label is other or no gender.

        Returns dictionary mapping model names to a list with accuracy at
        different levels of confidence.
        
        e.g. [all_results]['path'][k] is the accuracy of captions at 'path'
        computed with confidence k+1
        """
        all_results = {}
        for caption_path in self.caption_paths:
            model_results = []
            for confidence in range(1,6):
                predictions = json.load(open(caption_path[1]))
                model_results.append(
                    AnalysisBaseClass.retrieve_accuracy_with_confidence_per_model(
                        predictions,
                        confidence,
                        self.img_2_anno_dict_simple,
                        AnalysisBaseClass.man_word_list_synonyms,
                        AnalysisBaseClass.woman_word_list_synonyms,
                        filter_imgs=filter_imgs
                    )
                )
            all_results[caption_path[0]] = model_results
        return all_results

    ###############################################
    ###  Helpers (metrics over predicted set)  ####
    ###############################################

    @staticmethod
    def get_gender_count(
        predictions, man_word_list=['man'], woman_word_list=['woman']
    ):
        """
        Get gender count and ratios for predicted sentences.
        """
   
        man_count, woman_count, other_count = 0., 0., 0.

        for prediction in predictions:
            sentence_words = nltk.word_tokenize(prediction['caption'].lower())
            pred_man = AnalysisBaseClass.is_gendered(sentence_words, 
                                                     'man', 
                                                     man_word_list, 
                                                     woman_word_list)
            pred_woman = AnalysisBaseClass.is_gendered(sentence_words, 
                                                       'woman', 
                                                       man_word_list, 
                                                       woman_word_list)
        
            if pred_man & pred_woman:
                other_count += 1
            elif pred_man: 
                man_count += 1  
            elif pred_woman: 
                woman_count += 1  
            else: 
                other_count += 1  

        ratio = woman_count/(man_count + 1e-5)
        print "Man count\tWoman count\tOther count\tWoman:Man ratio"
        print "%d\t%d\t%d\t%0.05f" %(man_count, woman_count, 
                                    other_count, ratio)

        return man_count, woman_count, other_count, ratio 

    @staticmethod
    def get_error(predictions, label_dict, 
                  man_word_list=['man'], woman_word_list=['woman']):
        
        num_incorrect = 0.
        num_gendered = 0.
        for prediction in predictions:
            sentence_words = nltk.word_tokenize(prediction['caption'].lower())
            pred_man = AnalysisBaseClass.is_gendered(sentence_words, 
                                                     'man', 
                                                     man_word_list, 
                                                     woman_word_list)
            pred_woman = AnalysisBaseClass.is_gendered(sentence_words, 
                                                       'woman', 
                                                       man_word_list, 
                                                       woman_word_list)
            pred_other = ((not pred_man) and (not pred_woman)) or (pred_man and pred_woman)
            pred_other = ((not pred_man) and (not pred_woman)) or (pred_man and pred_woman)
            gt_man = bool(label_dict[prediction['image_id']]['male']) 
            gt_woman = bool(label_dict[prediction['image_id']]['female'])

            #among examples where gender predicted, how frequently incorrect
            if (pred_man != gt_man) and (pred_woman != gt_woman) and (not pred_other):
                num_incorrect += 1  
            if not pred_other:
                num_gendered += 1

        return num_incorrect/num_gendered

    @staticmethod
    def accuracy_per_model(
        predicted,
        man_word_list=['man'],
        woman_word_list=['woman'],
        filter_imgs=None,
        img_2_anno_dict_simple=None
    ):
        """
        Returns accuracy breakdown and ratio of predictions.
        
        Args:
            predicted: list of dictionaries keyed by 'image_id' and 'caption'.
                predicted[i]['image_id'] is integer.
            X_word_list: list of synonyms used in computing accuracy of gender X
            filter_imgs: list of images (as integer id) to include in calculation.
                if None, include all of predicted.

        Returns: tuple containing result dictionoary and dictionary of images
            organized by gt x pred label
        """

        f_tp = 0.
        f_fp = 0.
        f_tn = 0.
        f_other = 0.
        f_total = 0.
        f_person = 0.
        
        
        m_tp = 0.
        m_fp = 0.
        m_tn = 0.
        m_other = 0.
        m_total = 0.
        m_person = 0.        

        male_pred_female = []
        female_pred_male = []
        male_pred_male = []
        female_pred_female = []
        male_pred_other = []
        female_pred_other = []
        
        for prediction in predicted:
            image_id = prediction['image_id']
            if filter_imgs:
                if image_id not in filter_imgs:
                    continue
            male = img_2_anno_dict_simple[image_id]['male']
            female = img_2_anno_dict_simple[image_id]['female']
            sentence_words = nltk.word_tokenize(prediction['caption'].lower())
            pred_male = AnalysisBaseClass.is_gendered(
                sentence_words, 'man', man_word_list, woman_word_list
            )
            pred_female = AnalysisBaseClass.is_gendered(
                sentence_words, 'woman', man_word_list, woman_word_list
            )
            pred_other = (not pred_male) and (not pred_female)
            pred_person = (pred_other) and AnalysisBaseClass.contains_words(
                sentence_words
            )

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
            if (female & pred_other):
                f_other += 1
                female_pred_other.append(prediction)
                if pred_person:
                    f_person += 1
            if (male & pred_other):
                m_other += 1
                male_pred_other.append(prediction)
                if pred_person:
                    m_person += 1
            if female:
                f_total += 1
            if male:
                m_total += 1

        ratio = (f_tp + f_fp) / (m_tp + m_fp)

        results = {}
        results['male_correct'] = m_tp/m_total
        results['male_incorrect'] = f_fp/m_total
        results['male_other'] =  m_other/m_total
        results['female_correct'] = f_tp/f_total
        results['female_incorrect'] = m_fp/f_total
        results['female_other'] = f_other/f_total
        results['all_correct'] = (m_tp+f_tp)/(m_total+f_total)
        results['all_incorrect'] = (m_fp+f_fp)/(m_total+f_total)
        results['all_other'] = (m_other+f_other)/(f_total+m_total)
        results['ratio'] = ratio
        results['gt_ratio'] = f_total/m_total

        pred_images = {}
        pred_images['male_pred_male'] = male_pred_male
        pred_images['female_pred_female'] = female_pred_female
        pred_images['female_pred_male'] = female_pred_male
        pred_images['male_pred_female'] = male_pred_female
        pred_images['male_pred_other'] = male_pred_other
        pred_images['female_pred_other'] = female_pred_other
        
        return (results, pred_images)

    @staticmethod
    def retrieve_accuracy_with_confidence_per_model(
        predicted,
        confidence_threshold,
        img_2_anno_dict_simple,
        man_word_list=['man'],
        woman_word_list=['woman'],
        filter_imgs=None
    ):
        correct = 0.
        incorrect = 0.
        correct_m, correct_f = 0., 0.
        incorrect_m, incorrect_f = 0., 0.
        pred_m, pred_f = 0., 0.
        not_m, not_f = 0., 0.
        pred_m_not_m, pred_f_not_f = 0., 0.
        pred_m_is_m, pred_f_is_f = 0., 0.
        is_f, is_m = 0., 0.
        total = 0.

        bias_ids = list(img_2_anno_dict_simple.keys())
        for prediction in predicted:
            image_id = prediction['image_id']

            # TODO: consolidate conditions
            if filter_imgs:
                if image_id not in filter_imgs:
                    continue
            
            anno_dict = img_2_anno_dict_simple
            if image_id in bias_ids:
                is_male = anno_dict[image_id]['male']
                is_female = anno_dict[image_id]['female']
                gt_captions = AnalysisBaseClass.gt_caps[image_id]
                
                total += 1

                if is_male:
                    gender = 'man'
                else:
                    gender = 'woman'

                is_conf = AnalysisBaseClass.confidence_level(
                    gt_captions, gender
                ) >= confidence_threshold

                sentence_words = nltk.word_tokenize(
                    prediction['caption'].lower()
                )
                pred_male = AnalysisBaseClass.is_gendered(
                    sentence_words, 'man', man_word_list, woman_word_list
                )
                pred_female = AnalysisBaseClass.is_gendered(
                    sentence_words, 'woman', man_word_list, woman_word_list
                )

                if pred_male:
                    pred_m += 1

                if pred_female:
                    pred_f += 1

                # TODO: simplify logic
                if is_conf:
                    if is_female:
                        not_m += 1
                        is_f += 1
                    if is_male:
                        not_f += 1
                        is_m += 1
                    if (is_female & pred_female):
                        correct += 1
                        pred_f_is_f += 1
                        correct_f += 1
                    elif (is_male & pred_male):
                        correct += 1
                        pred_m_is_m += 1
                        correct_m += 1
                    else:
                        if is_female:
                            incorrect_f += 1
                            if pred_male:
                                pred_m_not_m += 1
                        else:
                            incorrect_m += 1
                            if pred_female:
                                pred_f_not_f += 1
                        incorrect += 1
                else:
                    not_m += 1
                    not_f += 1
                    pred_other = (not pred_male) & (not pred_female)
                    if pred_other:
                        correct += 1
                        if is_female:
                            correct_f += 1
                        else:
                            correct_m += 1
                    else:
                        if pred_male:
                            pred_m_not_m += 1
                        else:
                            pred_f_not_f += 1
                        if is_female:
                            incorrect_f += 1 
                        else:
                            incorrect_m += 1 
                        incorrect += 1

        male_tpr = pred_m_is_m / float(is_m)
        male_fpr = pred_m_not_m / float(not_m)
        female_tpr = pred_f_is_f / float(is_f)
        female_fpr = pred_f_not_f / float(not_f)

        return correct / (incorrect+correct)

    def biased_objects(
        self,
        gt_path,
        filter_imgs=[], 
        models_to_test=['Baseline-FT', 'Equalizer w/o ACL', \
            'Equalizer w/o Confident', 'Equalizer']
    ):
        """
        Compute error and ratio for individual biased objects. Should replicate
        results in Table 1 of supplemental.
        """

        caption_paths_local = [caption_path for caption_path in self.caption_paths if 
                               caption_path[0] in models_to_test]

        bias_words = ['umbrella', 'kitchen', 'cell', 'table', 'food', \
            'skateboard', 'baseball', 'tie', 'motorcycle', 'snowboard']

        gt = AnalysisBaseClass.format_gt_captions(gt_path)

        for bias_word in bias_words:

            print "Bias word is: %s" %bias_word
            anno_ids = open('../data/object_lists/intersection_%s_person.txt' %bias_word).readlines()
            anno_ids = [int(anno_id) for anno_id in anno_ids]

            #TODO: get numbers for GT

            gt_filtered = [item for item in gt \
                               if item['image_id'] in (set(filter_imgs) & set(anno_ids))]
            print "Model: GT"
            gender_dict = self.img_2_anno_dict_simple
            gt_man = sum([1. for item in gt_filtered if \
                          gender_dict[item['image_id']]['male'] == 1])
            gt_woman = sum([1. for item in gt_filtered if \
                          gender_dict[item['image_id']]['female'] == 1])
            gt_ratio = gt_woman/gt_man

            for caption_path in caption_paths_local:
                print "Model: %s" %caption_path[0]
                predictions = json.load(open(caption_path[1]))
                captions = [item for item in predictions \
                            if item['image_id'] in set(filter_imgs) & set(anno_ids)]
                _, _, _, ratio = \
                          AnalysisBaseClass.get_gender_count(captions, 
                                            AnalysisBaseClass.man_word_list_synonyms, 
                                            AnalysisBaseClass.woman_word_list_synonyms)
                print "Delta ratio: %0.04f" %(np.abs(ratio-gt_ratio)) 

                error = AnalysisBaseClass.get_error(captions, gender_dict,
                                            AnalysisBaseClass.man_word_list_synonyms, 
                                            AnalysisBaseClass.woman_word_list_synonyms)
                print "Error: %0.04f" %error 

    @staticmethod
    def bias_amplification_objects(predictions):

        #Read in MSCOCO synonyms from synonyms.txt and pre-computed object
        #labels for each image.

        synonym_file = '../data/synonyms.txt'
        synonym_list = open(synonym_file).readlines()
        synonym_list = [line.strip().split(', ') for line in synonym_list]

        synonym_dict = {}
        mscoco_words = []
        for item in synonym_list:
            if 'person' != item[0]:
                for i in item:
                    synonym_dict[i] = item[0]
                mscoco_words.extend(item) 

        labels = pickle.load(open('../data/gt_labels.p', 'rb'))

        man_count = collections.defaultdict(int) 
        woman_count = collections.defaultdict(int)
        gt_all_count = collections.defaultdict(int)

        for count_p, prediction in enumerate(predictions):
    
            sys.stdout.write("\r%d/%d" %(count_p, len(predictions)))

            sentence_words = nltk.word_tokenize(prediction['caption'].lower().strip('.'))
            pred_male = AnalysisBaseClass.is_gendered(
                sentence_words, 
                'man', 
                AnalysisBaseClass.man_word_list_synonyms, 
                AnalysisBaseClass.woman_word_list_synonyms
            )
            pred_female = AnalysisBaseClass.is_gendered(
                sentence_words, 
                'woman', 
                AnalysisBaseClass.man_word_list_synonyms, 
                AnalysisBaseClass.woman_word_list_synonyms
            )

            sentence_words = [singularize(word) for word in sentence_words]
            num_words = len(sentence_words) - 1

            #This is to account for words like "hot dog".
            for i in range(num_words):
                sentence_words.append(
                    ' '.join([sentence_words[i], sentence_words[i+1]])
                )
            #which words are both in setence and mscoco objects
            count_words = list(set(sentence_words) & set(mscoco_words))        
     
            for word in count_words:
                if pred_male:
                    man_count[synonym_dict[word]] += 1
                elif pred_female:
                    woman_count[synonym_dict[word]] += 1
            for word in set(labels['labels'][prediction['image_id']]):
                gt_all_count[word] += 1 

        output_dict = {}
        output_dict['man'] = {}
        output_dict['woman'] = {}

        print "Bias amplification (comparison to man):"
        b = 0.
        count = 0
        for word in set(synonym_dict.values()):        
            if gt_all_count[word] > 0: 
                bb =  man_count[word]/(gt_all_count[word]+0.0005)
                output_dict['man'][word] = bb
                print "%s\t%0.03f" %(word, bb)

            b += bb 
            count += 1
        print "%0.03f" %(b/count) 
 
        print "Bias amplification (comparison to woman):"
        b = 0.
        count = 0
        for word in set(synonym_dict.values()):        
            if gt_all_count[word] > 0: 
                bb =  woman_count[word]/(gt_all_count[word]+0.0005)
                output_dict['woman'][word] = bb
                print "%s\t%0.03f" %(word, bb)
            b += bb 
            count += 1
        print "%0.03f" %(b/count) 
        return output_dict

    @staticmethod
    def bias_amplification_objects_stats(
        gt_path, filter_imgs=[], models_to_test=['Baseline-FT', 'Equalizer']):
        """
        Computes bias amplification for baseline and Equalizer captions.
        Inputs:
            gt_path:  Path to ground truth captions
            filter_imgs: List of image ids.
        Outputs:
            bias amplification numbers as described in
            ``Object Gender Co-Occurrence''  
        """

        gt = AnalysisBaseClass.format_gt_captions(gt_path)
        gt = [item for item in gt if item['image_id'] in filter_imgs]
        gt_output = AnalysisBaseClass.bias_amplification_objects(gt)

        caption_paths_local = [caption_path for caption_path
            in AnalysisBaseClass.caption_paths
            if caption_path[0] in models_to_test]

        for caption_path in caption_paths_local:

            predictions = json.load(open(caption_path[1]))
            captions = [item for item
                in predictions
                if item['image_id'] in filter_imgs]
            gen_output = AnalysisBaseClass.bias_amplification_objects(captions)
    
            #get absolute mean of difference
    
            print("Average bias amplification across objects for man for caption model %s: " %caption_path[0])
            obj_diffs = []
            for obj in gt_output['man']:
                obj_diffs.append(
                    np.abs(gt_output['man'][obj] - gen_output['man'][obj])
                )
            print("Average bias amplification for man: %0.04f" %(np.mean(obj_diffs))) 
    
            print ("Average bias amplification across objects for woman for caption model %s: " %caption_path[0])
            obj_diffs = []
            for obj in gt_output['woman']:
                obj_diffs.append(
                    np.abs(gt_output['woman'][obj] - gen_output['woman'][obj])
                )
            print ("Average bias amplification for woman: %0.04f" %(np.mean(obj_diffs))) 

    ###############################################
    ########             Utils             ########
    ###############################################

    

    @staticmethod
    def confidence_level(caption_list, gender):
        """Returns number of captions that say correct gender."""
        conf = 0
        for cap in caption_list:
            if AnalysisBaseClass.is_gendered(
                nltk.word_tokenize(cap.lower()), gender_type=gender
            ):
                conf += 1
        return conf

    @staticmethod
    def convert_filenames_to_ids(fnames):
        """Converts a list of COCO filenames to list of corresponding ids."""
        new_list = []
        for fname in fnames:
            new_list.append(int(fname.split('.')[0].split('_')[2]))
        return new_list

    @staticmethod
    def contains_words(words, word_list=person_word_list_synonyms):
        """
        Returns true if the words in words contain
        any of the words in the word_list.
        """
        words = [str(word) for word in words]
        word_list = [str(word) for word in word_list]
        return len(set(words) & set(word_list)) > 0

    @staticmethod
    def is_gendered(
        words,
        gender_type='woman',
        man_word_list=['man'],
        woman_word_list=['woman']
    ):
        """
        Returns true if the words in words contain a gender word from the
        specified gender type. If the caption contains more than one gender,
        return False.
        """
        m = False
        f = False
        check_m = (gender_type == 'man')
        check_f = (gender_type == 'woman')
        m = AnalysisBaseClass.contains_words(words, man_word_list)
        f = AnalysisBaseClass.contains_words(words, woman_word_list)
        if m != f:
            return (m and check_m) or (f and check_f)
        else:
            return False

    @staticmethod
    def format_gt_captions(gt_file):
        gt = json.load(open(gt_file))
        gt_caps = []
        for annotation in gt['annotations']:
            gt_caps.append({
                'image_id': annotation['image_id'],
                'caption': annotation['caption']
            })
        return gt_caps

    @staticmethod
    def simplify_anno_dict(img_2_anno_dict):
        img_2_anno_dict_simple = {}
        for key, value in img_2_anno_dict.items():
            id = int(key.split('_')[-1].split('.jpg')[0])
            img_2_anno_dict_simple[id] = {}
            img_2_anno_dict_simple[id]['male'] = value[0]
            img_2_anno_dict_simple[id]['female'] = int(not value[0])
            assert int(not value[0]) == value[1]

        return img_2_anno_dict_simple

    @staticmethod
    def get_shopping_split(
        fpath='./data/bias_splits/dev.data'
    ):
        """Returns split from men also like shopping as list of filenames."""
        data = []
        samples = pickle.load(open(fpath, 'rb'))
        for sample in samples:
            data.append(sample['img'])

        return data

    @staticmethod
    def create_dict_from_list(samples):
        """
        Converts ground truth annotations from reducing bias paper to dictionary
        indexed by image name.
        """
        img2anno = {}
        for sample in samples:
            img = sample['img']
            img2anno[img] = sample['annotation']
        return img2anno

    
    @staticmethod
    def get_confident_split(img_2_anno_dict_simple, conf_level=4):
        """
        Return list of image ids with confidence greater than or equal to
        conf_level. Confidence is number of ground truth human annotators
        who said correct_gender.

        returns: a list of captions with specified confidence
        """
        bias_ids = list(img_2_anno_dict_simple.keys())
        conf_ids = []
        img_2_anno = img_2_anno_dict_simple
        conf_calculator = AnalysisBaseClass.confidence_level

        for image_id in bias_ids:
            gt_captions = AnalysisBaseClass.gt_caps[image_id]
            is_male = img_2_anno[image_id]['male']
            is_female = img_2_anno[image_id]['female']
            if is_male:
                gender = 'man'
            else:
                gender = 'woman'
            if conf_calculator(gt_captions, gender) >= conf_level:
                conf_ids.append(image_id)
        return conf_ids

