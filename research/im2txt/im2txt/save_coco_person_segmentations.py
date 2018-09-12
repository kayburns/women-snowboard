import os
import sys
import json
import os.path as osp
import PIL.Image
import matplotlib
import sys
sys.path.append('im2txt/data/mscoco/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import matplotlib.image as mpimg


def get_polygons(anns, catIds, W, H):
    polygons = []
    for ann in anns:
        if ann['category_id'] in catIds:
            # polygon            
            for seg in ann['segmentation']:
                poly = []
                num_points = len(seg) // 2
                for i in xrange(num_points):
                    x, y = seg[2 * i], seg[2 * i + 1]
                    assert(x<=W)
                    assert(y<=H)
                    poly.append((min(x,W-1), min(y,H-1)))
                poly = np.array(poly)
                polygons.append(poly)                
    return polygons

def save_person_segmentations(coco, cocoImgDir, coco_masks, img_path):    
    catIds = coco.getCatIds(catNms='person');
    person_id = catIds[0]

    lines = open(img_path, 'r').readlines()
    imgIds = [int(line) for line in lines]
    imgIds = list(set(imgIds) & set(coco.getImgIds()))

    for i, imgId in enumerate(imgIds):
        sys.stdout.write('\r%d/%d' %(i, len(imgIds)))
        img = coco.loadImgs(imgId)[0]
        save_file = '%s/COCO_%s_%012d.npy' %(coco_masks, dataType, img['id'])
        if not os.path.isfile(save_file):
            fig = plt.figure(frameon=False)#
            ax = fig.gca()#        
            I=mpimg.imread('%s/COCO_%s_%012d.jpg' %(cocoImgDir, dataType, imgId))
            plt.cla()#
            H, W = I.shape[0:2]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
            anns = coco.loadAnns(annIds)
            polygons = get_polygons(anns, [person_id], W, H)      
            seg_map = np.zeros(I.shape[:2])
            for poly in polygons:
                rr, cc = polygon(poly[:, 0], poly[:, 1])
                seg_map[cc, rr] = 1
            #plt.imshow(seg_map)#
            #plt.show()#
            np.save(save_file, seg_map)
    print('done')

if __name__ == "__main__":
    coco_dir = 'im2txt/data/mscoco/' # Anja: make it an argument

    dataType = 'val2014'
    annFile='{}/annotations/instances_{}.json'.format(coco_dir,dataType)
    cocoImgDir = '{}/images/{}/'.format(coco_dir, dataType)
    coco_masks = '{}/masks/{}/'.format(coco_dir, dataType)

    if not os.path.isdir(coco_masks):
        os.makedirs(coco_masks)
   
    coco=COCO(annFile)

    save_person_segmentations(coco, cocoImgDir, coco_masks, './data/balanced_split/val_woman.txt')
    save_person_segmentations(coco, cocoImgDir, coco_masks, './data/balanced_split/val_man.txt')
    save_person_segmentations(coco, cocoImgDir, coco_masks, './data/balanced_split/test_woman.txt')
    save_person_segmentations(coco, cocoImgDir, coco_masks, './data/balanced_split/test_man.txt')

