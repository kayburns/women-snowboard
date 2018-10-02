#Code for earth mover distance: https://github.com/wmayner/pyemd

from pyemd import emd
import numpy as np
import pdb
import sys
from scipy.stats import spearmanr
import random
import scipy.io as scio
from scipy.misc import imresize
import json

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

class heatmap_metrics(object):

  def __init__(self, gt_map, gen_map, gt_type='human', SIZE=(14,14)):
    gen_maps = []
    gt_maps = []

    def process_multi_channel(m):
      m = np.sum(m, axis=0)
      m = m/np.sum(m)
      return m

    def random_anno():
      x = int(random.random()*SIZE[0]) 
      y = int(random.random()*SIZE[1]) 
      mat = np.zeros(SIZE)
      mat[x,y] = 1
      return mat

    def uniform_anno():
      mat = np.ones(SIZE)/(SIZE[0]*SIZE[1])
      return mat

    gen = gen_map
    gt = gt_map

    if gt_type == 'human':
      gen = gen_map
    elif gt_type == 'random':
      gen = random_anno()
    elif gt_type == 'uniform':
      gen = uniform_anno()
    else:
      raise Exception ("Must pick valid annotation type")

    if len(gen.shape) > 2:
      gen = process_multi_channel(gen)

    assert gt.shape == gen.shape
    assert np.abs(np.sum(gen)-1.0) < 0.00005
    gt_maps.append(gt)
    gen_maps.append(gen)
    
    self.gen = gen_maps
    self.gt = gt_maps
    self.gt_type = gt_type

  def earth_mover(self, distance='manhattan'):
    #make distance matrix

    def manhattan(a,b):
      return np.abs(a[0]-b[0]) + np.abs(a[1]-b[1])

    def euclidean(a,b):
      return np.sqrt(np.sum((np.array(a)-np.array(b))**2)) 

    def squared(a,b):
      return np.sum((np.array(a)-np.array(b))**2) 

    distance_metric_dict = {'manhattan': manhattan,
                            'euclidean': euclidean,
                            'squared': squared}

    assert distance in distance_metric_dict.keys()
    distance_fnc = distance_metric_dict[distance]

    N = self.gen[0].shape[0]
    distance_matrix = np.zeros((N**2,N**2))
    for x1 in range(N):
      for y1 in range(N):
        for x2 in range(N):
          for y2 in range(N):
            d = distance_fnc((x1, y1), (x2,y2))
            X = x1*N + y1
            Y = x2*N + y2
            distance_matrix[X,Y] = d 
    assert np.sum(distance_matrix - distance_matrix.T) == 0 #distance matrix should be symmetric

    distances = []
    print "Compute distance matrix"
    count = 0
    for gen, gt in zip(self.gen, self.gt):
      d = emd(gen.flatten().astype('float64'), gt.flatten().astype('float64'), 
              distance_matrix.astype('float64'))
      distances.append(d)
      count += 1
      sys.stdout.write('\r%d' %count)

    em = np.mean(distances)
    return em, distances 

  def spearman_correlation(self):
    
    def mean_rank_preprocess(mat):
      mat = mat.flatten()
      mat = mat + 1e-14 * np.random.randn(mat.shape[0])
      mat = mat/np.linalg.norm(mat)
      return mat  

    def argsort_tie(mat):
      sorted_mat = np.empty_like(mat)
      count = 0
      prev = np.min(mat)
      i = 0
      for s, a in zip(np.sort(mat), np.argsort(mat)):
        if s > prev:
          count += 1
          prev = s
        sorted_mat[i] = count
        i += 1
      sorted_mat = sorted_mat/np.linalg.norm(sorted_mat)
      return sorted_mat

    distances = []
    for gen, gt in zip(self.gen, self.gt):
      gen = mean_rank_preprocess(gen)
      gt = mean_rank_preprocess(gt)
      dist, p_val = spearmanr(gen, gt, axis=None)
      distances.append(dist)
    mrc = np.mean(distances)
    return mrc, distances

  def mean_rank_correlation(self):
    
    def mean_rank_preprocess(mat):
      mat = mat.flatten()
      mat = mat/np.linalg.norm(mat)
      return mat  

    def argsort_tie(mat):
      sorted_mat = np.empty_like(mat)
      count = 0
      prev = np.min(mat)
      i = 0
      for s, a in zip(np.sort(mat), np.argsort(mat)):
        if s > prev:
          count += 1
          prev = s
        sorted_mat[a] = count
        i += 1
      return sorted_mat

    distances = []
    for gen, gt in zip(self.gen, self.gt):
      gen = mean_rank_preprocess(gen)
      gt = mean_rank_preprocess(gt)
      gen_rank = argsort_tie(gen) 
      gt_rank = argsort_tie(gt)
      dist, p_val = spearmanr(gen_rank, gt_rank, axis=None)
      distances.append(dist)
    mrc = np.mean(distances)
    return mrc 

  def iou(self):
    ious = []
    for gen, gt in zip(self.gen, self.gt):
      gen_flat = gen.flatten()
      gt_flat = gt.flatten()
      products = [gen_flat[i]*gt_flat[i] for i in range(len(gen_flat))]
      ious.append(np.sum(products))
    m_iou = np.mean(ious)
    return m_iou

  def pointing(self):
    pointings = []
    for gen, gt in zip(self.gen, self.gt):
      gen_flat = gen.flatten()
      gt_flat = gt.flatten()
      p = gt_flat[np.argmax(gen_flat)]
      pointings.append(p)
    m_pointing = np.mean(pointings)
    return m_pointing


