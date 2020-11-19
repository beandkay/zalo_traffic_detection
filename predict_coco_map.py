
%matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
# import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

gt_path = ''
res_path = ''
cocoGt=COCO(gt_path)

cocoDt=cocoGt.loadRes(res_path)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = cocoGt.getImgIds()
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()