tf_efficientdet_d?_ap


```
Algorithm 1: Det-AdvProp

**Input:** Object detection dataset D

**Output:** Learned network parameter θ

**for each training epoch do:**

  1. Sample a random batch `{x_i, (y_i, b_i)}` from dataset D
  2. Generate adversarial example `x_i^{cls}` based on classification loss L_cls(x_i, y_i) using auxiliary batchnorm
  3. Generate adversarial example `x_i^{loc}` based on localization loss L_loc(x_i, b_i) using auxiliary batchnorm
  4. Select final adversarial example `x_i` based on Equation (5) (replace with actual equation if available)
  5. Compute detection loss L_det(x_i, (y_i, b_i)) with main batchnorm
  6. Compute detection loss L_det(x_i^{adv}, (y_i, b_i)) with auxiliary batchnorm
  7. Perform a step of gradient descent w.r.t. θ:
     min( L_det(x_i, y_i, b_i) + L_det(x_i^{adv}, y_i, b_i) )
  8. **end for**
  ```
```
# Define data paths and splits
data_dir = '/kaggle/input/coco-2017-dataset/coco2017'
splits = {
    'train': {
        'ann_filename': f"{data_dir}/annotations/instances_train2017.json",
        'img_dir': f"{data_dir}/train2017",
    },
    'val': {
        'ann_filename': f"{data_dir}/annotations/instances_val2017.json",
        'img_dir': f"{data_dir}/val2017",
    },
}
```
```
import tensorflow as tf
train_img_dataset = tf.data.Dataset.list_files(f"{splits['train']['img_dir']}/*.jpg")
```
```
%matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
```
```
from keras_cv_attention_models import efficientdet
model = efficientdet.EfficientDetD0(pretrained="coco")

# Run prediction
from keras_cv_attention_models import test_images
imm = test_images.dog_cat()
preds = model(model.preprocess_input(imm))
bboxs, lables, confidences = model.decode_predictions(preds)[0]

# Show result
from keras_cv_attention_models.coco import data
data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=90)
```
```
"root":{3 items
"info":{...}6 items
"licenses":[...]8 items
"images":368 items
[100 items
0:{8 items
"license":int3
"file_name":string"000000391895.jpg"
"coco_url":string"http://images.cocodataset.org/train2017/000000391895.jpg"
"height":int360
"width":int640
"date_captured":string"2013-11-14 11:18:45"
"flickr_url":string"http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg"
"id":int391895
}
1:{8 items
"license":int4
"file_name":string"000000522418.jpg"
"coco_url":string"http://images.cocodataset.org/train2017/000000522418.jpg"
"height":int480
"width":int640
"date_captured":string"2013-11-14 11:38:44"
"flickr_url":string"http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg"
"id":int522418
}
2:{8 items
"license":int3
"file_name":string"000000184613.jpg"
"coco_url":string"http://images.cocodataset.org/train2017/000000184613.jpg"
"height":int336
"width":int500
"date_captured":string"2013-11-14 12:36:29"
"flickr_url":string"http://farm3.staticflickr.com/2169/2118578392_1193aa04a0_z.jpg"
"id":int184613
}
```

```
OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information
```
