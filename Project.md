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
