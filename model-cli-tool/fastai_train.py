#utils
from __future__ import print_function
from __future__ import division
import os
import json
import argparse
import zipfile
import shutil
import subprocess

#data science/ml
from pytorchcv.model_provider import get_model as ptcv_get_model
import fastai
from fastai.vision import *

import warnings
warnings.filterwarnings('ignore')
#import helpers

from helpers import update_swish, get_data, get_learner, get_metrics, get_precision_scores, get_sensitivity_scores, get_specificity, run_test

print("Using fast.ai version: ", fastai.__version__)
update_swish()


##### SETUP/GET ARGS #####
parser = argparse.ArgumentParser(description='Retrains EfficientNet-b1 on COVIDx dataset using Fast.ai')

parser.add_argument('--data',
                       type=Path,
                       default=None,
                       help='Path to data archive.')


parser.add_argument('--bs',
                       type=int,
                       default=16,
                       help='Batch Size. Defaults to 8.')

parser.add_argument('--size',
                       type=int,
                       default=240,
                       help='Image Size. Defaults to 240x240.')

parser.add_argument('--transforms',
                       type=list,
                       default=[(contrast(scale=(1.5, 1.85), p=0.7), crop_pad(),
                        *zoom_crop(scale=(1.15,1.5), do_rand=True, p=.2)), () ],
                        help='Fastai Transforms to apply. For all transforms, visit docs at https://docs.fast.ai/vision.transform.html')

parser.add_argument('--freeze',
                       type=bool,
                       default=False,
                       help='Train Frozen (True) or Unfrozen(False). Defaults to False.')

parser.add_argument('--cycles',
                       type=int,
                       default=6,
                       help='Number of One-Policy-Cycles. Defaults to 6.')

parser.add_argument('--max_lr',
                       type=int,
                       default=4e-3,
                       help='Max learning rate for this run. Defaults to 4e-3.')

parser.add_argument('--wd',
                       type=int,
                       default=2e-3,
                       help='Weight decay for this run. Defaults to 2e-3.')


parser.add_argument('--with_metrics',
                       type=bool,
                       default=True,
                       help='Whether or not to report Key Metrics after run.')

parser.add_argument('--checkpoint_file',
                       type=str,
                       default=None,
                        help='Path to checkpoint.')

parser.add_argument('--save',
                       type=bool,
                       default=True,
                       help='Whether or not to save checkpoint after run.')

parser.add_argument('--save_name',
                       type=str,
                       default=None,
                        help='Checkpoint name for saving.')


print("Parsing Args...")
args = parser.parse_args()
data_zip = zipfile.ZipFile(str(args.data))
bs = args.bs
size = args.size
tfms = args.transforms
retrain = args.freeze
cycles = args.cycles
max_lr = args.max_lr
wd = args.wd
report_metrics = args.with_metrics
checkpoint_file = args.checkpoint_file
save = args.save
save_name = args.save_name
print("Done parsing args.")

print("Creating DataBunch and Learner for training.")
print("Extracting files...")
data_zip.extractall(".")
print("Done.")
print("Instantiating DataBunch and Learner...")
data = get_data(sz=size, tfs=tfms)
learn = get_learner(data)
if checkpoint_file is not None:
  learn.load(checkpoint_file)
  print("Checkpoint loaded.")
if retrain:
  learn.unfreeze()
  print("Model state: Unfrozen.")
else:
  learn.freeze()
  print("Model state: Frozen.")

print("DataBunch and Learner ready to go.")

##### TRAIN/FINE-TUNE #####
print(f"Starting Training for {cycles} cycles; Hyperparams: bs={bs}, lr={max_lr}, wd={wd}")
learn.fit_one_cycle(cycles, slice(max_lr), wd)
if save:
  print("Saving Model...")
  learn.save(save_name)
  print("Done.")

##### TRAIN/FINE-TUNE #####
if report_metrics:
  print("Preparing Confusion Matrix")
  interp = ClassificationInterpretation.from_learner(learn)
  losses, idxs = interp.top_losses()
  print(f"Samples, predictions, and losses of equal length: {len(data.valid_ds)==len(losses)==len(idxs)}")
  print("Most Confused Classes")
  print(interp.most_confused(min_val=2))
  print("Confusion Matrix:")
  print(interp.confusion_matrix())
  conf_matrix = interp.confusion_matrix()
  training_metrics = get_metrics(conf_matrix)
  print("Precision Scores:")
  get_precision_scores(training_metrics)
  print("Sensitivity Scores:")
  get_sensitivity_scores(training_metrics)
  print("Specificity Scores:")
  get_specificity(training_metrics)

print("Training Complete.")



