from fastai import *
import fastai
from fastai.vision import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from pathlib import Path
import torch

def update_swish():
  """
  Patches pytorchcv implementation of Swish Activation function
  with a Torch Autograd Function to optimize memory usage in
  backwards pass.
  """
  print("Updating Swish Activation to a Torch Autograd Function...")
  with open('./common.py','r') as writer_file:
    contents_to_write = writer_file.read()
  with open('./lib/python3.7/site-packages/pytorchcv/models/common.py' ,'w') as file_to_overwrite:
    file_to_overwrite.write(contents_to_write)
    print("Swish Activation is Optimized.")

def extract_data(file_path):
  """
  Extracts working directory (data, label files) from zip archive.
  Args:
    - file_path -> type:str -> path to data archive.
  """
  data_zip = zipfile.ZipFile(file_path)
  data_zip.extractall()
  path = Path("./data")
  path.ls()
  return path



def get_data(sz, tfs, bs=16, path=Path(".")):
  """
  Automates the instantiation of DataLoaders with oversampling of
  minority class.
  Args:
    - sz --> type:int --> img size.
    - tfms --> type:List --> transforms.
    - batch --> type:int, default=8 --> batch size.
    - workers --> type:int, default=4 --> number of workers for each dataloader
  """
  class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Args:
        - indices (list, optional): a list of indices
        - num_samples (int, optional): number of samples to draw
    """
    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            for l in label:
                if l in label_to_count:
                    label_to_count[l] += 1
                else: label_to_count[l]=1

        # weight for each sample
        weights = [1.0 / min([label_to_count[l] for l in self._get_label(dataset, idx)])
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.y[idx].obj #for category obj

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
  labels = pd.read_csv(path/'data/train_split_v2.csv', header='infer')

  data = ImageDataBunch.from_df(path,
                                labels,
                                folder='data/train',
                                ds_tfms=tfs, fn_col=1,
                                label_col=2,
                                size=sz,
                                resize_method=ResizeMethod.SQUISH,
                                bs=bs)
  data.normalize(imagenet_stats)
  train_ds, val_ds = data.train_ds, data.valid_ds
  sampler = ImbalancedDatasetSampler(train_ds)
  train_dl = DataLoader(train_ds, bs, sampler=sampler)
  val_dl = DataLoader(val_ds, 2*bs, False)

  data = ImageDataBunch(train_dl=train_dl,
                        valid_dl=val_dl).normalize(imagenet_stats)
  return data


def arch_summary(arch):
  """
  A helper for listing the sequential layer groups of the network
  Args:
    - arch --> type:nn.ModuleList --> A list of modules.
  https://github.com/PPPW/deep-learning-random-explore/blob/master/CNN_archs/utils.py
  """
  model = arch(False)
  tot = 0
  for i, l in enumerate(model.children()):
      n_layers = len(flatten_model(l))
      tot += n_layers
      print(f'({i}) {l.__class__.__name__:<12}: {n_layers:<4}layers (total: {tot})')



def get_groups(m, layer_groups):
  """
  a helper method for listing the layer groups after the layer cuts have been
  determined for discriminative learning rates.
  Args:
    - m --> type:nn.ModuleList/nn.Sequential --> A list of modules.
    - layer_groups --> type:list --> List of layers grouped by split defined in
    Learner class.
  """
  group_indices = [len(g) for g in layer_groups]
  curr_i = 0
  group = []
  for layer in model:
      group_indices[curr_i] -= len(flatten_model(layer))
      group.append(layer.__class__.__name__)
      if group_indices[curr_i] == 0:
          curr_i += 1
          print(f'Group {curr_i}:', group)
          group = []


def efficientnet_b1(pretrained=True):
  """
  Returns efficientnet-b1 feature extraction layers, using pytorchcv's
  ptcv_get_model helper method.
  Args:
    pretrained --> type:bool (default: True)
  """
  return ptcv_get_model("efficientnet_b1", pretrained=pretrained).features

#Fastai Learner (Model with built-in Training Tools)


def get_learner(train_data):
  """
  Generates a fastai Learner with passed fastai ImageDataBunch data loaders and
  feature extraction layers from a pretrained EfficientNet-b1 network.
  Uses Half-Tensor Precision.
  Args:
    train_data --> type:ImageDataBunch
  """
  learn = cnn_learner(train_data,
                      efficientnet_b1,
                      metrics=[error_rate, accuracy],
                      callback_fns=[ShowGraph])
  return learn


def get_metrics(cm):
  """
  Tabulates relevant metrics for medical diagnosis from stored rows/diagonal
  entries of confusion matrics; Returns an array of metrics:true positives (tp),
  false positives (fp), true negatives (tn), and false negatives (fn) for each
  class in a 3x3 confusion matrix (12 int64 values).
  Args:
    cm --> type:np.darray --> a confusion matrix.
  """
  tp_c1 = cm[0][0]
  fp_c1 = cm[1][0] + cm[2][0]
  tn_c1 = np.sum(cm) - tp_c1
  fn_c1 = cm[0][1] + cm[0][2]

  tp_c2 = cm[1][1]
  fp_c2 = cm[0][1] + cm[2][1]
  tn_c2 = np.sum(cm) - tp_c2
  fn_c2 = cm[1][0] + cm[1][2]

  tp_c3 = cm[2][2]
  fp_c3 = cm[0][2] + cm[1][2]
  tn_c3 = np.sum(cm) - tp_c3
  fn_c3 = cm[2][0] + cm[2][1]

  return [tp_c1,fp_c1,tn_c1,fn_c1,tp_c2,fp_c2,tn_c2,fn_c2, tp_c3,fp_c3,tn_c3,fn_c3]

def get_precision_scores(metrics):
  """
  Tabulates Precision Scores from list of key metrics.
  Args:
  - metrics -> type:list --> output from get_metrics()
  """
  covid_precision = metrics[0]/(metrics[0]+metrics[1])
  norm_precision = metrics[4]/(metrics[4]+metrics[5])
  pneum_precision = metrics[8]/(metrics[8]+metrics[9])
  train_precision_scores = {
    "COVID-19": covid_precision,
    "NORMAL": norm_precision,
    "PNEUMONIA": pneum_precision
  }
  train_precision_scores = pd.DataFrame(train_precision_scores,
                                      columns=train_precision_scores.keys(),
                                      index=[0])
  print(train_precision_scores)
  #return train_precision_scores



def get_sensitivity_scores(metrics):
  """
  Tabulates Sensitivity Scores from list of key metrics.
  Args:
  - metrics -> type:list --> output from get_metrics()
  """
  covid_recall = training_metrics[0]/(training_metrics[0]+training_metrics[3])
  norm_recall = training_metrics[4]/(training_metrics[4]+training_metrics[7])
  pneum_recall = training_metrics[8]/(training_metrics[8]+training_metrics[11])
  train_sensitivity_scores = {
    "COVID-19": covid_recall,
    "NORMAL": norm_recall,
    "PNEUMONIA": pneum_recall
    }
  train_sensitivity_scores = pd.DataFrame(train_sensitivity_scores,
                         columns=train_sensitivity_scores.keys(),
                         index=[0])
  print(train_sensitivity_scores)
  #return train_sensitivity_scores



def get_specificity(metrics):
  """
  Tabulates Specificity Scores from list of key metrics.
  Args:
  - metrics -> type:list --> output from get_metrics()
  """
  covid_specificity = metrics[2]/(metrics[2]+metrics[1])
  norm_specificity = metrics[6]/(metrics[6]+metrics[5])
  pneum_specificity = metrics[10]/(metrics[10]+metrics[9])
  specificity_scores = {"COVID-19": covid_specificity,
                      "NORMAL": norm_specificity,
                      "PNEUMONIA": pneum_specificity}
  specificity_df = pd.DataFrame(specificity_scores,
                              columns=specificity_scores.keys(),
                              index=[0])
  print(specificity_df)
  #return specificity_df



def run_test():
  """
  todo
  """
  pass

def generate_report():
  """
  todo
  """
  pass





