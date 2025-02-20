import numpy as np

from skimage.color import gray2rgb, rgb2gray

from skimage.segmentation import mark_boundaries

import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
import sys
import json

import innvestigate
import innvestigate.utils
from keras import backend as K
from keras.models import Model

import gc

class InnvestigateExplainer(object):
  """docstring for InnvestigateExplainer"""
  def __init__(self, model, explanation_method=None):
    super(InnvestigateExplainer, self).__init__()
    self.model = model
    if explanation_method is None:
        self.explanation_method = "gradient"
    else:
        self.explanation_method = explanation_method

    self.requires_fresh_session = True
		
  def GetColorMap(self):
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((0./255, 0./255, (129.+l*100)/255,l))
    for l in np.linspace(0, 1, 100):
        colors.append(((155.+l*100)/255, 0./255, 0./255,l))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    return red_transparent_blue
  
  def InnvestigateAttributionToImage(self, innvestigate_attribution):
    img = np.sum(innvestigate_attribution, axis=2)
    
    return img*255

  def GenerateInnvestigateExplanationImage(self,input_image,explanation_values):
    # print(np.max(explanation_values))
    # print(np.min(explanation_values))
    # print("explanation_values.shape",explanation_values.shape)
    
    if(len(input_image.shape) == 4):
      input_image = np.squeeze(input_image)
    
    x_curr = input_image.copy()

    # make sure
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
        x_curr = x_curr.reshape(x_curr.shape[:2])
    if x_curr.max() > 1:
        x_curr /= 255.

    if len(explanation_values[0].shape) == 2:
        abs_vals = np.stack([np.abs(explanation_values[i]) for i in range(len(explanation_values))], 0).flatten()
    else:
        abs_vals = np.stack([np.abs(explanation_values[i].sum(-1)) for i in range(len(explanation_values))], 0).flatten()
    max_val = np.nanpercentile(abs_vals, 70)

    fig, ax = plt.subplots()
    fig.subplots_adjust(0,0,1,1)
    
    plt.autoscale(tight=True)
    plt.gcf().set_size_inches(10,10) 
    
    sv = explanation_values
    plt_img = plt.imshow(x_curr, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(0, sv.shape[0], sv.shape[1], 0))
    plt.imshow(sv, cmap=self.GetColorMap(), vmin=-max_val, vmax=max_val)
    plt.axis('off')

    ax = plt.gca()
    canvas = ax.figure.canvas 
    canvas.draw()

    w,h = canvas.get_width_height()
    buf = np.fromstring ( canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( h, w,3 )
    
    buf = cv2.resize(buf, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    plt.gcf().clear()
    plt.gca().clear()
    plt.clf()
    plt.cla()
    plt.close()
    
    return buf

    
  def Explain(self,input_image, additional_args = {}):

    #load additional arguments or set to default
    
    # cv2_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("x image",cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if(len(input_image.shape) == 3):
        input_image = np.array([input_image])      
    
    try:
      input_image = self.model.CheckInputArrayAndResize(input_image,self.model.min_height,self.model.min_width)
    except:
      print("couldn't use model image size check")    
    
    fmodel = innvestigate.utils.model_wo_softmax(self.model.model)
    analyzer = innvestigate.create_analyzer(self.explanation_method, fmodel, neuron_selection_mode="index")

    prediction_scores,prediction  = self.model.Predict(input_image, True)
    print(prediction)
    print(prediction_scores)
    predicted_class = np.argmax(prediction_scores)

    innvestigate_values = [analyzer.analyze(input_image, predicted_class)]
    print(innvestigate_values[0].shape)

    # cv2_image = cv2.cvtColor(explanation_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("explanation_image LRP",cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ## for testing:
    # shap.image_plot(shap_values, np.multiply(input_image,255.0))

    if(not isinstance(prediction_scores,list)):
      prediction_scores = prediction_scores.tolist()

    attributions_list = [innv_value.tolist() for innv_value in innvestigate_values]
    attributions_list = attributions_list[0][0]
        
    additional_outputs = {"attribution_map":attributions_list, "innvestigate_values":[innv_value.tolist() for innv_value in innvestigate_values],"prediction_scores":prediction_scores}

    explanation_text = "Evidence towards predicted class shown in red."
    del fmodel
    del analyzer
    del innvestigate_values
    gc.collect()
    
    return None, explanation_text, predicted_class, additional_outputs
  


if __name__ == '__main__':
  import os
  import sys
  

  ### Setup Sys path for easy imports
  # base_dir = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
  # base_dir = "/media/upsi/fs1/harborned/repos/interpretability_framework"

  def GetProjectExplicitBase(base_dir_name="interpretability_framework"):
    cwd = os.getcwd()
    split_cwd = cwd.split("/")

    base_path_list = []
    for i in range(1, len(split_cwd)):
      if(split_cwd[-i] == base_dir_name):
        base_path_list = split_cwd[:-i+1]

    if(base_path_list == []):
      raise IOError('base project path could not be constructed. Are you running within: '+base_dir_name)

    base_dir_path = "/".join(base_path_list)

    return base_dir_path

  base_dir = GetProjectExplicitBase(base_dir_name="interpretability_framework")


  #add all model folders to sys path to allow for easy import
  models_path = os.path.join(base_dir,"models")

  model_folders = os.listdir(models_path)

  for model_folder in model_folders:
    model_path = os.path.join(models_path,model_folder)
    sys.path.append(model_path)

  print("example not present!")
  # from CNN import SimpleCNN

  # from tensorflow.examples.tutorials.mnist import input_data
  # mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

  # from skimage.segmentation import mark_boundaries
  # import matplotlib.pyplot as plt

  # model_input_dim_height = 28
  # model_input_dim_width = 28 
  # model_input_channels = 1
  # n_classes = 10 

  # additional_args = {}

  # cnn_model = SimpleCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir ="mnist", additional_args = additional_args )

  # test_image = mnist.test.images[:1]
    
  # lime_explainer = LimeExplainer(cnn_model)

  # additional_args = {
  # "num_samples":1000,
  # "num_features":100,
  # "min_weight":0.01
  # }
  # explanation_image, explanation_text, predicted_class, additional_outputs = lime_explainer.Explain(test_image,additional_args)
  
  # # prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,labels=list(range(n_classes)),num_samples=10,top_labels=n_classes)
  # # prediction, explanation = lime_explainer.ClassifyWithLIME(test_image,num_samples=1000,labels=list(range(n_classes)), top_labels=n_classes)

  # predicted_class = np.argmax(prediction)
  # print("predicted_class",predicted_class)
  # print("mnist.test.labels[:1]",mnist.test.labels[:1])

  # print(explanation_text)
  # cv2.imshow("explanation",explanation_image)
  # 