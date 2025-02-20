import numpy as np

from skimage.color import gray2rgb, rgb2gray

from skimage.segmentation import mark_boundaries

import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
import sys
import json

from deepexplain.tensorflow import DeepExplain
from keras import backend as K
from keras.models import Model

class LRPExplainer(object):
  """docstring for LRPExplainer"""
  def __init__(self, model):
    super(LRPExplainer, self).__init__()
    self.model = model

    self.requires_fresh_session = True
		
  def GetColorMap(self):
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((0./255, 0./255, (129.+l*100)/255,l))
    for l in np.linspace(0, 1, 100):
        colors.append(((155.+l*100)/255, 0./255, 0./255,l))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    return red_transparent_blue
  
  def DeepExplainAttributionToImage(self, deep_attribution):
    img = np.sum(deep_attribution, axis=2)
    
    return img*255

  def GenerateLRPExplanationImage(self,input_image,explanation_values):
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
    
    prediction_scores,prediction = self.model.Predict(input_image, True)
    predicted_class = np.argmax(prediction_scores)
    #print("explanation prediction output",prediction)
    #print("explanation predicted_class",predicted_class)

    ####LRP
    prediction_mask = [0]* self.model.n_classes
    prediction_mask[predicted_class] = 1
    prediction_mask = np.array(prediction_mask)

    #print(prediction_mask)
    
    try:
      input_image = self.model.CheckInputArrayAndResize(input_image,self.model.min_height,self.model.min_width)
    except:
      print("couldn't use model image size check")    
    
    attributions = None

    with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
      # Need to reconstruct the graph in DeepExplain context, using the same weights.
      # With Keras this is very easy:
      # 1. Get the input tensor to the original model
      input_tensor = self.model.model.layers[0].input
      
      # 2. We now target the output of the last dense layer (pre-softmax)
      # To do so, create a new model sharing the same layers untill the last dense (index -2)
      fModel = Model(inputs=input_tensor, outputs = self.model.model.layers[-2].output)
      target_tensor = fModel(input_tensor)
      

      # attributions = de.explain('grad*input', target_tensor * ys, input_tensor, xs)
      # attributions = de.explain('saliency', target_tensor * ys, input_tensor, xs)
      #attributions = de.explain('intgrad', target_tensor * ys, input_tensor, xs)
      #attributions = de.explain('deeplift', target_tensor * ys, input_tensor, xs)
      attributions = de.explain('elrp', target_tensor , input_tensor, input_image)
      #attributions = de.explain('occlusion', target_tensor * ys, input_tensor, xs)


    lrp_explanation = self.DeepExplainAttributionToImage(attributions[0])

    
    
    
    explanation_image = self.GenerateLRPExplanationImage(input_image,lrp_explanation[:])
    explanation_image = cv2.cvtColor(explanation_image, cv2.COLOR_RGB2BGR)

    # cv2_image = cv2.cvtColor(explanation_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("explanation_image LRP",cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ## for testing:
    # shap.image_plot(shap_values, np.multiply(input_image,255.0))

    if(not isinstance(prediction_scores,list)):
      prediction_scores = prediction_scores.tolist()
    
    attributions_list = attributions.tolist()
    attributions_list = attributions_list[0]
        
    additional_outputs = {"attribution_map":attributions_list, "lrp_values":[lrp_value.tolist() for lrp_value in attributions],"prediction_scores":prediction_scores[0]}

    explanation_text = "Evidence towards predicted class shown in blue, evidence against shown in red."
    
    return explanation_image, explanation_text, predicted_class, additional_outputs
  


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