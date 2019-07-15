import sys
innvestigate_path = "../innvestigate/"
sys.path.append(innvestigate_path)
from innvestigate_explanation import *

class DeepTaylorExplainer(InnvestigateExplainer):
  """docstring for DeepTaylorExplainer"""
  def __init__(self, model):
    super(DeepTaylorExplainer, self).__init__(model, explanation_method="deep_taylor")
  
  def DeepTaylorAttributionToImage(self, deep_attribution):
    return super(DeepTaylorExplainer, self).InnvestigateAttributionToImage(deep_attribution)

  def GenerateDeepTaylorExplanationImage(self,input_image,explanation_values):
      return super(DeepTaylorExplainer, self).GenerateInnvestigateExplanationImage(input_image,explanation_values)

    
  def Explain(self,input_image, additional_args = {}):
    return super(DeepTaylorExplainer, self).Explain(input_image, additional_args=additional_args)


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