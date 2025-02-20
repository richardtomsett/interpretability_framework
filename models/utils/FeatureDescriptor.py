import tensorflow as tf
from tensorflow.python import keras
import numpy as np
from PIL import Image


class FeatureDescriptor(object):
    """ Processes input images according to a pretrained network architecture
        For an example of pretrained network architectures see:
        https://github.com/tensorflow/tensorflow/tree/r1.10/tensorflow/python/keras/applications

        Arguments:
            input_shape: (image_x, image_y, n_channels) using channels-last style.
            batch_size: specify the batch size for processing. -1 means no batching
            architecture: specify a particular architecture.
            weights: 'None' for random, 'pretrainedt' for pretrained, or path for a custom.
            pooling: 'avg' or 'max'

    """
    def __init__(self, input_shape, batch_size=-1, architecture='vgg16', weights='pretrained', pooling='avg',
                 input_tensor=None, include_top=False):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.architecture = architecture
        self.pooling = pooling
        self.input_tensor = input_tensor
        self.include_top = include_top
        if weights == "pretrained":
            self.weights = "imagenet"  # keras' provided networks are trained with ImageNet
        else:
            self.weights = None
    def process_features(self, input_data):
        """
            Return an n-length feature vector based off input_data
        """

        # If input_data is a single image, add a dimension that defines the number of images as 1.
        if len(np.shape(input_data)) == 3:
            input_data = np.expand_dims(input_data, axis=0)


        with tf.Session().as_default():
            model = self.__load_architecture(self.architecture)
            
            return model.predict(input_data)
       
    def get_descriptor_op(self, ):
        """
            Return the model prediction op for a placeholder passed as a 
            parameter
        """
        
        model = self.__load_architecture(self.architecture)
        for layer in model.layers:
            layer.trainable = False
        return model.layers[-1].output

    def get_premade_model(self):
        return self.__load_architecture(self.architecture)

    def __load_architecture(self, architecture_name):
        """ Loads a particular architecture from keras.applications """

        # Optional arguments that are handy to have for the descriptor.
        model_args = {
            'include_top': self.include_top,
            'weights': self.weights,
            'input_shape': self.input_shape,
            'pooling': self.pooling
        }

        if self.input_tensor is not None:
            model_args['input_tensor'] = self.input_tensor

        return {
            'vgg16': lambda: keras.applications.VGG16(**model_args),
            'vgg19': lambda: keras.applications.VGG19(**model_args),
            'resnet50': lambda: keras.applications.ResNet50(**model_args),
            'xception': lambda: keras.applications.Xception(**model_args),
            'inceptionv2': lambda : keras.applications.InceptionResNetV2(**model_args),
            'inceptionv3': lambda : keras.applications.InceptionV3(**model_args),
            'densenet': lambda: keras.applications.DenseNet201(**model_args),
            'nasnet': lambda: keras.applications.NASNetLarge(**model_args),
            'nasnetmobile': lambda: keras.applications.NASNetMobile(**model_args),
            'mobile': lambda: keras.applications.MobileNet(**model_args),
        }[architecture_name]()


"""
    Test to ensure it's working - download an image and call it "test.jpg"
"""
if __name__ == "__main__":
    
    img = np.array(Image.open("test.jpg")) # an arbitrary image 
    d = FeatureDescriptor(np.shape(img))

    features = d.process_features(img)
    print("Feature length: " + str(np.shape(features)[1]))
    print("Features: ")
    print(features)




