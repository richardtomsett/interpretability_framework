import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout, Activation, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.models import Sequential
from keras.models import model_from_json
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D


class VGG16RichardNoBias(object):
    """VGG16 CNN feature descriptor (fully trained) with re-trained fully connected layers"""

    def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir,
                 additional_args={}):
        super(VGG16RichardNoBias, self).__init__()
        # model specific variables
        self.min_height = 32
        self.min_width = 32

        self.weight_decay = 1e-4
        
        self.model_input_dim_height = max(model_input_dim_height,self.min_height)
        self.model_input_dim_width = max(model_input_dim_width,self.min_width)
        self.model_input_channels = model_input_channels
        self.n_classes = n_classes
        self.model_dir = model_dir

        
        # Training Parameters

        if ("learning_rate" in additional_args):
            self.learning_rate = additional_args["learning_rate"]
        else:
            self.learning_rate = 0.001
            print("using default of " + str(self.learning_rate) + " for " + "learning_rate")

        if ("dropout" in additional_args):
            self.dropout = additional_args["dropout"]
        else:
            self.dropout = 0.5
            print("using default of " + str(self.dropout) + " for " + "dropout")

        self.model = None
        self.InitaliseModel(model_dir=self.model_dir)

        self.sess = keras.backend.get_session()

        self.input_ = self.model.layers[0].input
        self.labels_ = tf.placeholder(tf.float32, shape = [None, n_classes])
        self.logits = self.model.layers[-1].output

        self.loss = keras.losses.categorical_crossentropy(self.labels_, self.logits)


    ### Required Model Functions
    def InitaliseModel(self, model_dir="saved_models"):
        opts = tf.GPUOptions(allow_growth=True)
        conf = tf.ConfigProto(gpu_options=opts)
        # trainingConfig = tf.estimator.RunConfig(session_config=conf)
        set_session(tf.Session(config=conf))

        self.model = self.BuildModel(self.model_input_dim_height, self.model_input_dim_width, self.model_input_channels, self.n_classes,self.dropout)
        

    def TrainModel(self, train_x, train_y, batch_size, num_steps, val_x= None, val_y=None, early_stop=True, save_best_name=""):
        train_x = self.CheckInputArrayAndResize(train_x,self.min_height,self.min_width)
        if(val_x is not None):
            val_x = self.CheckInputArrayAndResize(val_x,self.min_height,self.min_width)


        if (type(train_x) != dict):
            input_dict = {"input": train_x}
        else:
            input_dict = train_x

        callbacks=[]
        if(early_stop):
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
            callbacks.append(es)

        if(save_best_name != ""):    
            mc = ModelCheckpoint(save_best_name+'.h5', monitor='val_loss', mode='min', save_best_only=True)
            callbacks.append(mc)
            

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=self.learning_rate),
              metrics=['accuracy'])

        if(val_x is not None and val_y is not None):
            self.model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=num_steps,
              verbose=1,
              validation_data=(val_x, val_y),callbacks=callbacks)
        else:
            self.model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=num_steps,
          verbose=1,callbacks=callbacks)


    def EvaluateModel(self, eval_x, eval_y, batch_size):
        eval_x = self.CheckInputArrayAndResize(eval_x,self.min_height,self.min_width)
        
        if (type(eval_x) != dict):
            input_dict = {"input": eval_x}
        else:
            input_dict = eval_x

        # Train the Model
        return self.model.evaluate(eval_x,eval_y, batch_size=batch_size)


    def Predict(self, predict_x, return_prediction_scores = False):
        predict_x = self.CheckInputArrayAndResize(predict_x,self.min_height,self.min_width)
        
        if (type(predict_x) != dict):
            input_dict = {"input": predict_x}
        else:
            input_dict = predict_x

        
        predictions = self.model.predict(predict_x)
        
        if(return_prediction_scores):
            return predictions, [np.argmax(prediction) for prediction in predictions]
        else:
            return [np.argmax(prediction) for prediction in predictions]


    def SaveModel(self, save_dir):
        model_json = self.model.to_json()
        with open(save_dir, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(save_dir+".h5")

        print("Saved model to:"+ str(save_dir+".h5"))


    def LoadModel(self, load_dir):
        if(load_dir[-3:] == ".h5"):
            load_h5_path = load_dir 
        else:
            load_h5_path = load_dir+".h5"

            self.model.load_weights(load_h5_path)
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=self.learning_rate),
              metrics=['accuracy'])
        
        print("Loaded model from:"+ str(load_h5_path))

    ### Model Specific Functions
    def BuildModel(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,dropout): 
        
        self.x_shape = [model_input_dim_height,model_input_dim_width,model_input_channels]

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(32, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            input_shape=self.x_shape, \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            input_shape=self.x_shape, \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            input_shape=self.x_shape, \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), \
            kernel_regularizer=regularizers.l2(weight_decay), \
            padding='same', \
            use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(n_classes, \
            use_bias=False))
        model.add(Activation('softmax'))
        return model

    def GetWeights(self):
        return [w for w in self.model.trainable_weights if 'kernel' in w.name]

    def GetPlaceholders(self):
        return [self.input_, self.labels_]

    def GetGradLoss(self):
        return tf.gradients(self.loss, self.GetWeights())
    
    def GetLayerByName(self,name):
        print("GetLayerByName - not implemented")
    
    def FetchAllVariableValues(self):
        print("FetchAllVariableValues - not implemented")


    def CheckInputDimensions(self,input_shape,min_height,min_width):
        if(len(input_shape) == 4):
            image_shape = input_shape[1:]
        else:
            image_shape = input_shape

        return (max(min_height,image_shape[0]),max(min_width,image_shape[1]),image_shape[2])


    def CheckInputArrayAndResize(self,image_array,min_height,min_width):
        image_array_shape = image_array.shape

        if(len(image_array_shape) == 4):
            image_shape = image_array_shape[1:]
        else:
            image_shape = image_array_shape

        target_shape = (max(min_height,image_shape[0]),max(min_width,image_shape[1]),image_shape[2])

        shape_difference = (np.array(target_shape) - np.array(image_shape))

        add_top = int(shape_difference[0]/2)
        add_bottom = shape_difference[0] - add_top

        add_left = int(shape_difference[1]/2)
        add_right = shape_difference[1] - add_left

        print("In CheckInputArrayAndResize:")
        print("Changing size by "+str(add_top)+", "+str(add_left))

        return np.pad(image_array,((0,0),(add_top,add_bottom),(add_left,add_right),(0,0)), mode='constant', constant_values=0)

