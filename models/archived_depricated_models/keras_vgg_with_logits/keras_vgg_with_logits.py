import numpy as np

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class KerasVGGlogits(object):
    """A simple CNN model implemented using the Tensorflow estimator API"""

    def __init__(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes, model_dir,
                 additional_args={}):
        super(KerasVGGlogits, self).__init__()
        self.model_input_dim_height = model_input_dim_height
        self.model_input_dim_width = model_input_dim_width
        self.model_input_channels = model_input_channels
        self.n_classes = n_classes
        self.model_dir = model_dir

        # model specific variables
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

    ### Required Model Functions
    def InitaliseModel(self, model_dir="model_dir"):
        opts = tf.GPUOptions(allow_growth=True)
        conf = tf.ConfigProto(gpu_options=opts)
        # trainingConfig = tf.estimator.RunConfig(session_config=conf)
        set_session(tf.Session(config=conf))

        self.model = self.BuildModel(self.model_input_dim_height, self.model_input_dim_width, self.model_input_channels, self.n_classes,self.dropout)
        

    def TrainModel(self, train_x, train_y, batch_size, num_steps, val_x= None, val_y=None):
        if (type(train_x) != dict):
            input_dict = {"input": train_x}
        else:
            input_dict = train_x

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=self.learning_rate),
              metrics=['accuracy'])

        if(val_x is not None and val_y is not None):
            self.model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=num_steps,
              verbose=1,
              validation_data=(val_x, val_y))
        else:
            self.model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=num_steps,
          verbose=1)


    def EvaluateModel(self, eval_x, eval_y, batch_size):
        if (type(eval_x) != dict):
            input_dict = {"input": eval_x}
        else:
            input_dict = eval_x

        # Train the Model
        return self.model.evaluate(eval_x,eval_y, batch_size=batch_size)


    def Predict(self, predict_x):
        if (type(predict_x) != dict):
            input_dict = {"input": predict_x}
        else:
            input_dict = predict_x

        
        predictions = self.model.predict(predict_x)
        print("[np.argmax(prediction) for prediction in predictions]",[np.argmax(prediction) for prediction in predictions])
        return [np.argmax(prediction) for prediction in predictions]


    def SaveModel(self, save_dir):
        model_json = self.model.to_json()
        with open(save_dir, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(save_dir+".h5")

        print("Saved model to:"+ str(self.model_dir))


    def LoadModel(self, load_dir):
        if(load_dir[-3:] == ".h5"):
            load_h5_path = load_dir 
        else:
            load_h5_path = load_dir+".h5"

            self.model.load_weights(load_h5_path)
        
        print("Loaded model from:"+ str(self.model_dir))

    ### Model Specific Functions
    def BuildModel(self, model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,dropout):
        model = Sequential()
        
        model.add(Reshape([model_input_dim_height, model_input_dim_width, model_input_channels],name="absolute_input"))
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=[model_input_dim_height, model_input_dim_width, model_input_channels], name="conv_1"))
        model.add(Conv2D(32, (3, 3), activation='relu', name="conv_2"))
        model.add(MaxPooling2D(pool_size=(2, 2),name="max_pool_1"))
        model.add(Dropout(dropout,name="dropout_1"))

        model.add(Conv2D(64, (3, 3), activation='relu', name="conv_3"))
        model.add(Conv2D(64, (3, 3), activation='relu', name="conv_4"))
        model.add(MaxPooling2D(pool_size=(2, 2),name="max_pool_2"))
        model.add(Dropout(dropout,name="dropout_2"))

        model.add(Flatten(name="feature_vector_1"))
        model.add(Dense(256, activation='relu',name="fully_connected_1"))
        model.add(Dropout(dropout,name="dropout_3"))
        model.add(Dense(n_classes, name="logits"))
        model.add(Activation('softmax',name="absolute_output"))
        return model
        

    
    def GetLayerByName(self,name):
        print("GetLayerByName - not implemented")
    
    def FetchAllVariableValues(self):
        print("FetchAllVariableValues - not implemented")


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    model_input_dim_height = 28
    model_input_dim_width = 28
    model_input_channels = 1
    n_classes = 10
    learning_rate = 0.001

    batch_size = 128
    num_train_steps = 200

    additional_args = {"learning_rate": learning_rate}

    cnn_model = KerasCNN(model_input_dim_height, model_input_dim_width, model_input_channels, n_classes,
                          model_dir="mnist", additional_args=additional_args)

    verbose_every = 10
    for step in range(verbose_every, num_train_steps + 1, verbose_every):
        print("")
        print("training")
        print("step:", step)
        cnn_model.TrainModel(mnist.train.images, mnist.train.labels, batch_size, verbose_every)
        print("")

        print("evaluation")
        print(cnn_model.EvaluateModel(mnist.test.images[:128], mnist.test.labels[:128], batch_size))
        print("")

    print(cnn_model.Predict(mnist.test.images[:5]))

    print(mnist.test.labels[:5])
