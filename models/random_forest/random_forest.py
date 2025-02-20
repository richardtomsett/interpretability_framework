import tensorflow as tf
import os
import sys
from models.utils.ConvFeatureDescriptor import ConvFeatureDescriptor


class RandomForest(object):

    def __init__(self, x_dim, y_dim, n_channels, n_classes, model_dir, n_trees=10, max_nodes=1000):
        self.x_dim = x_dim  # get dims from dataset json
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = self.x_dim * self.y_dim
        self.model_dir = model_dir
        self.n_trees = n_trees
        self.max_nodes = max_nodes
        self.model = None

    def InitialiseModel(self, model_dir="model_dir"):
        params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
            num_classes=self.n_classes,
            num_features=self.n_features,
            regression=False,  # We are strictly doing classification problems.
            num_trees=self.n_trees,
            max_nodes=self.max_nodes
        )

        self.model = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)
        self.descriptor = ConvFeatureDescriptor(batch_size=128, x_dim=self.x_dim, y_dim=self.y_dim)

    def TrainModel(self, x_train, y_train, batch_size, num_steps):

        descriptor = ConvFeatureDescriptor(batch_size=batch_size, x_dim=self.x_dim, y_dim=self.y_dim)
        x_train = descriptor.get_feature_vectors(x_train)  # Apply VGG16 feature detection to input data

        self.model.fit(x=x_train, y=y_train)

    def EvaluateModel(self, x_eval, y_eval, batch_size):

        descriptor = ConvFeatureDescriptor(batch_size=batch_size, x_dim=self.x_dim, y_dim=self.y_dim)
        x_eval = descriptor.get_feature_vectors(x_eval)  # Apply VGG16 feature detection to input data

        return self.model.evaluate(x=x_eval, y=y_eval)

    def Predict(self, x_predict):

        return self.model.predict(x=x_predict)

#
# if __name__ == "__main__":
#
#     base_path = os.getcwd().split("/")
#     while base_path[-1] != str("p5_afm_2018_demo"):
#         base_path = base_path[:-1]
#     sys.path.append(os.path.join("/".join(base_path) + "/models"))