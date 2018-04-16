from keras.models import Sequential, Model, load_model as lm
from keras.layers import Dropout, Dense, Activation, Conv2D, Input, Flatten, MaxPooling2D
from keras.optimizers import Adam
import keras


class NeuralModel:
    def __init__(self, available_actions_count, learning_rate, k_frames):
        self.resolution = [30, 45]
        self.kframes = k_frames
        self.available_actions_count = available_actions_count
        self.learning_rate = learning_rate

    def create_model(self, model_type):
        if model_type == 1:
            return self.model_one()
        elif model_type == 2:
            return self.model_two()
        elif model_type == 3:
            return self.model_three()

    # Conv + dense + dense layer
    def model_one(self):
        state_input = Input(shape=(self.kframes, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(state_input)
        pool=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv1)
        flatten = Flatten()(pool)
        dr = Dropout(0.2, input_shape=(128,))(flatten)
        fc1 = Dense(128, activation='relu')(flatten)
        fc2 = Dense(self.available_actions_count, input_shape=(128,))(dr)
        model = keras.models.Model(input=state_input, output=fc2)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return state_input, model

    # Conv + Conv + Conv + dense + dense
    def model_two(self):
        state_input = Input(shape=(self.kframes, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(state_input)
        pool1=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv1)
        conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(pool1)
        pool2=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv2)
        conv3 = Conv2D(filters=32, kernel_size=2, strides=1, activation='relu', data_format="channels_first")(pool2)
        pool3=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv3)
        flatten = Flatten()(pool3)
        dr = Dropout(0.2, input_shape=(128,))(flatten)
        fc1 = Dense(128, activation='relu')(dr)
        fc2 = Dense(self.available_actions_count, input_shape=(128,))(fc1)

        model = keras.models.Model(input=state_input, output=fc2)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return state_input, model

    # Conv + dense + dense + dense + dense layer
    def model_three(self):
        state_input = Input(shape=(self.kframes, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(state_input)
        pool1 = Conv2D(filters=32, kernel_size=2, strides=1, activation='relu', data_format="channels_first")(conv1)
        flatten = Flatten()(pool1)
        dr = Dropout(0.2, input_shape=(256,))(flatten)
        fc1 = Dense(256, activation='relu')(dr)
        fc2 = Dense(128, activation='relu')(fc1)
        fc3 = Dense(self.available_actions_count, input_shape=(128,))(fc2)

        model = keras.models.Model(input=state_input, output=fc3)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return state_input, model
