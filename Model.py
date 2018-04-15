from keras.models import Sequential, Model, load_model as lm
from keras.layers import Dense, Activation, Conv2D, Input, Flatten, TimeDistributed, LSTM
from keras.optimizers import Adam, RMSprop
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
        elif model_type == 4:
            return self.model_LSTM_one()

    # Conv + dense + dense layer
    def model_one(self):
        state_input = Input(shape=(self.kframes, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(state_input)
        flatten = Flatten()(conv1)
        fc1 = Dense(128, activation='relu')(flatten)
        fc2 = Dense(self.available_actions_count, input_shape=(128,))(fc1)

        model = keras.models.Model(input=state_input, output=fc2)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return state_input, model

    # Conv + Conv + Conv + dense + dense
    def model_two(self):
        state_input = Input(shape=(self.kframes, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(state_input)
        conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', data_format="channels_first")(conv1)
        conv3 = Conv2D(filters=32, kernel_size=2, strides=1, activation='relu', data_format="channels_first")(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(128, activation='relu')(flatten)
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
        flatten = Flatten()(conv1)
        fc1 = Dense(256, activation='relu')(flatten)
        fc2 = Dense(128, activation='relu')(fc1)
        fc3 = Dense(self.available_actions_count, input_shape=(128,))(fc2)

        model = keras.models.Model(input=state_input, output=fc3)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())

        return state_input, model

    # Conv + dense + dense + dense + dense layer
    def model_LSTM_one(self):

        visual_state_input = Input((1, self.resolution[0], self.resolution[1]))
        conv1 = Conv2D(8, 6, strides=3, activation='relu', data_format="channels_first")(visual_state_input)
        # filters, kernal_size, stride
        # conv2 = Conv2D(8, 3, strides=2, activation='relu', data_format="channels_first")(
        #     conv1)  # filters, kernal_size, stride

        flatten = Flatten()(conv1)
        fc1 = Dense(128, activation='relu')(flatten)
        fc2 = Dense(64, activation='relu')(fc1)
        visual_model = keras.models.Model(input=visual_state_input, output=fc2)

        state_input = Input((self.kframes, 1, self.resolution[0], self.resolution[1]))
        td_layer = TimeDistributed(visual_model)(state_input)

        lstm_layer = LSTM(64)(td_layer)  # IF return sequences is true, it becomes many to many lstm
        fc3 = Dense(32, activation='relu')(lstm_layer)
        fc4 = Dense(self.available_actions_count)(fc3)

        model = keras.models.Model(input=state_input, output=fc4)
        das_optimizer = RMSprop(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=das_optimizer)
        print(model.summary())
        return state_input, model

