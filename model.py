from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow.keras

class ClassificationModel:
    def __init__(self, input_shape, x_train, y_train):
        self.input_shape = input_shape
        self.x_train = x_train
        self.y_train = y_train
        self.model = None  # Initialize the model as None
        
    def build_model(self):
        inpx = Input(shape=self.input_shape)
        layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
        layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
        layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
        layer4 = Dropout(0.5)(layer3)
        layer5 = Flatten()(layer4)
        layer6 = Dense(250, activation='sigmoid')(layer5)
        layer7 = Dense(10, activation='softmax')(layer6)

        # Create the model and assign it to the instance
        self.model = Model(inputs=[inpx], outputs=layer7)

    def compile_model(self):
        if self.model is None:
            raise ValueError("The model has not been built yet. Call build_model() first.")
        
        self.model.compile(optimizer=tensorflow.keras.optimizers.Adadelta(),
                           loss=tensorflow.keras.losses.categorical_crossentropy,
                           metrics=['accuracy'])

        # epochs and batch sizes are hardcoded for now. 
    def train_model(self, epochs=12, batch_size=64):
        if self.model is None:
            raise ValueError("The model has not been built yet. Call build_model() first.")
        
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)