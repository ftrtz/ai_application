from tensorflow.keras.datasets.mnist import load_data

from ai_library import CustomData

#%%

class MNISTdata(CustomData):
    def __init__(self):
        super(MNISTdata, self).__init__()

    def load_mnist(self):
        return load_data()
