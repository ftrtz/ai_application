import datetime
import os.path
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.datasets.mnist import load_data
from ai_library import CustomData, NeuralNet, Visualisation

# Create a a child class of CustomData to load MNIST
class MNISTdata(CustomData):
    def __init__(self):
        super(MNISTdata, self).__init__()

    def load_mnist(self):
        return load_data()

# parameters
EPOCHS = 30
BATCH_SIZE = 128

# load MNIST data
data_gen = MNISTdata()
(x_train, y_train), (x_test, y_test) = data_gen.load_mnist()

# preprocessing
x_train, y_train, labels_train = data_gen.preprocessing(x_train, y_train)
x_test, y_test, labels_test = data_gen.preprocessing(x_test, y_test)

# number of classes
num_classes = len(labels_train)

# initialize model
model = NeuralNet(num_classes)

# callbacks
# early stopping
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# tensorboard
logtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "results/logs/fit/" + logtime
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [early_stopping, tensorboard_cb]

# train model
history = model.comp_and_fit(x_train, y_train, EPOCHS, BATCH_SIZE, callbacks)

# save trained model
model.save("results/model")

# model evaluation
loss, accuracy = model.test_eval(x_test, y_test)

# predictions
y_preds = model.test_pred(x_test)


# SAVE RESULTS
# save loss and accuracy in csv
d = {"logtime": logtime, "loss": loss, "accuracy": accuracy}
eval_df = pd.DataFrame(data=d, index=[0])

if os.path.isfile("results/loss_acc.csv"):
    header = False
else:
    header = True

eval_df.to_csv("results/loss_acc.csv", mode="a", index=False, index_label=False, header=header)

# figures
viz = Visualisation()
# confusion matrix
cm = viz.cm_plot(y_test, y_preds, labels_test)
# image samples
sample_img = viz.img_plot("Sample Images and corresponding labels", x_train, y_train)
sample_img_pred = viz.img_plot("Sample images and predicted labels", x_test, y_preds)

# save figures
cm.savefig("results/cm.png")
sample_img.savefig("results/sample_img.png")
sample_img_pred.savefig("results/sample_img_pred.png")


