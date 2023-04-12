from convertXMLToVector import xmlLoader
from CNN_VAE import cnn_vae
from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())
print("---------------------")
print(tf.__version__)

print("---------------------")


c = xmlLoader("data/")
x_all, y_all = c.convert("C", "major")

print(y_all.shape[2])
v = cnn_vae(x_all, y_all)
vae = v.make_model()
vae.fit(x_all, x_all, epochs = 500)
vae.save("saves/cnn/model.h5")


