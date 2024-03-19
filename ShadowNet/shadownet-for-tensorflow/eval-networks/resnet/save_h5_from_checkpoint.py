from resnet import ResNetBlock
import tensorflow as tf
from IPython import embed

from resnet import resnet_v2

from tensorflow.keras.datasets import cifar10
def main():
    embed()
    latest = tf.train.latest_checkpoint("./training_2")
    (train_imgs, _), (_, _) = cifar10.load_data()
    input_shape = train_imgs.shape[1:]
    model = resnet_v2(input_shape=input_shape, depth=44)
    model.load_weights(latest)
     
    print(model.summary())
    model.save("resnet-new-44.h5")
if __name__ == '__main__':
    main() 
