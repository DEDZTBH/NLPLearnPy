import tensorflow as tf
import numpy as np
from bilstm_crf.extract import get_my_data

X_train, X_test, y_train, y_test = get_my_data()

tf.enable_eager_execution()
sess = tf.InteractiveSession()

www = np.array(list(X_train[1]))
dataset = tf.data.Dataset.from_tensor_slices(www)
www2 = [str(a.numpy(), encoding='utf-8') for a in dataset]
print(www2)
