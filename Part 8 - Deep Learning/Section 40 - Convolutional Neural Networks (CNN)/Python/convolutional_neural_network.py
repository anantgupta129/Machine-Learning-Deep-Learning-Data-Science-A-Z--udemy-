# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # CONVOLUTiONAL NEURAL NETWORK

# %%
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# %%
tf.__version__

# %% [markdown]
# ## P1 - Data Preprocessing
# %% [markdown]
# ### Creating Training set

# %%
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')

# %% [markdown]
# ### Creating Test Set

# %%
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')

# %% [markdown]
# ## P2 - Building the CNN
# %% [markdown]
# ### Step 1 - Convolution

# %%
"""filters are numbers of features, kernel_size refer to size of feature detector matrix, input_shape 3 is for colored and 64, 64 we chose above in data preprocessing"""
cnn = tf.keras.models.Sequential()      #Initialising the CNN
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# %% [markdown]
# ### Step 2 - Pooling

# %%
""" pool_size is size of pool martrix, strides is how we move like in example we moved sidewise by leaving 2 pixel"""
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# %% [markdown]
# ### Adding a second convolutional layer

# %%
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# %% [markdown]
# ### Step 3 - Flattening

# %%
cnn.add(tf.keras.layers.Flatten())

# %% [markdown]
# ### Step 4 - Full Connection

# %%
""" 128 are number of neurons"""
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# %% [markdown]
# ### Step 5 - Output Layer

# %%
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# %% [markdown]
# ## P3 - Training the CNN
# %% [markdown]
# ### Compiling the CNN

# %%
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% [markdown]
# ### Training the CNN on the Training set and evaluating it on the Test set

# %%
cnn.fit(x = train_set, validation_data= test_set, epochs=25)

# %% [markdown]
# ## P4 - Making a single prediction

# %%
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
res = cnn.predict(test_image)
train_set.class_indices
if res[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)


# %%



