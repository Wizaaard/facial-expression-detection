# import necessary packages 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize datta generators
train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# load data as grayscale and resize
train_data = train_data_generator.flow_from_directory(
    'data/train',
    target_size = (48,48),
    batch_size = 16,
    color_mode= 'grayscale',
    class_mode = 'categorical'
)
test_data = test_data_generator.flow_from_directory(
    'data/test',
    target_size = (48,48),
    batch_size = 16,
    color_mode= 'grayscale',
    class_mode = 'categorical'
)

# machine learning model
emotion_model = Sequential()

