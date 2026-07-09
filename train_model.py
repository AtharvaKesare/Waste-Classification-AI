import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------
# PATHS (VERY IMPORTANT)
# ---------------------------
train_dir = "Resources/Dataset/Train"
test_dir = "Resources/Dataset/Test"

# ---------------------------
# IMAGE SETTINGS
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ---------------------------
# DATA LOADERS
# ---------------------------
train_gen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

test_gen = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ---------------------------
# MODEL (MOBILENET)
# ---------------------------
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(224,224,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# TRAINING
# ---------------------------
model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# ---------------------------
# SAVE MODEL TO YOUR FLASK PATH
# ---------------------------
model.save("Resources/Model/final_model_weights.hdf5")

print("\nMODEL TRAINED AND SAVED SUCCESSFULLY!!!\n")
