# Peter Arvidsson (250126)
# Sifferigenkänning och bildigenkänning kombinerat  

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ladda datasetet
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

# Dela upp och standardisera datasetet
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).reshape(-1, 28, 28, 1)
X_val_scaled = scaler.transform(X_val).reshape(-1, 28, 28, 1)
X_test_scaled = scaler.transform(X_test).reshape(-1, 28, 28, 1)

# Förbättrad modell
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dataaugmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
datagen.fit(X_train_scaled)

# Träna modellen
model.fit(datagen.flow(X_train_scaled, y_train, batch_size=64),
          validation_data=(X_val_scaled, y_val),
          epochs=20)

# Streamlit-applikation
st.title("Handskriven Sifferigenkänning")
st.write("Rita en siffra eller ladda upp en bild:")

uploaded_file = st.file_uploader("Ladda upp en bild", type=["png", "jpg", "jpeg"])
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    st.write(f"**Förutsagd siffra:** {np.argmax(predictions)}")
elif canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    image = ImageOps.invert(image.convert('L'))
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    st.write(f"**Förutsagd siffra:** {np.argmax(predictions)}")