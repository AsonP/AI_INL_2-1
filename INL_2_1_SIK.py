# Peter Arvidsson (250125)
# Sifferigenkänning handritade siffror
# Sparar tränad modell "mnist_cnn_model.keras"
# Labbat med värdena epochs=100 och batch_size=128
# Predikteringen verkar inte bli bättre
 
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# Försök att hämta datasetet från webben
try:
    print("Försöker hämta datasetet från OpenML...")
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)
    print("Datasetet laddades från webben.")
except Exception as e:
    print("Kunde inte hämta datasetet från webben. Försöker ladda från lokal fil...")
    try:
        mnist_path = 'mnist_784.pkl'
        mnist = joblib.load(mnist_path)
        X = mnist["data"]
        y = mnist["target"].astype(np.uint8)
        print("Datasetet laddades från lokal fil.")
    except FileNotFoundError:
        print("Lokal fil 'mnist_784.pkl' hittades inte. Kontrollera att filen finns eller internetanslutningen fungerar.")
    except Exception as e:
        print(f"Ett fel inträffade vid laddning av datasetet från lokal fil: {e}")

# Dela upp datasetet
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardisera data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Omforma data för CNN
X_train_scaled = X_train_scaled.reshape(-1, 28, 28, 1)
X_val_scaled = X_val_scaled.reshape(-1, 28, 28, 1)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28, 1)

def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Ladda eller träna modellen
try:
    model = tf.keras.models.load_model("mnist_cnn_model.keras")
except:
    model = create_cnn_model()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Dataaugmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    datagen.fit(X_train_scaled)

    # Callbacks
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5)
    custom_lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.95 ** epoch, verbose=1)

    # Träna modellen med augmentation
    model.fit(datagen.flow(X_train_scaled, y_train, batch_size=128),
              validation_data=(X_val_scaled, y_val),
              epochs=100,
              callbacks=[early_stopping, lr_scheduler, custom_lr_scheduler])

    tf.keras.models.save_model(model, "mnist_cnn_model.keras")

# Utvärdera modellen
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Testnoggrannhet: {test_acc * 100:.2f}%")

# Funktion för att centrera bilden
def center_image(img_array):
    cy, cx = center_of_mass(img_array)
    rows, cols = img_array.shape
    shift_y, shift_x = np.round(rows / 2 - cy).astype(int), np.round(cols / 2 - cx).astype(int)
    img_array = np.roll(img_array, shift_y, axis=0)
    img_array = np.roll(img_array, shift_x, axis=1)
    return img_array

# Förbehandla bilden
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = center_image(img_array)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit-applikation
st.title("Handskriven Sifferigenkänning med CNN")
st.write("Rita en siffra (0-9) i rutan nedan:")

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

if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA')
    image = image.convert('L')
    image = ImageOps.invert(image)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.write(f"**Förutsagd siffra:** {predicted_class}")
    st.write(f"**Trovärdighet:** {confidence:.2f}%")

    st.write("**Sannolikheter för varje siffra:**")
    for i, prob in enumerate(predictions[0]):
        st.write(f"Siffra {i}: {prob * 100:.2f}%")