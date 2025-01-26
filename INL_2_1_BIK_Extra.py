# Peter Arvidsson (20250126)
# Bildigenkänning V1, enklare version

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from PIL import Image
import numpy as np

# Ladda förtränad modell
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

def predict_image(image):
    # Resiza bilden till 224x224 (krävs för MobileNetV2)
    image = image.resize((224, 224))
    
    # Konvertera till numpy-array
    image_array = np.asarray(image)
    
    # Lägg till batch-dimension och förbearbeta bilden
    image_array = np.expand_dims(image_array, axis=0)  # (1, 224, 224, 3)
    image_array = preprocess_input(image_array)  # Normalisera för modellen
    
    # Gör förutsägelser
    predictions = model.predict(image_array)
    
    # Dekoda resultaten
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # De tre bästa resultaten
    return decoded_predictions

# Streamlit-applikation

st.markdown("<h1 style='font-size: 48px;'>Peters bildigenkänningsprogram</h1>", unsafe_allow_html=True)
st.subheader("Framtiden är här!")
st.write("Ladda upp en bild så identifierar modellen vad den föreställer.")

# Ladda upp bild
uploaded_file = st.file_uploader("Ladda upp en bild i PNG- eller JPG-format", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Öppna och visa bilden
        image = Image.open(uploaded_file)
        st.image(image, caption="Uppladdad bild", use_column_width=True)
        
        # Gör en förutsägelse
        predictions = predict_image(image)
        
        # Visa resultaten
        st.write("### Identifierade objekt:")
        for pred in predictions:
            st.write(f"**{pred[1]}**: {pred[2]*100:.2f}% sannolikhet")
    
    except Exception as e:
        st.error(f"Ett fel inträffade vid bearbetning av bilden: {e}")
