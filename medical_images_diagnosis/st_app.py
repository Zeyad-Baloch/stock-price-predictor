import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import cv2


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('final_x-ray_model.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:

        if not model.built:
            model.build(input_shape=(None, 224, 224, 3))

        base_model = model.layers[0]
        try:
            conv_layer = base_model.get_layer(last_conv_layer_name)
        except ValueError:
            st.error(f"Layer '{last_conv_layer_name}' not found.")
            available_layers = [layer.name for layer in base_model.layers if 'mixed' in layer.name]
            st.write("Available layers:", available_layers)
            return None

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output]
        )

        remaining_layers = model.layers[1:]

        with tf.GradientTape() as tape:

            conv_outputs, base_output = grad_model(img_array)

            x = base_output
            for layer in remaining_layers:
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            class_channel = predictions[:, pred_index]


        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-7)
        return heatmap.numpy()

    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None



def preprocess_img(img):

    img_array = np.array(img)


    if len(img_array.shape) == 3 and img_array.shape[2] == 3:

        img_array = cv2.resize(img_array, [224, 224])
    else:

        img_array = cv2.resize(img_array, [224, 224])


    img_array = img_array.astype(np.float32) / 255.0


    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def create_gradcam_visualization(img_array, heatmap, original_img):
    try:

        heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], [224, 224])
        heatmap_resized = tf.squeeze(heatmap_resized)


        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_uint8]


        jet_heatmap_pil = Image.fromarray(np.uint8(jet_heatmap * 255))


        original_array = np.array(original_img.resize((224, 224))) / 255.0
        jet_heatmap_array = np.array(jet_heatmap_pil) / 255.0

        superimposed = jet_heatmap_array * 0.4 + original_array * 0.6
        superimposed_img = Image.fromarray(np.uint8(superimposed * 255))

        return superimposed_img

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


#Streamlit Layout
st.set_page_config(page_title="X-ray Diagnosis")


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, div, p, span, input, textarea, label,
    h1, h2, h3, h4, h5, h6,
    .stTextInput, .stSelectbox, .stButton, .stMarkdown,
    .css-1d391kg, .css-1v0mbdj, .css-1kyxreq, .css-q8sbsg {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Style columns to have card appearance */
div[data-testid="stColumn"] > div {
    background-color: #ecebe3;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #bb5a38;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 0.5rem;
    min-height: 300px;
}

/* Center align images */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}

/* Style expander container */
div[data-testid="stExpander"] {
    background-color: #ecebe3;
    border-radius: 10px;
    border: 1px solid #bb5a38;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
    padding: 1rem;
}

/* Remove extra styling from nested elements inside expander */
div[data-testid="stExpander"] div[data-testid="stColumn"] > div {
    background: none;
    border: none;
    box-shadow: none;
    margin: 0;
    padding: 0;
    min-height: auto;
}

/* Style file uploader */
div[data-testid="stFileUploader"] {
    background-color: #ecebe3;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #bb5a38;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

/* Remove default streamlit padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Prevent empty columns from having card styling */
div[data-testid="stColumn"] > div:empty {
    background: none;
    border: none;
    box-shadow: none;
    margin: 0;
    padding: 0;
    min-height: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("Medical Images Diagnosis")
st.markdown("Upload an X-ray to receive a diagnosis along with a Grad-CAM explanation.")



# Load model
model = load_model()

if model is None:
    st.error("Could not load the model. Please ensure 'best_model.keras' is in the same directory.")
    st.stop()

uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:

        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:

        with st.spinner("Analyzing image..."):
            img_array = preprocess_img(img)


            y_probs = model.predict(img_array, verbose=0)[0]


            y_pred = np.argmax(y_probs)

            class_labels = ['NORMAL', 'PNEUMONIA']

            predicted_class = class_labels[y_pred]

            confidence = y_probs[y_pred]

            st.subheader("Diagnosis Result")
            st.markdown(
                f"**Prediction:** `{predicted_class}`")
            st.markdown(f"**Confidence:** `{confidence:.2%}`")

            st.markdown("**Class Probabilities:**")
            for i, (label, prob) in enumerate(zip(class_labels, y_probs)):
                st.markdown(f"- {label}: `{prob:.2%}`")


    # Grad-CAM visualization
    with st.expander("🔬 View Grad-CAM Heatmap"):
        with st.spinner("Generating heatmap..."):
            last_conv_layer_name = 'mixed10'
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

            if heatmap is not None:
                cam_image = create_gradcam_visualization(img_array, heatmap, img)
                if cam_image is not None:

                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(cam_image, caption="Important regions highlighted", use_container_width=True)
            else:
                st.error("Could not generate Grad-CAM visualization.")

