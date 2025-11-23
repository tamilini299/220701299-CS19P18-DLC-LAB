import os
os.environ["TORCH_DISABLE_SIGNAL_HANDLERS"] = "1"
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from predict import predict_image, load_model
import time

# 🌈 Page settings
st.set_page_config(page_title="Satellite Image Classifier", layout="wide", page_icon="🛰️")

# 🌟 Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# 🧭 Sidebar Navigation
with st.sidebar:
    st.title("🧭 Navigation")
    selection = st.selectbox("Go to", ["Home", "About the Project", "Contact"])
    st.markdown("---")
    st.markdown("Made with ❤️ using PyTorch + Streamlit")

# 🧠 Load model
model = load_model("model.pth")

# 🌍 Main content based on sidebar selection
if selection == "Home":
    # 🛰️ Title and description
    st.title("🛰️ Satellite Image Classifier")
    st.subheader("🌎 Detect Forests, Crops, Highways, Lakes & More from Satellite Images")

    st.markdown("""
    <div style="background-color:#DFF0D8; padding: 15px; border-radius: 10px;">
        Upload a satellite image and our AI will classify it into one of five classes: 
        <b>Forest</b>, <b>Residential</b>, <b>Annual Crop</b>, <b>Highway</b>, or <b>Sea/Lake</b>.
    </div>
    """, unsafe_allow_html=True)

    # 📤 Upload image
    image = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

    if image:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        with st.spinner("⏳ Analyzing the image... Please wait..."):
            prediction, confidence = predict_image(image, model)
            time.sleep(1)
            st.balloons()

        st.success(f"🎯 **Prediction**: {prediction}")
        st.info(f"📈 **Confidence**: {confidence:.2f}%")

        # 🔎 Class Descriptions
        class_descriptions = {
            "Forest": "🌳 Areas dominated by trees and vegetation.",
            "Residential": "🏠 Human-made structures like buildings or houses.",
            "AnnualCrop": "🌾 Land used for farming seasonal crops.",
            "Highway": "🛣️ Roads, streets, or highways seen from above.",
            "SeaLake": "🌊 Water bodies like seas and lakes."
        }

        st.markdown(f"""
        <div style='background-color:#E8F6F3;padding:10px;border-radius:10px;'>
        <b>Class Description:</b><br>{class_descriptions.get(prediction, 'No description available.')}
        </div>
        """, unsafe_allow_html=True)

        # ✅ Feedback
        st.write("### 🧠 Was the prediction accurate?")
        feedback = st.radio("Your Feedback:", ("👍 Yes", "👎 No"))
        if feedback == "👎 No":
            st.warning("Thanks for your feedback! We'll improve the model.")

        # 🔍 Expanders
        st.write("---")
        st.write("### ℹ️ Want to know how this works?")

        with st.expander("🔬 How does the model work?"):
            st.markdown("""
            This model uses **ResNet-50** pretrained on ImageNet and fine-tuned on the EuroSAT dataset.
            It uses convolutional neural networks to extract features and classify satellite image scenes
            into distinct land use categories.

            - **Framework**: PyTorch  
            - **Dataset**: EuroSAT RGB images  
            - **Classes**: 5 major land types  
            - **Accuracy**: ~95% on validation  
            """)

        with st.expander("📂 About the Dataset"):
            st.markdown("""
            The **EuroSAT** dataset contains 27,000 labeled images from Sentinel-2 satellites across Europe.  
            It supports land use and land cover classification, supporting environmental monitoring, urban planning, and more.
            """)

        with st.expander("🚀 Future Improvements"):
            st.markdown("""
            - Add support for multi-class classification  
            - Improve UI with zoomable maps  
            - Track prediction history and model feedback  
            - Connect to a geolocation database  
            """)

        # 🔗 Explore More Resources
        with st.expander("🌐 Explore more resources"):
            st.markdown("""
            - [Earth Observation AI - GEO](https://earthobservations.org/about-us/news/harnessing-ai-earth-observations-all)  
            - [Satellite Imagery](https://en.wikipedia.org/wiki/Satellite_imagery)
            """)

    # 🦶 Footer
    st.write("---")
    st.markdown("<center>🌍 Built for the love of Earth | Powered by Deep Learning 🚀</center>", unsafe_allow_html=True)

elif selection == "About the Project":
    st.title("📘 About the Project")
    st.markdown("""
    The **Satellite Image Classifier** is a deep learning-powered web application built using **PyTorch** and **Streamlit**.
    It allows users to upload satellite images and get real-time predictions for land use classification.

    This project was developed by Tamilini and Chandra Harsha as a mini-project to demonstrate the application of 
    **convolutional neural networks** in remote sensing and image classification.

    **Key Highlights:**
    - Uses ResNet-50 for robust feature extraction  
    - Trained on EuroSAT dataset with 5 land type classes  
    - Real-time prediction through a user-friendly UI  
    - Supports feedback collection for future improvements  
    """)

elif selection == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
    **👩 Tamilini**  
    - 📧 Email: 220701299@rajalakshmi.edu.in  
    - 📱 Contact No: 6354735687  

    **👨 Chandra Harsha**  
    - 📧 Email: 220701314@rajalakshmi.edu.in  
    - 📱 Contact No: 6305470740  
    """)
