import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import plotly.express as px
import pandas as pd

# 🔹 Path to best model
MODEL_PATH = r"C:\Guvi\Fish Classification\Tranied models\VGG16\VGG16_final.h5"

# 🔹 Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# 🔹 Class names (from your dataset)
CLASS_NAMES = [
    'Animal Fish',
    'Animal Fish Bass',
    'Black Sea Sprat',
    'Gilt Head Bream',
    'Horse Mackerel',
    'Red Mullet',
    'Red Sea Bream',
    'Sea Bass',
    'Shrimp',
    'Striped Red Mullet',
    'Trout'
]

# 🔹 Page configuration
st.set_page_config(
    page_title="DeepFish Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔹 Header section
st.markdown("# 🐟 DeepFish Classification System")
st.markdown("### *Advanced AI-Powered Marine Species Identification*")
st.markdown("---")

# 🔹 Sidebar information
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.info("""
    **Model Architecture:** VGG16
    
    **Classes Supported:** 11 Fish Species
    
    **Input Size:** 224x224 pixels
    
    **Accuracy:** High precision classification
    """)
    
    st.markdown("## ⚙️ Confidence Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=95,
        value=80,
        step=5,
        help="Predictions below this threshold will be rejected"
    )
    
    st.markdown("## 🎯 Supported Species")
    for i, class_name in enumerate(CLASS_NAMES, 1):
        st.markdown(f"{i}. {class_name}")
    
    st.markdown("---")
    st.markdown("## 📝 Instructions")
    st.markdown("""
    1. Upload a clear fish image
    2. Ensure good lighting
    3. Fish should be clearly visible
    4. Supported formats: JPG, JPEG, PNG
    5. Adjust confidence threshold as needed
    """)

# 🔹 Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a fish image for classification",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a fish for species identification"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(
            image, 
            caption=f'Uploaded Image: {uploaded_file.name}',
            use_container_width=True
        )
        
        # Image details
        st.markdown("### 📋 Image Details")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Width", f"{image.size[0]}px")
        with col_b:
            st.metric("Height", f"{image.size[1]}px")
        with col_c:
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())

with col2:
    st.markdown("## 🔍 Classification Results")
    
    if uploaded_file is not None:
        with st.spinner('🧠 Analyzing image with AI model...'):
            # Preprocess image
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(np.max(predictions) * 100)  # Convert to Python float
        
        # Display threshold information
        st.markdown(f"### 🎯 Primary Prediction (Threshold: {confidence_threshold}%)")
        
        # Apply threshold-based filtering
        if confidence >= confidence_threshold:
            # High confidence prediction - accepted
            st.success(f"✅ **ACCEPTED PREDICTION**")
            st.success(f"**Species:** {predicted_class}")
            st.success(f"**Confidence:** {confidence:.2f}%")
            
            # Additional success metrics
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.metric("Status", "✅ Accepted", delta="High Confidence")
            with col_status2:
                st.metric("Reliability", "High", delta=f"+{confidence-confidence_threshold:.1f}%")
                
        else:
            # Low confidence prediction - rejected
            st.error(f"❌ **PREDICTION REJECTED**")
            st.error(f"**Species:** {predicted_class}")
            st.error(f"**Confidence:** {confidence:.2f}%")
            st.warning(f"⚠️ Confidence below threshold ({confidence_threshold}%)")
            
            # Rejection metrics
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.metric("Status", "❌ Rejected", delta="Low Confidence")
            with col_status2:
                st.metric("Reliability", "Low", delta=f"-{confidence_threshold-confidence:.1f}%")
            
            # Suggestions for rejected predictions
            st.markdown("### 💡 Suggestions")
            st.info("""
            **To improve prediction quality:**
            - Try a clearer, higher resolution image
            - Ensure better lighting conditions
            - Make sure the fish is the main subject
            - Lower the confidence threshold if appropriate
            - Try a different angle or photo
            """)
        
        # Confidence meter (always show)
        st.markdown("### 📊 Confidence Level")
        
        # Color-coded progress bar
        if confidence >= confidence_threshold:
            st.success(f"Confidence: {confidence:.2f}% (Above threshold)")
        else:
            st.error(f"Confidence: {confidence:.2f}% (Below threshold)")
        
        # Progress bar with threshold line
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.progress(confidence/100)
        with progress_col2:
            st.markdown(f"**{confidence:.1f}%**")
        
        # Threshold indicator
        st.markdown(f"🎯 **Threshold Line:** {confidence_threshold}%")
        
        # Top 3 predictions (always show for analysis)
        st.markdown("### 🏆 Top 3 Predictions")
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices):
            rank = i + 1
            species = CLASS_NAMES[idx]
            conf = float(predictions[0][idx] * 100)
            
            # Color code based on threshold
            if conf >= confidence_threshold:
                status_icon = "✅"
                status_color = "normal"
            else:
                status_icon = "❌"
                status_color = "inverse"
            
            col_rank, col_species, col_conf, col_status = st.columns([1, 3, 2, 1])
            with col_rank:
                st.markdown(f"**#{rank}**")
            with col_species:
                st.markdown(f"{species}")
            with col_conf:
                st.markdown(f"{conf:.1f}%")
            with col_status:
                st.markdown(f"{status_icon}")
        
        # Prediction distribution chart
        st.markdown("### 📈 Prediction Distribution")
        
        # Create dataframe for plotting
        df_predictions = pd.DataFrame({
            'Species': CLASS_NAMES,
            'Confidence': [float(x * 100) for x in predictions[0]],
            'Above_Threshold': [float(x * 100) >= confidence_threshold for x in predictions[0]]
        }).sort_values('Confidence', ascending=True)
        
        # Create horizontal bar chart with threshold line
        fig = px.bar(
            df_predictions.tail(5),
            x='Confidence',
            y='Species',
            orientation='h',
            title=f'Top 5 Species Confidence Scores (Threshold: {confidence_threshold}%)',
            color='Above_Threshold',
            color_discrete_map={True: '#00CC96', False: '#FF6B6B'}
        )
        
        # Add threshold line
        fig.add_vline(
            x=confidence_threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: {confidence_threshold}%"
        )
        
        fig.update_layout(height=300, showlegend=True)
        fig.update_layout(legend=dict(
            title="Status",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📊 Prediction Summary")
        total_above_threshold = sum(1 for x in predictions[0] if float(x * 100) >= confidence_threshold)
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Above Threshold", f"{total_above_threshold}/11")
        with col_sum2:
            st.metric("Max Confidence", f"{confidence:.1f}%")
        with col_sum3:
            threshold_status = "PASS" if confidence >= confidence_threshold else "FAIL"
            st.metric("Threshold Status", threshold_status)
        
    else:
        st.info("👆 Please upload an image to see classification results")
        
        # Show example placeholder
        st.markdown("### 🖼️ Example Output")
        st.markdown(f"""
        Once you upload an image, you'll see:
        - **Threshold-filtered prediction** (Current: {confidence_threshold}%)
        - **Acceptance/Rejection status**
        - **Top 3 most likely species** with threshold indicators
        - **Confidence distribution chart** with threshold line
        - **Detailed analysis metrics**
        """)

# 🔹 Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("### 🔬 Technology")
    st.markdown("- TensorFlow/Keras")
    st.markdown("- VGG16 Architecture")
    st.markdown("- Transfer Learning")
    st.markdown("- Confidence Filtering")

with col_footer2:
    st.markdown("### 📊 Performance")
    st.markdown("- Real-time Classification")
    st.markdown("- Threshold-based Filtering")
    st.markdown("- 11 Species Support")
    st.markdown("- Quality Assurance")

with col_footer3:
    st.markdown("### 🌊 Applications")
    st.markdown("- Marine Research")
    st.markdown("- Fishing Industry")
    st.markdown("- Educational Tools")
    st.markdown("- Quality Control")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🐟 DeepFish Classification System | Powered by AI & Machine Learning | Quality Assured"
    "</div>", 
    unsafe_allow_html=True
)
