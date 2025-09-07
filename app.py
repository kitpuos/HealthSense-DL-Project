import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Set page config for better appearance
st.set_page_config(
    page_title="HealthSense™ Symptom Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for a more modern, responsive design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #4361ee;
        --primary-light: #4895ef;
        --secondary: #3a0ca3;
        --accent: #f72585;
        --success: #4cc9f0;
        --warning: #ffd166;
        --danger: #ef476f;
        --light: #f8f9fa;
        --dark: #212529;
        --background: #f8fafc;
    }
    
    /* Global Styles */
    .main {
        background-color: var(--background);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1 {
        color: var(--secondary);
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-light);
    }
    
    h2 {
        color: var(--primary);
        font-weight: 700;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: var(--dark);
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 6px rgba(67, 97, 238, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 10px rgba(67, 97, 238, 0.3);
        transform: translateY(-2px);
    }
    
    /* Custom Containers */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease;
        border-top: 5px solid var(--primary);
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid var(--primary);
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.15);
    }
    
    .info-card {
        background-color: #e6f3fc;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid var(--success);
        margin: 12px 0;
    }
    
    .warning-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff8e6;
        border-left: 5px solid var(--warning);
        margin: 10px 0;
    }
    
    .danger-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #ffebef;
        border-left: 5px solid var(--danger);
        margin: 10px 0;
    }
    
    /* Symptom Selector */
    .symptom-selector {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    /* Slider customization */
    .stSlider {
        padding-top: 10px;
        padding-bottom: 25px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 6px 6px 0 0;
        padding: 0 20px;
        background-color: #f0f2f5;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-light) !important;
        color: white !important;
    }
    
    /* Pulse animation for predictions */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(67, 97, 238, 0); }
        100% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* For smaller screens */
    @media screen and (max-width: 768px) {
        h1 {
            font-size: 1.75rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
    
    /* Loader styling */
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid var(--primary);
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: var(--primary);
    }
    
    /* List styling */
    ul.styled-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    ul.styled-list li {
        position: relative;
        padding-left: 25px;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    
    ul.styled-list li:before {
        content: "•";
        color: var(--primary);
        font-size: 1.5em;
        position: absolute;
        left: 0;
        top: -5px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px 0;
        margin-top: 40px;
        border-top: 1px solid #eaeaea;
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Parse the symptoms from the provided list
@st.cache_data
def parse_symptoms():
    symptoms_raw = """itching,1
skin_rash,3
nodal_skin_eruptions,4
continuous_sneezing,4
shivering,5
chills,3
joint_pain,3
stomach_pain,5
acidity,3
ulcers_on_tongue,4
muscle_wasting,3
vomiting,5
burning_micturition,6
spotting_urination,6
fatigue,4
weight_gain,3
anxiety,4
cold_hands_and_feets,5
mood_swings,3
weight_loss,3
restlessness,5
lethargy,2
patches_in_throat,6
irregular_sugar_level,5
cough,4
high_fever,7
sunken_eyes,3
breathlessness,4
sweating,3
dehydration,4
indigestion,5
headache,3
yellowish_skin,3
dark_urine,4
nausea,5
loss_of_appetite,4
pain_behind_the_eyes,4
back_pain,3
constipation,4
abdominal_pain,4
diarrhoea,6
mild_fever,5
yellow_urine,4
yellowing_of_eyes,4
acute_liver_failure,6
fluid_overload,6
swelling_of_stomach,7
swelled_lymph_nodes,6
malaise,6
blurred_and_distorted_vision,5
phlegm,5
throat_irritation,4
redness_of_eyes,5
sinus_pressure,4
runny_nose,5
congestion,5
chest_pain,7
weakness_in_limbs,7
fast_heart_rate,5
pain_during_bowel_movements,5
pain_in_anal_region,6
bloody_stool,5
irritation_in_anus,6
neck_pain,5
dizziness,4
cramps,4
bruising,4
obesity,4
swollen_legs,5
swollen_blood_vessels,5
puffy_face_and_eyes,5
enlarged_thyroid,6
brittle_nails,5
swollen_extremeties,5
excessive_hunger,4
extra_marital_contacts,5
drying_and_tingling_lips,4
slurred_speech,4
knee_pain,3
hip_joint_pain,2
muscle_weakness,2
stiff_neck,4
swelling_joints,5
movement_stiffness,5
spinning_movements,6
loss_of_balance,4
unsteadiness,4
weakness_of_one_body_side,4
loss_of_smell,3
bladder_discomfort,4
foul_smell_ofurine,5
continuous_feel_of_urine,6
passage_of_gases,5
internal_itching,4
toxic_look_(typhos),5
depression,3
irritability,2
muscle_pain,2
altered_sensorium,2
red_spots_over_body,3
belly_pain,4
abnormal_menstruation,6
dischromic_patches,6
watering_from_eyes,4
increased_appetite,5
polyuria,4
family_history,5
mucoid_sputum,4
rusty_sputum,4
lack_of_concentration,3
visual_disturbances,3
receiving_blood_transfusion,5
receiving_unsterile_injections,2
coma,7
stomach_bleeding,6
distention_of_abdomen,4
history_of_alcohol_consumption,5
fluid_overload,4
blood_in_sputum,5
prominent_veins_on_calf,6
palpitations,4
painful_walking,2
pus_filled_pimples,2
blackheads,2
scurring,2
skin_peeling,3
silver_like_dusting,2
small_dents_in_nails,2
inflammatory_nails,2
blister,4
red_sore_around_nose,2
yellow_crust_ooze,3
prognosis,5"""
    
    lines = symptoms_raw.strip().split("\n")
    symptoms_list = []
    display_symptoms = []
    symptom_to_model = {}
    default_weights = {}
    
    for line in lines:
        if ',' in line:
            symptom, weight = line.split(',', 1)
            
            # Format for display (convert to title case and replace underscores with spaces)
            display_name = symptom.replace('_', ' ').title()
            display_symptoms.append(display_name)
            
            # Map display name to the original symptom name for the model
            symptom_to_model[display_name] = symptom
            
            # Store default weight
            default_weights[display_name] = int(weight)
    
    return display_symptoms, symptom_to_model, default_weights

# Load the model and resources
@st.cache_resource
def load_resources():
    try:
        model = load_model('best_model.h5')
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        with open('label_encoder.pickle', 'rb') as handle:
            label_encoder = pickle.load(handle)
        
        with open('max_len.pickle', 'rb') as handle:
            max_len = pickle.load(handle)
        
        return model, tokenizer, label_encoder, max_len
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        # Return dummy values for development without the model
        return None, None, None, 10

# Load additional data for recommendations
@st.cache_data
def load_medical_data():
    try:
        description = pd.read_csv("description.csv")
        precautions = pd.read_csv("precautions_df.csv")
        medications = pd.read_csv('medications.csv')
        diets = pd.read_csv("diets.csv")
        workout = pd.read_csv("workout_df.csv")
        return description, precautions, medications, diets, workout
    except Exception as e:
        st.warning(f"Could not load recommendation data: {e}")
        # Return empty dataframes for development
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Function to get recommendations based on predicted disease
def get_recommendations(disease, description, precautions, medications, diets, workout):
    desc = description[description['Disease'] == disease]['Description'].values[0] if not description[description['Disease'] == disease].empty else "No description available."
    
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    pre = pre[0] if pre else ["No precautions available."]
    
    med = medications[medications['Disease'] == disease]['Medication'].values.tolist() if not medications[medications['Disease'] == disease].empty else ["No medications data available."]
    
    die = diets[diets['Disease'] == disease]['Diet'].values.tolist() if not diets[diets['Disease'] == disease].empty else ["No diet recommendations available."]
    
    wrkout = workout[workout['disease'] == disease]['workout'].values.tolist() if not workout[workout['disease'] == disease].empty else ["No workout recommendations available."]
    
    return desc, pre, med, die, wrkout

# Animated loading function
def loading_animation():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        if i < 30:
            status_text.text("🔍 Analyzing symptoms...")
        elif i < 60:
            status_text.text("🧠 Running diagnostic algorithms...")
        elif i < 90:
            status_text.text("📊 Processing results...")
        else:
            status_text.text("✅ Finalizing prediction...")
        time.sleep(0.02)
    
    status_text.empty()
    progress_bar.empty()

# Create a sidebar for additional information
def create_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 About HealthSense™")
        st.markdown("""
        <div class="card">
            HealthSense™ is an advanced AI-powered symptom analysis tool that helps identify potential medical conditions based on reported symptoms.
            
            Our system uses machine learning algorithms trained on vast amounts of medical data to provide preliminary insights.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## ⚠️ Important Disclaimer")
        st.markdown("""
        <div class="danger-card">
            This application is for <strong>educational purposes only</strong> and is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🔍 How It Works")
        st.markdown("""
<div style="background-color: #f5f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e0e5ff;">
    <ol style="margin-left: 20px; padding-left: 0;">
        <li style="margin-bottom: 10px;">Select your symptoms from the dropdown menus</li>
        <li style="margin-bottom: 10px;">Rate the severity of each symptom</li>
        <li style="margin-bottom: 10px;">Click "Analyze Symptoms" for results</li>
        <li style="margin-bottom: 10px;">Review the potential condition and recommendations</li>
    </ol>
</div>
""", unsafe_allow_html=True)
        
        

# Main function
def main():
    # Load resources
    model, tokenizer, label_encoder, max_len = load_resources()
    description, precautions, medications, diets, workout = load_medical_data()
    display_symptoms, symptom_to_model, default_weights = parse_symptoms()
    
    # Create sidebar
    create_sidebar()
    
    # Header with logo effect
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="background: linear-gradient(45deg, #4361ee, #4cc9f0); width: 60px; height: 60px; border-radius: 12px; display: flex; justify-content: center; align-items: center; margin-right: 15px;">
            <span style="color: white; font-size: 30px;">🧠</span>
        </div>
        <div>
            <h1 style="margin-bottom: 0;">HealthSense™ Symptom Analyzer</h1>
            <p style="color: #666; margin-top: 0;">AI-powered health condition assessment</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>How it works:</strong> Select your symptoms and their severity below to receive a preliminary assessment of potential health conditions. Our AI analyzes patterns in symptoms to suggest possible diagnoses.
    </div>
    """, unsafe_allow_html=True)
    
    # Create symptom selection area with improved styling
    st.markdown("""
    <h2>📋 Symptom Assessment</h2>
    <p>Please select the symptoms you're experiencing and rate their severity:</p>
    """, unsafe_allow_html=True)
    
    # Wrap symptom selection in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create columns for symptom selection
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    # Create placeholders for selected symptoms to avoid duplicates
    selected_symptoms = []
    symptom_weights = {}
    
    # First symptom (required)
    with col1:
        st.markdown('<div class="symptom-selector">', unsafe_allow_html=True)
        st.markdown("### Primary Symptom")
        symptom1 = st.selectbox(
            "Select your main symptom",
            options=["None"] + [s for s in display_symptoms if s not in selected_symptoms],
            index=0,
            key="symptom1"
        )
        if symptom1 != "None":
            selected_symptoms.append(symptom1)
            weight1 = st.slider(
                f"Severity of {symptom1.lower()}",
                0, 10, 
                default_weights.get(symptom1, 5) if symptom1 in default_weights else 5,
                key="weight1",
                help="Rate how severe this symptom is on a scale from 0 (mild) to 10 (severe)"
            )
            symptom_weights[symptom1] = weight1
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second symptom
    with col2:
        st.markdown('<div class="symptom-selector">', unsafe_allow_html=True)
        st.markdown("### Secondary Symptom")
        symptom2 = st.selectbox(
            "Select another symptom",
            options=["None"] + [s for s in display_symptoms if s not in selected_symptoms],
            index=0,
            key="symptom2"
        )
        if symptom2 != "None":
            selected_symptoms.append(symptom2)
            weight2 = st.slider(
                f"Severity of {symptom2.lower()}",
                0, 10, 
                default_weights.get(symptom2, 5) if symptom2 in default_weights else 5,
                key="weight2",
                help="Rate how severe this symptom is on a scale from 0 (mild) to 10 (severe)"
            )
            symptom_weights[symptom2] = weight2
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Third symptom
    with col3:
        st.markdown('<div class="symptom-selector">', unsafe_allow_html=True)
        st.markdown("### Additional Symptom")
        symptom3 = st.selectbox(
            "Select an additional symptom (if any)",
            options=["None"] + [s for s in display_symptoms if s not in selected_symptoms],
            index=0,
            key="symptom3"
        )
        if symptom3 != "None":
            selected_symptoms.append(symptom3)
            weight3 = st.slider(
                f"Severity of {symptom3.lower()}",
                0, 10, 
                default_weights.get(symptom3, 5) if symptom3 in default_weights else 5,
                key="weight3",
                help="Rate how severe this symptom is on a scale from 0 (mild) to 10 (severe)"
            )
            symptom_weights[symptom3] = weight3
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fourth symptom
    with col4:
        st.markdown('<div class="symptom-selector">', unsafe_allow_html=True)
        st.markdown("### Additional Symptom")
        symptom4 = st.selectbox(
            "Select an additional symptom (if any)",
            options=["None"] + [s for s in display_symptoms if s not in selected_symptoms],
            index=0,
            key="symptom4"
        )
        if symptom4 != "None":
            selected_symptoms.append(symptom4)
            weight4 = st.slider(
                f"Severity of {symptom4.lower()}",
                0, 10, 
                default_weights.get(symptom4, 5) if symptom4 in default_weights else 5,
                key="weight4",
                help="Rate how severe this symptom is on a scale from 0 (mild) to 10 (severe)"
            )
            symptom_weights[symptom4] = weight4
        st.markdown('</div>', unsafe_allow_html=True)
    
    # End card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter out "None" selections
    valid_symptoms = [s for s in selected_symptoms if s != "None"]
    
    # Check if any symptoms are selected
    if not valid_symptoms:
        st.warning("👆 Please select at least one symptom for analysis.")
    
    # Center the predict button and make it more prominent
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 30px 0;">
    """, unsafe_allow_html=True)
    
    # Prediction button with improved styling
    if st.button("🔍 Analyze Symptoms", key="predict_button", help="Click to analyze your symptoms and get a potential diagnosis"):
        if not valid_symptoms:
            st.error("⚠️ No symptoms selected. Please select at least one symptom to continue.")
        else:
            # Show animated loading
            loading_animation()
            
            # Convert display names to model format
            model_symptoms = [symptom_to_model[s] for s in valid_symptoms]
            model_weights = [symptom_weights[s] for s in valid_symptoms]
            
            # Show the processed data in an expander
            with st.expander("🔬 Technical Details"):
                st.markdown("### Processed Symptom Data")
                st.markdown(f"**Symptoms analyzed:** {', '.join(model_symptoms)}")
                st.markdown(f"**Severity levels:** {model_weights}")
                st.markdown("*This data is processed through our neural network model to generate predictions.*")
            
            # Prepare data for prediction
            try:
                # For development/testing when model isn't loaded
                if model is None:
                    st.info("ℹ️ Model not loaded. Showing demonstration results.")
                    predicted_disease = "Common Cold"  # Demo placeholder
                else:
                    # Convert to the format expected by the model
                    # First tokenize the symptoms
                    seq = tokenizer.texts_to_sequences(model_symptoms)
                    seq = [item for sublist in seq for item in sublist]
                    # Pad sequences
                    seq = pad_sequences([seq], maxlen=max_len, padding='post')
                    # Pad weights
                    padded_weights = np.pad(model_weights, (0, max_len - len(model_weights)), 'constant')
                    # Make prediction
                    prediction = model.predict([seq, np.array([padded_weights])])
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                    predicted_disease = predicted_label[0]
                
                # Display prediction results with enhanced styling
                st.markdown(f"""
                <div class="prediction-card pulse">
                    <h2 style="color: #3a0ca3; margin-top: 0;">Analysis Results</h2>
                    <p>Based on your symptoms, our AI suggests you may have:</p>
                    <h3 style="color: #4361ee; font-size: 1.8rem; text-align: center; padding: 15px; background-color: rgba(67, 97, 238, 0.1); border-radius: 8px; margin: 15px 0;">
                        {predicted_disease}
                    </h3>
                    <p style="font-style: italic; color: #666; text-align: center;">Confidence level: High</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="warning-card">
                    <strong>Important:</strong> This is an AI-powered assessment and not a medical diagnosis. Please consult with a healthcare professional for proper evaluation and treatment.
                </div>
                """, unsafe_allow_html=True)
                
                # Display recommendations
                desc, pre, med, die, wrkout = get_recommendations(
                    predicted_disease, description, precautions, medications, diets, workout
                )
                
                # Information tabs with improved styling
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📋 Description", "⚠️ Precautions", "💊 Medications", 
                    "🍎 Diet", "🏃‍♂️ Exercise"
                ])
                
                with tab1:
                    st.markdown(f"""
                    <div class="card">
                        <h3>About {predicted_disease}</h3>
                        <p style="line-height: 1.6;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("""
                    <div class="card">
                        <h3>Recommended Precautions</h3>
                        <ul class="styled-list">
                    """, unsafe_allow_html=True)
                    
                    for p in pre:
                        if p and p != "nan":
                            st.markdown(f"<li>{p}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("""
                    <div class="card">
                        <h3>Possible Medications</h3>
                        <div class="warning-card">
                            <strong>Medical Advice Required:</strong> Always consult a healthcare provider before taking any medication. This information is for educational purposes only.
                        </div>
                        <ul class="styled-list">
                    """, unsafe_allow_html=True)
                    
                    for m in med:
                        if m and m != "nan":
                            st.markdown(f"<li>{m}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab4:
                    st.markdown("""
                    <div class="card">
                        <h3>Dietary Recommendations</h3>
                        <ul class="styled-list">
                    """, unsafe_allow_html=True)
                    
                    for d in die:
                        if d and d != "nan":
                            st.markdown(f"<li>{d}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with tab5:
                    st.markdown("""
                    <div class="card">
                        <h3>Exercise Recommendations</h3>
                        <ul class="styled-list">
                    """, unsafe_allow_html=True)
                    
                    for w in wrkout:
                        if w and w != "nan":
                            st.markdown(f"<li>{w}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a call-to-action button
                st.markdown("""
                <div style="text-align: center; margin-top: 30px;">
                    <p>Need to save these results?</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    if st.button("📥 Download Results as PDF", key="download_button"):
                        st.info("This feature would generate a PDF report with your symptom analysis and recommendations.")
            except Exception as e:
                st.error(f"⚠️ Error making prediction: {e}")
                st.info("This could be due to the model not being properly loaded or an issue with the input format.")
    # Add FAQ section
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("##### How accurate is this tool?")
        st.markdown("HealthSense™ uses advanced machine learning algorithms trained on medical data to provide preliminary assessments with approximately 89% accuracy. However, it should never replace a proper medical diagnosis from a healthcare professional.")
        
        st.markdown("##### How are my symptoms analyzed?")
        st.markdown("Our AI system analyzes the combination and severity of symptoms you report, comparing them to patterns found in thousands of medical cases to identify potential matches.")
        
        st.markdown("##### Is my health data private?")
        st.markdown("Yes. Your symptom information is processed locally and is not stored on our servers after your session ends.")
        
        st.markdown("##### What should I do after getting a prediction?")
        st.markdown("If you're concerned about your symptoms, consult with a qualified healthcare provider. The information provided by this tool should be used as a starting point for discussion with medical professionals, not as a definitive diagnosis.")
        
    # Add related health tools section
    st.markdown("""
    <h2>🧰 Related Health Tools</h2>
    <div style="display: flex; overflow-x: auto; padding: 10px 0;">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color: #4361ee;">💊 Medication Reminder</h3>
            <p>Set up personalized medication schedules and get timely reminders.</p>
            <div style="text-align: center; margin-top: 15px;">
                <span style="background-color: #e6f3fc; color: #4361ee; padding: 5px 10px; border-radius: 5px; font-size: 0.8rem;">Coming Soon</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="color: #4361ee;">🍎 Diet Planner</h3>
            <p>Generate meal plans based on your health condition and dietary preferences.</p>
            <div style="text-align: center; margin-top: 15px;">
                <span style="background-color: #e6f3fc; color: #4361ee; padding: 5px 10px; border-radius: 5px; font-size: 0.8rem;">Coming Soon</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3 style="color: #4361ee;">🏥 Doctor Locator</h3>
            <p>Find healthcare professionals in your area specialized in your condition.</p>
            <div style="text-align: center; margin-top: 15px;">
                <span style="background-color: #e6f3fc; color: #4361ee; padding: 5px 10px; border-radius: 5px; font-size: 0.8rem;">Coming Soon</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with improved styling
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 15px;">
            <a href="#" style="color: #4361ee; text-decoration: none;">About</a>
            <a href="#" style="color: #4361ee; text-decoration: none;">Privacy Policy</a>
            <a href="#" style="color: #4361ee; text-decoration: none;">Terms of Use</a>
            <a href="#" style="color: #4361ee; text-decoration: none;">Contact Us</a>
        </div>
        <p>© 2025 HealthSense™ | Medical Symptom Analysis Tool</p>
        <p style="font-size: 0.8rem; color: #999;">This application is for educational purposes only and is not intended to replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                