import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config must be the first streamlit command
st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('student_performance_rf_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Support Lists for Dropdowns
school_opts = ['Gabriel Pereira (GP)', 'Mousinho da Silveira (MS)']
sex_opts = ['Female (F)', 'Male (M)']
address_opts = ['Urban (U)', 'Rural (R)']
famsize_opts = ['Greater than 3 (GT3)', 'Less or equal to 3 (LE3)']
pstatus_opts = ['Living together (T)', 'Apart (A)']
mjob_opts = ['at_home', 'health', 'other', 'services', 'teacher']
fjob_opts = ['at_home', 'health', 'other', 'services', 'teacher']
reason_opts = ['course', 'other', 'home', 'reputation']
guardian_opts = ['mother', 'father', 'other']
yes_no_opts = ['no', 'yes']

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        color: white;
        border: none;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: 500;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #1f2937;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #374151;
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-center.png", width=80)
    st.title("Student AI")
    st.info("This tool uses Machine Learning to predict student final grades (G3) based on demographic and behavioral data.")
    st.markdown("---")
    st.subheader("Configuration")
    school = st.selectbox("School", school_opts)
    sex = st.selectbox("Sex", sex_opts)
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    address = st.selectbox("Address", address_opts)
    st.markdown("---")
    st.caption("Student Performance Prediction v1.0")

# Main Content
st.title("🎓 Student Performance Prediction")
st.markdown('<p class="big-font">Enter student details to forecast academic performance.</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    
    tab1, tab2, tab3 = st.tabs(["🏠 Family & Background", "📚 Academic & Study", "🎉 Lifestyle & Social"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Family Status")
            famsize = st.selectbox("Family Size", famsize_opts)
            pstatus = st.selectbox("Parent's Cohabitation Status", pstatus_opts)
            guardian = st.selectbox("Guardian", guardian_opts)
        
        with col2:
            st.subheader("Parent's Profile")
            col2a, col2b = st.columns(2)
            with col2a:
                medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=2, format_func=lambda x: f"{x}: " + ["None", "Primary (4th)", "5th-9th", "Secondary", "Higher"][x])
                mjob = st.selectbox("Mother's Job", mjob_opts)
            with col2b:
                fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], index=2, format_func=lambda x: f"{x}: " + ["None", "Primary (4th)", "5th-9th", "Secondary", "Higher"][x])
                fjob = st.selectbox("Father's Job", fjob_opts)
        
        st.subheader("Reason for Choosing School")
        reason = st.radio("Reason", reason_opts, horizontal=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Study Habits")
            traveltime = st.select_slider("Travel Time", options=[1, 2, 3, 4], value=1, format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30 min-1 hr", 4: ">1 hr"}[x])
            studytime = st.select_slider("Weekly Study Time", options=[1, 2, 3, 4], value=2, format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x])
            failures = st.number_input("Past Class Failures", 0, 3, 0)
        
        with col2:
            st.subheader("Support & Extras")
            schoolsup = st.toggle("Extra Educational Support", value=False)
            famsup = st.toggle("Family Educational Support", value=True)
            paid = st.toggle("Extra Paid Classes", value=False)
            nursery = st.toggle("Attended Nursery School", value=True)
            higher = st.toggle("Wants Higher Education", value=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Social Life")
            internet = st.checkbox("Internet Access at Home", value=True)
            romantic = st.checkbox("In a Romantic Relationship", value=False)
            activities = st.checkbox("Extra-curricular Activities", value=True)
            st.markdown("---")
            famrel = st.slider("Family Relationships Quality", 1, 5, 4)
            freetime = st.slider("Free Time after School", 1, 5, 3)
            goout = st.slider("Going Out with Friends", 1, 5, 3)
            
        with col2:
            st.subheader("Health & Consumption")
            health = st.slider("Current Health Status", 1, 5, 4)
            absences = st.number_input("Number of School Absences", 0, 93, 2, step=1)
            st.warning("Alcohol Consumption (1: Very Low - 5: Very High)")
            dalc = st.slider("Workday Alcohol", 1, 5, 1)
            walc = st.slider("Weekend Alcohol", 1, 5, 1)

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Performance")

if submitted:
    # --- Preprocessing ---
    # 1. Label Encoding mappings (Alphabetical)
    school_val = 0 if 'GP' in school else 1
    sex_val = 0 if 'F' in sex else 1
    address_val = 0 if 'R' in address else 1
    famsize_val = 0 if 'GT3' in famsize else 1
    pstatus_val = 0 if 'A' in pstatus else 1
    
    # Yes/No mapping
    schoolsup_val = 1 if schoolsup else 0
    famsup_val = 1 if famsup else 0
    paid_val = 1 if paid else 0
    activities_val = 1 if activities else 0
    nursery_val = 1 if nursery else 0
    higher_val = 1 if higher else 0
    internet_val = 1 if internet else 0
    romantic_val = 1 if romantic else 0

    # 2. Derived Features
    total_alc = dalc + walc
    parent_edu = (medu + fedu) / 2.0
    total_support = schoolsup_val + famsup_val

    # 3. One Hot Encoding with drop_first (Manual)
    mjob_health = 1 if mjob == 'health' else 0
    mjob_other = 1 if mjob == 'other' else 0
    mjob_services = 1 if mjob == 'services' else 0
    mjob_teacher = 1 if mjob == 'teacher' else 0
    
    fjob_health = 1 if fjob == 'health' else 0
    fjob_other = 1 if fjob == 'other' else 0
    fjob_services = 1 if fjob == 'services' else 0
    fjob_teacher = 1 if fjob == 'teacher' else 0

    reason_home = 1 if reason == 'home' else 0
    reason_other = 1 if reason == 'other' else 0
    reason_reputation = 1 if reason == 'reputation' else 0

    guardian_mother = 1 if guardian == 'mother' else 0
    guardian_other = 1 if guardian == 'other' else 0

    # Construct DataFrame
    data = {
        'school': [school_val],
        'sex': [sex_val],
        'age': [age],
        'address': [address_val],
        'famsize': [famsize_val],
        'Pstatus': [pstatus_val],
        'Medu': [medu],
        'Fedu': [fedu],
        'traveltime': [traveltime],
        'studytime': [studytime],
        'failures': [failures],
        'schoolsup': [schoolsup_val],
        'famsup': [famsup_val],
        'paid': [paid_val],
        'activities': [activities_val],
        'nursery': [nursery_val],
        'higher': [higher_val],
        'internet': [internet_val],
        'romantic': [romantic_val],
        'famrel': [famrel],
        'freetime': [freetime],
        'goout': [goout],
        'Dalc': [dalc],
        'Walc': [walc],
        'health': [health],
        'absences': [absences],
        'Total_Alc': [total_alc],
        'Parent_Edu': [parent_edu],
        'Total_Support': [total_support],
        'Mjob_health': [mjob_health],
        'Mjob_other': [mjob_other],
        'Mjob_services': [mjob_services],
        'Mjob_teacher': [mjob_teacher],
        'Fjob_health': [fjob_health],
        'Fjob_other': [fjob_other],
        'Fjob_services': [fjob_services],
        'Fjob_teacher': [fjob_teacher],
        'reason_home': [reason_home],
        'reason_other': [reason_other],
        'reason_reputation': [reason_reputation],
        'guardian_mother': [guardian_mother],
        'guardian_other': [guardian_other]
    }
    
    input_df = pd.DataFrame(data)

    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric("Predicted G3 Grade", f"{prediction:.2f} / 20")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.write("### Analysis")
            if prediction < 10:
                st.error("⚠️ **High Risk**: The predicted grade is below 10/20. Intervention is recommended.")
                st.progress(min(prediction / 20.0, 1.0))
            else:
                st.success("✅ **Good Performance**: The predicted grade is passing (>= 10/20). Keep it up!")
                st.progress(min(prediction / 20.0, 1.0))
            
            st.caption("Note: This is an AI generation based on the provided inputs. Actual results may vary.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Input Data for Debugging:", input_df)

