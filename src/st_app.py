import streamlit as st
import requests
from src import inference

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

API_URL = "http://localhost:8000"

def check_api():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False


def make_prediction(customer_data):
    """Send data to API and get prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=customer_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Errors: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


st.title("Churn Prediction App")
if check_api():
    st.success("✅ API Connected")
else:
    st.error("API Not Connected - Start the backend first!")
    st.stop()

st.markdown("---")
st.header("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"], key='partner')
    dependents = st.selectbox("Dependents", ["Yes", "No"], key='dependents')
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key='senior_citizen')
    

with col2:    
    st.subheader("Billing")
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=200)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=1.0, value=10.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=846.0, step=10.0)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Services")
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key='multiple_lines')
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key='internet_service')
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], key='phone_service')
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key='online_security')
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key='online_backup')
with col4:
    st.subheader("Additional Services")
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key='device_protection')
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key='tech_support')
    streaming_tv = st.selectbox("Streaming Tv", ["Yes", "No", "No internet service"], key='streaming_tv')
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key='streaming_movies')
st.markdown("---")

# Predict button
# if st.button("🔮 Predict Churn", use_container_width=True):
customer_data = {
    'tenure': tenure,
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract_type,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}




with st.spinner("Making Prediction..."):
    result = make_prediction(customer_data)

if result:
    st.markdown("---")
    st.header("📊 Prediction Result")
    # st.write(result)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['prediction'] == 1:
            st.metric("Prediction", "WILL CHURN ⚠️", delta="High Risk")
        else:
            st.metric("Prediction", "WILL STAY ✅", delta="Low Risk")
        
    with col2:
        st.metric("Churn Probability", f"{(result['churn_prob']*100):.2f}%")
    
    with col3:
        st.metric("Risk Level", f"{result['risk_level'].capitalize()}")
        
    #     # Interpretation
    #     st.markdown("---")
    #     if result['prediction'] == 1:
    #         st.error(f"""
    #         **⚠️ High Churn Risk!**
            
    #         This customer has a {result['probability']:.1%} probability of churning.
            
    #         **Recommended Actions:**
    #         - Reach out with retention offer
    #         - Consider contract upgrade incentive
    #         - Investigate service quality issues
    #         """)
    #     else:
    #         st.success(f"""
    #         **✅ Low Churn Risk**
            
    #         This customer has only a {result['probability']:.1%} probability of churning.
            
    #         **Opportunities:**
    #         - Customer is satisfied
    #         - Good candidate for upselling
    #         - Can be used as reference
    #         """)