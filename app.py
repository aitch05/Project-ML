import streamlit as st
st.title("My Awesome Web App")
st.write("Hello world! This is my Streamlit app running from GitHub.")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

try:
    df = pd.read_csv("dating_app_behavior_dataset (final).csv")

    X = df.drop(columns=['match_outcome', 'mutual_matches'])
    y = df['match_outcome']

    categorical_features = ['education_level', 'gender', 'income_bracket']
    numerical_features = ['likes_received', 'swipe_right_ratio']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_pipeline.fit(X_train, y_train)
    print("✅ Success: Real model successfully trained on your dataset features!")

    joblib.dump(best_pipeline, 'best_multiclass_model.pkl')
    print("💾 Success: Pipeline saved locally as 'best_multiclass_model.pkl'")

except FileNotFoundError:
    print("❌ ERROR: Could not find 'dating_app_behavior_dataset (final).csv'.")
    print("👉 Please click the folder icon on the left of Colab and upload your CSV file first!")
    raise

# STEP 2: Install Streamlit, LocalTunnel, and write the website script

import os
os.system('pip install -q streamlit joblib scikit-learn pandas numpy')
os.system('npm install -g localtunnel')
print("✅ Success: Environments and libraries prepared.")

with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Layout Configurations
st.set_page_config(page_title="Dating Match Oracle", page_icon="🔮", layout="centered")

st.title("🔮 Love, Left on Read, or Lost")
st.subheader("Predicting Full Relationship Outcomes via Machine Learning")
st.write("Adjust profile configurations in the sidebar to see which `match_outcome` class the model predicts.")
st.markdown("---")

# Load your group's actual trained model file
@st.cache_resource
def load_my_real_model():
    return joblib.load('best_multiclass_model.pkl')

try:
    trained_pipeline = load_my_real_model()
except Exception as e:
    st.error("Could not find the 'best_multiclass_model.pkl' file.")

# Sidebar input structure fields
st.sidebar.header("👤 Profile Attributes")
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
user_edu = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
user_income = st.sidebar.selectbox("Income Bracket", ["Low", "Medium", "High"])

st.sidebar.markdown("---")
st.sidebar.header("📊 Behavior Trends")
user_likes = st.sidebar.slider("Likes Received", min_value=0, max_value=500, value=120, step=5)
user_swipe = st.sidebar.slider("Swipe Right Ratio", min_value=0.0, max_value=1.0, value=0.40, step=0.05)

# Wrap user answers into a single-row DataFrame
input_data = pd.DataFrame([{
    'gender': user_gender,
    'education_level': user_edu,
    'income_bracket': user_income,
    'likes_received': user_likes,
    'swipe_right_ratio': user_swipe
}])

# Execute prediction step when user updates trigger
if st.button("Predict Match Outcome"):
    prediction = trained_pipeline.predict(input_data)[0]
    st.subheader("🎯 Predicted Class Result")

    if prediction == 'Mutual Match':
        st.success(f"💖 **Outcome: {prediction}!** This behavioral profile indicates highly successful connection alignments.")
        st.balloons()
    elif prediction == 'Ghosted':
        st.error(f"👻 **Outcome: {prediction}.** High initial interaction with severe conversational cutoff thresholds observed.")
    elif prediction == 'Ignored':
        st.warning(f"⏳ **Outcome: {prediction}.** Profile metrics point to minimal interaction engagement.")
    else:
        st.info(f"🎭 **Outcome: {prediction}.** Discrepancies exist between user patterns and expected profiles.")
''')
print("✅ Success: 'app.py' written to workspace memory.")

import os
import ssl

# 1. Tell Python and standard libraries to ignore SSL certificate validation
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Install pyngrok while telling pip to trust the hosts despite the missing certificate
!pip install pyngrok --quiet --trusted-host pypi.org --trusted-host files.pythonhosted.org

# 3. Authenticate with ngrok (Skip SSL verification for the CLI config tool)
# Swap in your actual token here
!ngrok config add-authtoken 3E1wzlL9myZYT7F5Jx3LRlJSOg0_4Mc2Ao8Wx6CoFNjUaNuAE

# 4. Start Streamlit cleanly in the background
!nohup streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false &

# 5. Give the server a moment to spin up
import time
time.sleep(3)

# 6. Open the stable Ngrok tunnel
from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501, "http")

print("\n🎉 Success! Click the link below to open your stable app:")
print(public_url.public_url)
