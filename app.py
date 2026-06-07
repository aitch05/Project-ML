
import streamlit as st
import pandas as pd
import numpy as np

# Page Layout Configurations
st.set_page_config(page_title="Dating Match Oracle", page_icon="🔮", layout="centered")

st.title("🔮 Love, Left on Read, or Lost")
st.subheader("Predicting Full Relationship Outcomes via Machine Learning")
st.write("Adjust profile configurations in the sidebar to see which `match_outcome` class the model predicts.")
st.markdown("---")

# 🟢 OPTION A: Train the model instantly on the server using your CSV file!
@st.cache_resource
def train_model_live():
    df = pd.read_csv("dating_app_behavior_dataset (final).csv")
    X = df.drop(columns=['match_outcome', 'mutual_matches'])
    y = df['match_outcome']

    categorical_features = ['education_level', 'gender', 'income_bracket']
    numerical_features = ['likes_received', 'swipe_right_ratio']

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    best_pipeline.fit(X, y)
    return best_pipeline

trained_pipeline = train_model_live()

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
