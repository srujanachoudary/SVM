import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# =============================
# Load CSS
# =============================
def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # prevents Streamlit crash if css not found

load_css("style.css")

# =============================
# Title
# =============================
st.markdown("""
<div class="card">
    <h1>🏦 Smart Loan Approval System</h1>
    <p>This system uses <b>Support Vector Machines (SVM)</b> to predict loan approval.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Load Dataset
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# =============================
# Dataset Preview
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Data Preprocessing
# =============================
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

for col in df.select_dtypes(include='object'):
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# =============================
# Kernel Selection
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚙️ Select SVM Kernel")

kernel = st.radio(
    "Choose Kernel Type",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Confusion Matrix
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Sidebar - Loan Eligibility
# =============================
st.sidebar.header("🔍 Check Loan Eligibility")

# Mappings
gender_map = {"Female": 0, "Male": 1}
married_map = {"No": 0, "Yes": 1}
education_map = {"Not Graduate": 0, "Graduate": 1}
self_emp_map = {"No": 0, "Yes": 1}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

user_input = []

user_input.append(gender_map[
    st.sidebar.selectbox("Gender", list(gender_map.keys()))
])

user_input.append(married_map[
    st.sidebar.selectbox("Married", list(married_map.keys()))
])

user_input.append(dependents_map[
    st.sidebar.selectbox("Dependents", list(dependents_map.keys()))
])

user_input.append(education_map[
    st.sidebar.selectbox("Education", list(education_map.keys()))
])

user_input.append(self_emp_map[
    st.sidebar.selectbox("Self Employed", list(self_emp_map.keys()))
])

user_input.append(st.sidebar.number_input("Applicant Income", min_value=0))
user_input.append(st.sidebar.number_input("Coapplicant Income", min_value=0))
user_input.append(st.sidebar.number_input("Loan Amount", min_value=0))
user_input.append(st.sidebar.number_input("Loan Amount Term", min_value=0))
user_input.append(st.sidebar.selectbox("Credit History", [0, 1]))

user_input.append(property_map[
    st.sidebar.selectbox("Property Area", list(property_map.keys()))
])

user_input = np.array(user_input).reshape(1, -1)
user_input = scaler.transform(user_input)

# =============================
# Prediction
# =============================
if st.sidebar.button("Check Loan Eligibility"):
    result = model.predict(user_input)[0]
    confidence = model.predict_proba(user_input).max()

    if result == 1:
        st.sidebar.success("✅ Loan Approved")
        decision = "likely"
    else:
        st.sidebar.error("❌ Loan Rejected")
        decision = "unlikely"

    st.sidebar.info(
        f"Based on credit history and income patterns, "
        f"the applicant is **{decision} to repay the loan**."
    )

    st.sidebar.caption(
        f"Kernel Used: **{kernel}** | Confidence: **{confidence*100:.2f}%**"
    )
