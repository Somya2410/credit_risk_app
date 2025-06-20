{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "98e4d176-4dcd-4c6d-a9d5-6c72f3f597b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-20 15:36:02.383 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\somya Nigam\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-06-20 15:36:02.389 No runtime found, using MemoryCacheStorageManager\n",
      "2025-06-20 15:36:02.391 No runtime found, using MemoryCacheStorageManager\n",
      "2025-06-20 15:36:02.403 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "st.title(\"💳 AI-Driven Credit Risk & Customer Scoring System\")\n",
    "\n",
    "@st.cache_data\n",
    "def load_data():\n",
    "    path = \"C:/Users/somya Nigam/Desktop/credit_data_synthetic.csv\"\n",
    "    return pd.read_csv(path)\n",
    "\n",
    "# Load the data once and keep it cached\n",
    "df = load_data()\n",
    "\n",
    "# Check for any missing or null values and replace with None or NaN\n",
    "df = df.replace({None: np.nan, 'null': np.nan})\n",
    "\n",
    "# Display data sample for understanding (optional)\n",
    "# st.dataframe(df.head())\n",
    "\n",
    "st.sidebar.header(\"📋 Enter Customer Details\")\n",
    "age = st.sidebar.slider(\"Age\", 18, 70, 30)\n",
    "income = st.sidebar.number_input(\"Monthly Income (INR)\", 10000, 1000000, value=50000, step=1000)\n",
    "loan = st.sidebar.number_input(\"Loan Amount Requested (INR)\", 1000, 1000000, value=20000, step=1000)\n",
    "credit_score = st.sidebar.slider(\"Credit Score\", 300, 850, 600)\n",
    "\n",
    "# Preprocessing the data\n",
    "X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']].dropna()  # Removing rows with NaN values for training\n",
    "y = df['Default'].dropna()\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Train the model once using a cached function\n",
    "@st.cache_resource\n",
    "def train_model():\n",
    "    model = LogisticRegression()\n",
    "    model.fit(X_scaled, y)\n",
    "    return model\n",
    "\n",
    "# Load the pre-trained model\n",
    "model = train_model()\n",
    "\n",
    "# Input from sidebar to make predictions\n",
    "input_data = pd.DataFrame([[age, income, loan, credit_score]], \n",
    "                          columns=['Age', 'Income', 'LoanAmount', 'CreditScore'])\n",
    "input_scaled = scaler.transform(input_data)\n",
    "\n",
    "# Prediction probability\n",
    "default_prob = model.predict_proba(input_scaled)[0][1]\n",
    "credit_risk_score = (1 - default_prob) * 100\n",
    "\n",
    "# Display Results\n",
    "st.subheader(\"📊 Prediction Results\")\n",
    "st.write(f\"**Probability of Default:** {default_prob:.2f}\")\n",
    "st.write(f\"**Credit Risk Score:** {credit_risk_score:.1f} / 100\")\n",
    "\n",
    "# Display Risk level based on the credit risk score\n",
    "if credit_risk_score > 80:\n",
    "    st.success(\"🟢 Low Risk: Good to Approve\")\n",
    "elif credit_risk_score > 50:\n",
    "    st.warning(\"🟡 Medium Risk: Evaluate Further\")\n",
    "else:\n",
    "    st.error(\"🔴 High Risk: Likely to Default\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cddb2455-677e-4802-97d6-9caed7635c24",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
