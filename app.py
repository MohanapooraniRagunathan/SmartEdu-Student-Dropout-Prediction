import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("dropout_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

categorical_columns = [
    "Gender", 
    "Family_Income", 
    "Father_Education", 
    "Mother_Education", 
    "Disciplinary_Issues", 
    "Internet_Access", 
    "Parental_Support"
]

st.image("logo.png", width=120) 
st.title("📚 SmartEdu+ : Student Dropout Prediction Dashboard")

st.sidebar.title("🔍 Navigation")
st.sidebar.info("Use the tabs to explore prediction, explanations, visualizations, and feedback.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Prediction", 
    "🔎 SHAP Explanations", 
    "📈 Visualizations", 
    "📝 Feedback"
])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.header("📊 Student Dropout Prediction")

    age = st.number_input("Age", min_value=10, max_value=25, value=17)
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    father_edu = st.selectbox("Father's Education", ["Unknown", "Primary", "Secondary", "Graduate"])
    mother_edu = st.selectbox("Mother's Education", ["Unknown", "Primary", "Secondary", "Graduate"])
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    marks_prev_exam = st.slider("Marks in Previous Exam", 0, 100, 70)
    study_hours = st.slider("Study Hours Per Day", 0, 12, 2)
    disciplinary = st.selectbox("Disciplinary Issues", ["Yes", "No"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])

    gender_map = {"Male": 0, "Female": 1}
    disciplinary_map = {"No": 0, "Yes": 1}
    internet_map = {"No": 0, "Yes": 1}
    family_income_map = {"Low": 0, "Medium": 1, "High": 2}
    education_map = {"Unknown": 0, "Primary": 1, "Secondary": 2, "Graduate": 3}
    parental_support_map = {"Low": 0, "Medium": 1, "High": 2}

    new_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_map[gender],
        "Family_Income": family_income_map[family_income],
        "Father_Education": education_map[father_edu],
        "Mother_Education": education_map[mother_edu],
        "Attendance_Percentage": attendance,
        "Marks_Previous_Exam": marks_prev_exam,
        "Study_Hours_Per_Day": study_hours,
        "Disciplinary_Issues": disciplinary_map[disciplinary],
        "Internet_Access": internet_map[internet_access],
        "Parental_Support": parental_support_map[parental_support]
    }])

    if st.button("🔮 Predict Dropout Risk"):
        prediction = model.predict(new_data)[0]
        probability = model.predict_proba(new_data)[0][1] * 100

        if prediction == 1:
            st.error(f"⚠️ High Risk of Dropout! (Probability: {probability:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Dropout (Probability: {probability:.2f}%)")

# ---------------- TAB 2: SHAP Explanations ----------------

with tab2:
    st.header("🔎 SHAP Explanations")
    st.write("Explainable AI (SHAP) helps us understand *why* the model predicted a student as High or Low risk.")

    data = pd.read_csv("E:/3rd yr MINI PROJ Dataset/datasets/student_dropout_300.csv")

    if "Student_ID" in data.columns:
        student_ids = data["Student_ID"]
        data = data.drop("Student_ID", axis=1)
    else:
        student_ids = pd.Series(range(len(data)))

    X = data.drop("Dropout", axis=1)
    y = data["Dropout"]

    X["Gender"] = X["Gender"].map(gender_map)
    X["Family_Income"] = X["Family_Income"].map(family_income_map)
    X["Father_Education"] = X["Father_Education"].map(education_map)
    X["Mother_Education"] = X["Mother_Education"].map(education_map)
    X["Disciplinary_Issues"] = X["Disciplinary_Issues"].map(disciplinary_map)
    X["Internet_Access"] = X["Internet_Access"].map(internet_map)
    X["Parental_Support"] = X["Parental_Support"].map(parental_support_map)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    selected_id = st.selectbox("🎓 Select Student by ID", student_ids)
    student_idx = student_ids[student_ids == selected_id].index[0]
    selected_student = X.iloc[[student_idx]]
    profile_data = data.drop("Dropout", axis=1).iloc[student_idx]

    pred = model.predict(selected_student)[0]
    prob = model.predict_proba(selected_student)[0][1] * 100

    st.markdown("### 👤 Student Profile & 🎯 Dropout Risk")
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("#### 📑 Student Details")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("🧑 Age", profile_data["Age"])
            st.metric("⚥ Gender", profile_data["Gender"])
            st.metric("💰 Family Income", profile_data["Family_Income"])
            st.metric("🎓 Father Edu.", profile_data["Father_Education"])

        with c2:
            st.metric("🎓 Mother Edu.", profile_data["Mother_Education"])
            st.progress(int(profile_data["Attendance_Percentage"]))
            st.caption(f"📊 Attendance: {profile_data['Attendance_Percentage']}%")
            st.progress(int(profile_data["Marks_Previous_Exam"]))
            st.caption(f"📝 Marks: {profile_data['Marks_Previous_Exam']}%")
            st.metric("⏱ Study Hours", profile_data["Study_Hours_Per_Day"])

    with col2:
        import plotly.graph_objects as go
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Dropout Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob > 50 else "green"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 100], "color": "pink"}
                ]
            }
        ))
        gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        if pred == 1:
            st.error(f"⚠️ Student {selected_id}: High Risk of Dropout ({prob:.2f}%)")
        else:
            st.success(f"✅ Student {selected_id}: Low Risk of Dropout ({prob:.2f}%)")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Feature Importance")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig1)

    with col4:
        st.subheader("🎯 Individual SHAP Explanation")
        shap_values_for_student = shap_values[student_idx]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, shap_values_for_student, X.iloc[student_idx, :], show=False
        )
        st.pyplot(fig2)

    st.markdown("### 🔑 Top 3 Factors Driving Prediction")
    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value": shap_values[student_idx]
    }).assign(abs_shap=lambda df: df["SHAP Value"].abs()).sort_values(by="abs_shap", ascending=False)

    cols = st.columns(3)
    for i, (_, row) in enumerate(shap_df.head(3).iterrows()):
        with cols[i]:
            if row["SHAP Value"] > 0:
                st.error(f"⬆️ {row['Feature']}")
            else:
                st.success(f"⬇️ {row['Feature']}")

# ---------------- TAB 3: Visualizations ----------------
import plotly.express as px
import plotly.graph_objects as go

with tab3:
    st.header("📈 Dataset Visualizations")
    st.write("Explore overall patterns and trends in the student dataset.")

    data_viz = pd.read_csv("E:/3rd yr MINI PROJ Dataset/datasets/student_dropout_300.csv")

    st.subheader("📊 Dropout Distribution")
    dropout_counts = data_viz["Dropout"].value_counts().reset_index()
    dropout_counts.columns = ["Dropout", "Count"]
    fig_pie = px.pie(
        dropout_counts, names="Dropout", values="Count",
        color="Dropout", color_discrete_map={0: "green", 1: "red"},
        title="Dropout vs Non-Dropout"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📝 Average Marks vs Dropout Status")
    marks_avg = data_viz.groupby("Dropout")["Marks_Previous_Exam"].mean().reset_index()

    fig_marks_avg = px.bar(
        marks_avg, x="Dropout", y="Marks_Previous_Exam",
        color="Dropout",
        color_discrete_map={0: "green", 1: "red"},
        text_auto=".1f",
        title="Average Marks (Stayed vs Dropped)"
    )
    fig_marks_avg.update_xaxes(tickvals=[0, 1], ticktext=["Stayed", "Dropped"])
    fig_marks_avg.update_layout(yaxis_title="Average Marks (%)")
    st.plotly_chart(fig_marks_avg, use_container_width=True)

    st.subheader("⏱ Study Hours per Day vs Dropout")
    fig_study = px.histogram(
        data_viz, x="Study_Hours_Per_Day", color="Dropout",
        barmode="overlay", nbins=10,
        color_discrete_map={0: "blue", 1: "red"},
        title="Study Hours Distribution (Dropout vs Non-Dropout)"
    )
    st.plotly_chart(fig_study, use_container_width=True)

    st.subheader("💰 Family Income vs Dropout")
    income_dropout = (
        data_viz.groupby(["Family_Income", "Dropout"]).size()
        .reset_index(name="Count")
    )
    income_total = income_dropout.groupby("Family_Income")["Count"].transform("sum")
    income_dropout["Percentage"] = (income_dropout["Count"] / income_total) * 100

    fig_income = px.bar(
        income_dropout, 
        x="Family_Income", y="Percentage",
        color="Dropout", barmode="stack",
        title="Dropout % by Family Income (Stacked)",
        color_discrete_map={0: "green", 1: "red"},
        text_auto=".1f"
    )
    fig_income.update_layout(yaxis_title="Percentage of Students")
    st.plotly_chart(fig_income, use_container_width=True)

# ---------------- TAB 4: Feedback ----------------
import os
import pandas as pd
import streamlit as st

dropout_file = "dropout_reasons.csv"

with tab4:
    st.header("📝 Dropout Feedback")
    st.write("Tell us the reason why you feel you might drop out. Your feedback will help us support you better.")

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    with st.form("dropout_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        reason = st.text_area("What is the main reason you're considering dropping out?")

        submit = st.form_submit_button("📩 Submit")

        if submit:
            if not name or not email or not reason:
                st.error("Please fill in all the fields.")
            else:
                new_entry = pd.DataFrame([{
                    "Name": name,
                    "Email": email,
                    "Reason": reason
                }])

                if os.path.exists(dropout_file):
                    existing = pd.read_csv(dropout_file)
                    updated = pd.concat([existing, new_entry], ignore_index=True)
                else:
                    updated = new_entry

                updated.to_csv(dropout_file, index=False)
                st.session_state.submitted = True

    if st.session_state.submitted:
        st.success("✅ Thank you! Your reason has been recorded.")
        
        st.session_state.submitted = False

   
    st.markdown("---")
    st.subheader("📂 Previous Dropout Reasons")

    if os.path.exists(dropout_file):
        reasons_data = pd.read_csv(dropout_file)
        if reasons_data.empty:
            st.info("No dropout reasons submitted yet.")
        else:
            for i, row in reasons_data.iterrows():
                st.markdown(f"**{row['Name']} ({row['Email']})**")
                st.write(f"🗒 Reason: {row['Reason']}")
                st.markdown("---")
    else:
        st.info("No dropout reasons submitted yet.")
