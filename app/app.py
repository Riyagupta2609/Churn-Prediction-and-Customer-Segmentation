
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import openpyxl

# ── Page config ──
st.set_page_config(page_title="Churn Predictor",page_icon="^^", layout="wide")

# ── Load data ──
@st.cache_data
def load_data():
    try:
        df=pd.read_excel(".ipynb_checkpoints/Telco_customer_churn.xlsx")
        return df
    except FileNotFoundError:
        st.error("Run save_model.py first to generate model/churn_data.pkl")
        st.stop()

@st.cache_resource
def load_model():
    try:
        with open("model/churn_model.pkl", "rb") as f:
            model = pickle.load(f)
            with open("model/encoder_columns.pkl", "rb") as f:
                cols = pickle.load(f)
                return model, cols
    except FileNotFoundError:
        return None, None

df = load_data()
model, enc_cols = load_model()
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(0)

features = df[["Tenure Months", "Monthly Charges", "Total Charges"]]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster Segment"] = kmeans.fit_predict(features_scaled).astype(str)
# ════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════
st.sidebar.title("Churn Prediction and Customer Segmentation")
page = st.sidebar.radio("Go to", ["Overview", "Segments", "Predict"])

# ════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════
if page == "Overview":
    st.title("Overview")
    st.markdown("Churn across the telecom dataset.")
    st.divider()

    # ── 4 KPI metrics ──
    total     = len(df)
    churned   = df["Churn Value"].sum()
    churn_pct = df["Churn Value"].mean() * 100
    avg_charge= df["Monthly Charges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  f"{total:,}")
    c2.metric("Churned",          f"{int(churned):,}")
    c3.metric("Churn Rate",       f"{churn_pct:.1f}%")
    c4.metric("Avg Monthly Charge", f"${avg_charge:.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    # ── Churn by contract ──
    with col1:
        st.subheader("Churn rate by contract")
        gb = (df.groupby("Contract")["Churn Value"]
                .mean().mul(100).round(1).reset_index()
                .rename(columns={"Churn Value": "Churn Rate (%)"}))
        fig = px.bar(gb, x="Contract", y="Churn Rate (%)",
                     color="Churn Rate (%)",
                     color_continuous_scale=["green","orange","red"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False,
                          yaxis=dict(range=[0, 80]),
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Churn donut ──
    with col2:
        st.subheader("Churned vs Retained")
        counts = df["Churn Label"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig2 = px.pie(counts, values="Count", names="Status",
                      hole=0.5,
                      color="Status",
                      color_discrete_map={"Yes": "#f87171", "No": "#34d399"})
        fig2.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tenure histogram ──
    st.subheader("Tenure distribution by churn")
    fig3 = px.histogram(df, x="Tenure Months", color="Churn Label",
                        nbins=30, barmode="overlay",
                        color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
                        labels={"Tenure Months": "Tenure (months)",
                                "Churn Label": "Churned"})
    fig3.update_layout(margin=dict(t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════
# PAGE 2 — SEGMENTS
elif page == "Segments":
    st.title("Customer Segments")
    st.markdown("Customers based on their churn rate")
    col1,col2,col3 = st.columns(3)

    with col1:
        st.success("""
###  Loyal Customers

 Long tenure

 Low churn risk

 High lifetime value

**Recommendation**

Reward with loyalty programs.
""")

    with col2:
        st.warning("""
###  Regular Customers

 Moderate tenure

 Average spending

 Medium churn risk

**Recommendation**

Offer personalized promotions.
""")

    with col3:
        st.error("""
###  At-Risk Customers

 Short tenure

 High monthly bills

 High churn probability

**Recommendation**

Immediate retention campaign.
""")

    st.divider()

    # ── Segment summary table ──
    seg = (df.groupby("Cluster Segment").agg(
        Customers    = ("CustomerID", "count"),
        Churn_Rate   = ("Churn Value", "mean"),
        Avg_Tenure   = ("Tenure Months", "mean"),
        Avg_Monthly  = ("Monthly Charges", "mean"),
        Avg_CLTV     = ("CLTV", "mean"),
    ).reset_index())
    seg["Churn_Rate"] = (seg["Churn_Rate"] * 100).round(1).astype(str) + "%"
    seg["Avg_Tenure"]  = seg["Avg_Tenure"].round(1)
    seg["Avg_Monthly"] = seg["Avg_Monthly"].round(2)
    seg["Avg_CLTV"]    = seg["Avg_CLTV"].round(0).astype(int)
    seg.columns = ["Segment","Customers","Churn Rate",
                   "Avg Tenure (mo)","Avg Monthly ($)","Avg CLTV ($)"]
    st.dataframe(seg, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    # ── Churn rate per segment ──
    with col1:
        st.subheader("Churn rate per segment")
        seg_plot = (df.groupby("Cluster Segment")["Churn Value"]
                      .mean().mul(100).round(1).reset_index()
                      .rename(columns={"Churn Value": "Churn Rate (%)"}))
        fig = px.bar(seg_plot, x="Cluster Segment", y="Churn Rate (%)",
                     color="Churn Rate (%)",
                     color_continuous_scale=["green","orange","red"],
                     text="Churn Rate (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False,
                          yaxis=dict(range=[0, 80]),
                          margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Segment size pie ──
    with col2:
        st.subheader("Segment size")
        size = df["Cluster Segment"].value_counts().reset_index()
        size.columns = ["Cluster Segment", "Count"]
        fig2 = px.pie(size, values="Count", names="Cluster Segment",
                      color_discrete_sequence=["#f87171","#fbbf24","#34d399"])
        fig2.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Tenure vs Monthly Charges by Segment")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="Monthly Charges",
        y="Tenure Months",
        hue="Cluster Segment",
        palette="Spectral",
        ax=ax
    )

    ax.set_title("Customer Segments")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Tenure Months")

    st.pyplot(fig)

# ════════════════════════════════════════════
# PAGE 3 — PREDICT
# ════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict Churn")
    st.markdown("Fill in the customer details to get a churn probability.")
    st.divider()

    col_form, col_result = st.columns(2)

    with col_form:
        st.subheader("Customer details")
        gender       = st.selectbox("Gender",           ["Male", "Female"])
        senior       = st.selectbox("Senior citizen",   ["No", "Yes"])
        partner      = st.selectbox("Partner",          ["No", "Yes"])
        dependents   = st.selectbox("Dependents",       ["No", "Yes"])
        tenure       = st.slider("Tenure (months)", 0, 72, 12)
        contract     = st.selectbox("Contract",
                                    ["Month-to-month", "One year", "Two year"])
        internet     = st.selectbox("Internet service",
                                    ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech support",
                                    ["No", "Yes", "No internet service"])
        payment      = st.selectbox("Payment method",
                                    ["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)",
                                     "Credit card (automatic)"])
        monthly      = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, 0.5)
        total        = st.number_input("Total charges ($)", 0.0, 9000.0,
                                       value=round(monthly * tenure, 2))

        predict = st.button("Predict", use_container_width=True)

    with col_result:
        st.subheader("Result")

        def rule_prob(contract, tenure, monthly, internet, tech_support, senior):
            base = {"Month-to-month": 0.58,
                    "One year": 0.18, "Two year": 0.07}[contract]
            t_f  = max(0.3, 1 - tenure / 80)
            adds = (0.10 if internet == "Fiber optic" else 0.0) + \
                   (0.08 if tech_support == "No" else -0.05) + \
                   (0.08 if senior == "Yes" else 0.0) + \
                   (0.05 if monthly > 80 else 0.0)
            return round(min(max(base * t_f + adds, 0.02), 0.97), 3)

        if predict:
            prob = None
            if model is not None and enc_cols is not None:
                try:
                    row = pd.DataFrame([{
                        "Gender": gender,
                        "Senior Citizen": 1 if senior == "Yes" else 0,
                        "Partner": partner, "Dependents": dependents,
                        "Tenure Months": tenure,
                        "Phone Service": "Yes",
                        "Multiple Lines": "No",
                        "Internet Service": internet,
                        "Online Security": "No",
                        "Online Backup": "No",
                        "Device Protection": "No",
                        "Tech Support": tech_support,
                        "Streaming TV": "No",
                        "Streaming Movies": "No",
                        "Contract": contract,
                        "Paperless Billing": "Yes",
                        "Payment Method": payment,
                        "Monthly Charges": monthly,
                        "Total Charges": total,
                    }])
                    row_enc = pd.get_dummies(row, drop_first=True)
                    row_enc = row_enc.reindex(columns=enc_cols, fill_value=0)
                    prob = model.predict_proba(row_enc)[0][1]
                except Exception:
                    prob = None

            if prob is None:
                prob = rule_prob(contract, tenure, monthly,
                                 internet, tech_support, senior)

            pct   = round(prob * 100, 1)
            label = ("🔴 High Risk"   if prob > 0.6 else
                     "🟡 Medium Risk" if prob > 0.35 else "🟢 Low Risk")
            color = ("#f87171" if prob > 0.6 else
                     "#fbbf24" if prob > 0.35 else "#34d399")

            # Big probability number
            st.markdown(f"""
            <div style='text-align:center;padding:24px;background:#1a1d27;
                        border-radius:12px;border:1px solid #2a2d3e;margin-bottom:16px'>
                <div style='font-size:3rem;font-weight:700;color:{color}'>{pct}%</div>
                <div style='font-size:1.1rem;color:{color};margin-top:4px'>{label}</div>
                <div style='font-size:0.8rem;color:#6b7280;margin-top:6px'>churn probability</div>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(int(pct))

            st.divider()

            # Top reasons
            st.markdown("**Why this score:**")
            if contract == "Month-to-month":
                st.warning("Month-to-month contract — 3× higher churn risk")
            if tenure <= 12:
                st.warning(f"Only {tenure} months tenure — new customers churn more")
            if internet == "Fiber optic":
                st.info("Fiber optic users churn more than DSL")
            if tech_support == "No":
                st.info("No tech support — linked to higher churn")
            if monthly > 80:
                st.info(f"${monthly:.0f}/mo is above average — price sensitivity")
            if senior == "Yes":
                st.info("Senior citizens show higher churn rates")
            if prob <= 0.35:
                st.success("No major risk factors — customer looks stable")

            st.divider()

            # Recommendation
            st.markdown("**What to do:**")
            if prob > 0.6:
                st.error("Priority retention call. Offer annual contract discount + free tech support.")
            elif prob > 0.35:
                st.warning("Send loyalty offer. Consider a billing switch incentive.")
            else:
                st.success("Low risk. Standard engagement — no action needed.")
        else:
            st.markdown(
                "<div style='color:#6b7280;margin-top:40px;text-align:center'>"
                "Fill in the form and click Predict</div>",
                unsafe_allow_html=True
            )