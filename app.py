import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Page Settings
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("### SkillCraft Technology - Task 02")

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Show Dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Sidebar
st.sidebar.header("⚙️ Dashboard Controls")

clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

# Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# KMeans
kmeans = KMeans(
    n_clusters=clusters,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Clusters", clusters)
col3.metric("Average Age", round(df['Age'].mean(), 1))

# Cluster Graph
st.subheader("🎯 Customer Segmentation")

fig = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color=df['Cluster'].astype(str),
    size='Age',
    hover_data=['Gender', 'Age'],
    title='Customer Clusters'
)

st.plotly_chart(fig, use_container_width=True)

# Cluster Summary
st.subheader("📈 Cluster Summary")

summary = df.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean()

st.dataframe(summary)

# Income Distribution
st.subheader("💰 Annual Income Distribution")

income_fig = px.histogram(
    df,
    x='Annual Income (k$)',
    nbins=20,
    title='Income Distribution'
)

st.plotly_chart(income_fig, use_container_width=True)

# Spending Score Distribution
st.subheader("🛒 Spending Score Distribution")

spend_fig = px.histogram(
    df,
    x='Spending Score (1-100)',
    nbins=20,
    title='Spending Score Distribution'
)

st.plotly_chart(spend_fig, use_container_width=True)

# Gender Analysis
st.subheader("👨‍🦱👩 Gender Distribution")

gender_fig = px.pie(
    df,
    names='Gender',
    title='Male vs Female Customers'
)

st.plotly_chart(gender_fig, use_container_width=True)

# Customer Insights
st.subheader("💡 Customer Insights")

high_spenders = len(
    df[df['Spending Score (1-100)'] > 70]
)

high_income = len(
    df[df['Annual Income (k$)'] > 70]
)

st.success(
    f"🔥 High Spending Customers: {high_spenders}"
)

st.info(
    f"💰 High Income Customers: {high_income}"
)

st.balloons()

st.markdown("---")
st.markdown(
    "Developed for **SkillCraft Technology - Task 02** 🚀"
)