import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import os

# Streamlit page config
st.set_page_config(page_title="Crime Clustering Analysis", layout="wide")
st.title("🚔 Crime Clustering Analysis with DBSCAN")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("crime.csv")
    df = df.dropna(subset=["Longitude", "Latitude"]).copy()
    return df

df = load_data()

# Clustering
x = df[["Longitude", "Latitude"]].dropna()
ss = StandardScaler()
x_s = ss.fit_transform(x)

db = DBSCAN(eps=1.0, min_samples=2)
db.fit(x_s)

labels = db.labels_
score = silhouette_score(x_s, labels)
df["cluster"] = labels

# Sidebar with stats
st.sidebar.header("📈 Clustering Info")
st.sidebar.write(f"**Silhouette Score:** {score:.3f}")
st.sidebar.write(f"**Number of Clusters:** {len(set(labels)) - (1 if -1 in labels else 0)}")
st.sidebar.write(f"**Noise Points:** {list(labels).count(-1)}")

# Insights
st.subheader("📊 Insights")
col1, col2 = st.columns(2)
with col1:
    st.write("**Cluster Crime Counts:**")
    st.write(df["cluster"].value_counts())

with col2:
    st.write("**Most Common Crime per Cluster:**")
    most_common = df.groupby("cluster")["Crime type"].agg(lambda x: x.value_counts().index[0])
    st.write(most_common)

# Map Button - Opens in Browser
if st.button("🌍 Open Folium Map in Browser"):
    m = folium.Map(location=[x["Latitude"].mean(), x["Longitude"].mean()], zoom_start=13)

    # Heatmap Layer
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=8, blur=6, max_zoom=13).add_to(m)

    # Cluster Layer
    colors = ['red', 'blue', 'green', 'purple', 'orange', 
              'darkred', 'lightblue', 'lightgreen', 'beige']
    cluster_layer = folium.FeatureGroup(name="Crime Clusters")
    for _, row in df.iterrows():
        popup_text = (
            f"<b>Crime ID:</b> {row['Crime ID']}<br>"
            f"<b>Month:</b> {row['Month']}<br>"
            f"<b>Reported by:</b> {row['Reported by']}<br>"
            f"<b>Location:</b> {row['Location']}<br>"
            f"<b>Crime type:</b> {row['Crime type']}<br>"
            f"<b>Last outcome:</b> {row['Last outcome category']}<br>"
            f"<b>Cluster:</b> {row['cluster']}"
        )
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            popup=folium.Popup(popup_text, max_width=300),
            color=colors[row['cluster'] % len(colors)] if row['cluster'] != -1 else "black",
            fill=True
        ).add_to(cluster_layer)

    cluster_layer.add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Save and open in browser
    map_file = "crime_clusters_heatmap.html"
    m.save(map_file)
    webbrowser.open('file://' + os.path.realpath(map_file))

# Pairplot Button - Shows in App
if st.button("📈 Show Pairplot"):
    fig = sns.pairplot(df[['Longitude', 'Latitude', 'cluster']], hue="cluster", palette="husl")
    plt.suptitle("Pairplot of DBSCAN Clusters", y=1.02)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("📌 **Developed by:** Your Name | 🔍 DBSCAN | 📍 Folium | 📊 Seaborn")
