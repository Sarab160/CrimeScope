import pandas as pd
from seaborn import heatmap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import webbrowser
import os
import streamlit as st

st.set_page_config(page_title="Crime Clustering Analysis", layout="wide")
st.title("🚔 Crime Clustering Analysis with DBSCAN")
df=pd.read_csv("crime.csv")

print(df.info())

df = df.dropna(subset=["Longitude", "Latitude"]).copy()

x=df[["Longitude","Latitude"]].dropna()

ss=StandardScaler()
x_s=ss.fit_transform(x)

db=DBSCAN(eps=1.0,min_samples=2)
db.fit(x_s)

labels=db.labels_
score=silhouette_score(x_s,labels)
# print(score)

df["cluster"]=labels

st.sidebar.header("📈 Clustering Info")
st.sidebar.write(f"**Silhouette Score:** {score:.3f}")
st.sidebar.write(f"**Number of Clusters:** {len(set(labels)) - (1 if -1 in labels else 0)}")
st.sidebar.write(f"**Noise Points:** {list(labels).count(-1)}")

st.subheader("📊 Insights")
col1, col2 = st.columns(2)
with col1:
    st.write("**Cluster Crime Counts:**")
    st.write(df["cluster"].value_counts())

with col2:
    st.write("**Most Common Crime per Cluster:**")
    most_common = df.groupby("cluster")["Crime type"].agg(lambda x: x.value_counts().index[0])
    st.write(most_common)
if st.button("🌍 Open Folium Map in Browser"):
    m=folium.Map(location=[x["Latitude"].mean(),x["Longitude"].mean()],zoom_start=13)

    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows() if not np.isnan(row['Latitude']) and not np.isnan(row['Longitude'])]
    HeatMap(heat_data, radius=8, blur=6, max_zoom=13).add_to(m)

    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'darkred', 'lightblue', 'lightgreen', 'beige'
    ]
    cluster_layer = folium.FeatureGroup(name="Crime Clusters")
    for _, row in df.iterrows():
        popup_text = (
            f"<b>"
            f"Crime ID: {row['Crime ID']}<br>"
            f"Month: {row['Month']}<br>"
            f"Reported by: {row['Reported by']}<br>"
            f"Location: {row['Location']}<br>"
            f"Crime type: {row['Crime type']}<br>"
            f"Last outcome: {row['Last outcome category']}<br>"
            f"Cluster: {row['cluster']}"
            f"</b>"
    )

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        popup=folium.Popup(popup_text, max_width=300),
        color=colors[row['cluster'] % len(colors)] if row['cluster'] != -1 else "black",
        fill=True
    ).add_to(cluster_layer)

    heat_layer = folium.FeatureGroup(name="Heatmap")
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=8, blur=6, max_zoom=13).add_to(heat_layer)

    cluster_layer.add_to(m)
    heat_layer.add_to(m)

    folium.LayerControl(position="topright",collapsed=False).add_to(m)

    file="crime_clusters_heatmap.html"
    m.save(file)

    webbrowser.open('file://' + os.path.realpath(file))

if st.button("📈 Show Pairplot"):
    fig = sns.pairplot(df[['Longitude', 'Latitude', 'cluster']], hue="cluster", palette="husl")
    plt.suptitle("Pairplot of DBSCAN Clusters", y=1.02)
    st.pyplot(fig)


st.markdown("---")
st.markdown("📌 **Developed by:** Muhammad Sarab Rafique | 🔍 DBSCAN | 📍 Folium | 📊 Seaborn")


