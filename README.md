# 🚔 Crime Clustering Analysis with DBSCAN
This project detects and visualizes crime hotspots using DBSCAN clustering on geospatial data. The aim is to make crime data more interpretable and actionable through interactive maps and statistical insights.

## 📂 Dataset
Source: Official UK Police data (May 2025)

Format: CSV file (crime.csv)

Key Fields: Longitude, Latitude, Crime type, Location, Month, Reported by, Last outcome category

## 🛠 Technologies Used
Python – Core programming

Pandas – Data cleaning & manipulation

Seaborn / Matplotlib – Data visualization

DBSCAN (from sklearn) – Clustering algorithm

Folium – Interactive maps

Streamlit – Web app interface

Silhouette Score – Cluster quality evaluation

## ⚙️ How It Works
Data Preprocessing

Missing location values are removed.

Longitude & Latitude are scaled using StandardScaler.

Clustering

DBSCAN groups nearby crimes into clusters.

Noise points are marked as -1.

Silhouette Score is calculated to evaluate cluster quality.

Visualization

Interactive Folium Map with:

Cluster markers with popup crime details

Heatmap layer for intensity visualization

Pairplot to see how clusters are formed.

Streamlit Interface

Displays insights such as:

Silhouette Score

Number of clusters & noise points

Cluster crime counts

Most common crime type per cluster

Buttons to:

Open Folium Map in browser

Show Seaborn pairplot

## 📸 Features Demo
Folium Crime Map
Clickable markers with full crime details

Layer control for switching between Clusters and Heatmap

Pairplot Visualization
View how clusters separate in longitude/latitude space

## 🚀 Running the Project
Install dependencies:


pip install pandas seaborn matplotlib scikit-learn folium streamlit numpy
Run Streamlit app:


streamlit run app.py
Interact:

View insights in the Streamlit dashboard

Open interactive Folium map in your browser

Visualize cluster patterns

## 📌 Author
Muhammad Sarab Rafique
🔍 DBSCAN | 📍 Folium | 📊 Seaborn

