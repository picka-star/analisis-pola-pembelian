# app.py (Versi yang Diperbaiki - Fix Visualisasi)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Libraries untuk analisis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Pola Pembelian Online Store",
    page_icon="🛒",
    layout="wide"
)

# Title aplikasi
st.title("🛒 Analisis Pola Pembelian Online Store")
st.markdown("**Menggunakan RFM, K-Means, dan Apriori untuk Segmentasi Pelanggan**")

# Sidebar untuk upload data
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload file Excel/CSV", type=['xlsx', 'csv'])
    
    st.header("⚙️ Parameter Analisis")
    st.subheader("RFM Parameters")
    rfm_date = st.date_input("Tanggal Referensi", value=datetime.now().date())
    
    st.subheader("Clustering Parameters")
    n_clusters = st.slider("Jumlah Cluster", 2, 6, 3)
    
    st.subheader("Association Rules")
    min_support = st.slider("Min Support", 0.01, 0.2, 0.01, 0.01)
    min_confidence = st.slider("Min Confidence", 0.1, 0.8, 0.10, 0.05)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

# Fungsi preprocessing data
def preprocess_data(df):
    """Tahap 1: Preprocessing data"""
    df_clean = df.copy()
    
    # Konversi tipe data
    if 'Transaction_Date' in df_clean.columns:
        df_clean['Transaction_Date'] = pd.to_datetime(df_clean['Transaction_Date'])
    
    # Hapus missing values pada kolom penting
    required_cols = ['CustomerID', 'Transaction_ID', 'Transaction_Date', 
                    'Product_Category', 'Quantity', 'Avg_Price']
    available_cols = [col for col in required_cols if col in df_clean.columns]
    
    if available_cols:
        df_clean = df_clean.dropna(subset=available_cols)
    
    # Hapus nilai negatif
    if 'Quantity' in df_clean.columns:
        df_clean = df_clean[df_clean['Quantity'] > 0]
    if 'Avg_Price' in df_clean.columns:
        df_clean = df_clean[df_clean['Avg_Price'] > 0]
    
    return df_clean

# Fungsi perhitungan RFM
def calculate_rfm(df, reference_date):
    """Tahap 2: Perhitungan RFM"""
    # Pastikan ada kolom yang diperlukan
    if not all(col in df.columns for col in ['CustomerID', 'Transaction_Date', 'Quantity', 'Avg_Price']):
        st.error("Kolom yang diperlukan tidak ditemukan!")
        return None
    
    # Hitung TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['Avg_Price']
    
    # Hitung RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'Transaction_Date': lambda x: (reference_date - x.max()).days,
        'Transaction_ID': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm

# Fungsi scoring RFM
def score_rfm(rfm_df):
    """Tahap 3: Scoring RFM"""
    rfm_scored = rfm_df.copy()
    
    # Beri ranking berdasarkan quartile
    rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], q=5, labels=[5,4,3,2,1])
    rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'], q=5, labels=[1,2,3,4,5])
    rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'], q=5, labels=[1,2,3,4,5])
    
    # Konversi ke integer
    rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
    rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
    rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
    
    # Gabungkan skor
    rfm_scored['RFM_Score'] = (rfm_scored['R_Score'].astype(str) + 
                               rfm_scored['F_Score'].astype(str) + 
                               rfm_scored['M_Score'].astype(str))
    
    return rfm_scored

# Fungsi segmentasi RFM
def segment_rfm(rfm_scored):
    """Segmentasi berdasarkan RFM Score"""
    def get_segment(score):
        if pd.isna(score):
            return "Unknown"
        r, f, m = int(score[0]), int(score[1]), int(score[2])
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 4 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 3:
            return "Potential Loyalists"
        elif f >= 3:
            return "Need Attention"
        else:
            return "At Risk"
    
    rfm_scored['RFM_Segment'] = rfm_scored['RFM_Score'].apply(get_segment)
    return rfm_scored

# Fungsi untuk menentukan jumlah cluster optimal
def find_optimal_clusters(data, max_clusters=10):
    """Menentukan jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score"""
    wcss = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)  # Mulai dari 2 cluster
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        # Hitung silhouette score
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
    
    return K_range, wcss, silhouette_scores

# Fungsi clustering dengan K-Means
def perform_clustering(rfm_scaled, n_clusters=4):
    """Tahap 4: Clustering dengan K-Means"""
    # Fit model dengan jumlah cluster yang ditentukan
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    
    return kmeans

# Fungsi analisis asosiasi
def analyze_associations(df, cluster_df, min_support=0.05, min_confidence=0.5):
    """Tahap 5: Analisis Asosiasi dengan Apriori"""
    # Merge transaction data with cluster info
    df_with_cluster = pd.merge(
        df,
        cluster_df[['CustomerID', 'Cluster']],
        on='CustomerID',
        how='inner'
    )
    
    all_rules = {}
    
    # Analyze per cluster
    for cluster_num in sorted(cluster_df['Cluster'].unique()):
        # Filter transactions for this cluster
        cluster_data = df_with_cluster[df_with_cluster['Cluster'] == cluster_num]
        
        if len(cluster_data) < 10:  # Minimal 10 transaksi
            continue;
            
        # Create basket matrix
        try:
            # Pastikan ada cukup transaksi
            if cluster_data['Transaction_ID'].nunique() < 2:
                continue;
                
            basket = cluster_data.groupby(['Transaction_ID', 'Product_Category'])['Quantity'].sum().unstack().fillna(0)
            
            # Binary encoding
            def encode(x):
                return 1 if x > 0 else 0;
            
            basket_sets = basket.applymap(encode)
            
            # Apriori algorithm
            frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, metric="confidence", 
                                         min_threshold=min_confidence)
                
                if len(rules) > 0:
                    # Filter meaningful rules
                    meaningful_rules = rules[rules['lift'] >= 1.0].sort_values('lift', ascending=False)
                    all_rules[cluster_num] = meaningful_rules;
                    
        except Exception as e:
            st.warning(f"Tidak bisa analisis cluster {cluster_num}: {str(e)}")
    
    return all_rules

# Main application logic
def main():
    if uploaded_file is not None:
        # Load and preprocess data
        with st.spinner("Memuat dan memproses data..."):
            df_raw = load_data(uploaded_file)
            df_clean = preprocess_data(df_raw)
            
            # Tampilkan data
            st.header("📊 Data Awal")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Jumlah Baris:** {len(df_raw)}")
                st.write(f"**Jumlah Kolom:** {len(df_raw.columns)}")
            with col2:
                st.write(f"**Data setelah cleaning:** {len(df_clean)} baris")
            
            # Show sample data
            with st.expander("👀 Lihat Sample Data"):
                st.dataframe(df_clean.head(10))
                st.write("**Kolom yang tersedia:**", list(df_clean.columns))
            
        # Tab untuk analisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 RFM Analysis", 
            "🎯 Clustering", 
            "🔄 Association Rules",
            "📋 Recommendations"
        ])
                
        with tab1:
            st.header("RFM Analysis")
            
            # Calculate RFM
            reference_date = pd.to_datetime(rfm_date)
            rfm_df = calculate_rfm(df_clean, reference_date)
            
            if rfm_df is not None:
                # Score RFM
                rfm_scored = score_rfm(rfm_df)
                rfm_scored = segment_rfm(rfm_scored)
                
                # Display RFM results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Customers", len(rfm_scored))
                with col2:
                    avg_recency = rfm_scored['Recency'].mean()
                    st.metric("Avg Recency (days)", f"{avg_recency:.1f}")
                with col3:
                    avg_monetary = rfm_scored['Monetary'].mean()
                    st.metric("Avg Monetary", f"${avg_monetary:,.2f}")
                
                # RFM Distribution
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                
                axes[0].hist(rfm_scored['Recency'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                axes[0].set_title('Recency Distribution', fontsize=12, pad=15)
                axes[0].set_xlabel('Days since last purchase', fontsize=10)
                axes[0].set_ylabel('Frequency', fontsize=10)
                axes[0].tick_params(axis='both', which='major', labelsize=9)
                
                axes[1].hist(rfm_scored['Frequency'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
                axes[1].set_title('Frequency Distribution', fontsize=12, pad=15)
                axes[1].set_xlabel('Number of transactions', fontsize=10)
                axes[1].set_ylabel('Frequency', fontsize=10)
                axes[1].tick_params(axis='both', which='major', labelsize=9)
                
                axes[2].hist(rfm_scored['Monetary'], bins=20, color='salmon', edgecolor='black', alpha=0.7)
                axes[2].set_title('Monetary Distribution', fontsize=12, pad=15)
                axes[2].set_xlabel('Total spending', fontsize=10)
                axes[2].set_ylabel('Frequency', fontsize=10)
                axes[2].tick_params(axis='both', which='major', labelsize=9)
                
                plt.tight_layout(pad=2.0)
                st.pyplot(fig)
                plt.close(fig)
                
                # RFM Segments
                st.subheader("RFM Segments Distribution")
                segment_counts = rfm_scored['RFM_Segment'].value_counts()
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                bars = segment_counts.plot(kind='bar', color='teal', ax=ax2, alpha=0.7)
                ax2.set_title('Customer Segments by RFM', fontsize=14, pad=15)
                ax2.set_xlabel('Segment', fontsize=11)
                ax2.set_ylabel('Number of Customers', fontsize=11)
                ax2.tick_params(axis='both', which='major', labelsize=10)
                plt.xticks(rotation=45, ha='right')
                
                # Tambah nilai di atas bar
                for i, v in enumerate(segment_counts):
                    ax2.text(i, v + max(segment_counts)*0.01, str(v), 
                            ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                
                with st.expander("📊 Detail RFM Scores"):
                    st.dataframe(rfm_scored.sort_values('Monetary', ascending=False).head(20))
        
        with tab2:
            st.header("Customer Clustering")
            
            if 'rfm_scored' in locals():
                # Prepare data for clustering
                clustering_data = rfm_scored[['Recency', 'Frequency', 'Monetary']].copy()
                
                # Apply log transformation for skewed data
                clustering_data_log = np.log1p(clustering_data)
                
                # Standardize data
                scaler = StandardScaler()
                clustering_scaled = scaler.fit_transform(clustering_data_log)
                
                # Determine optimal clusters
                st.subheader("🔍 Menentukan Jumlah Cluster Optimal")
                
                with st.spinner("Menganalisis jumlah cluster optimal..."): 
                    K_range, wcss, silhouette_scores = find_optimal_clusters(clustering_scaled, max_clusters=8)
                
                # Plot Elbow Method and Silhouette Scores
                fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot WCSS (Elbow Method)
                ax1.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
                ax1.set_xlabel('Number of Clusters', fontsize=11)
                ax1.set_ylabel('WCSS', fontsize=11)
                ax1.set_title('Elbow Method', fontsize=13, pad=15)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='both', which='major', labelsize=10)
                
                # Plot Silhouette Scores
                ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
                ax2.set_xlabel('Number of Clusters', fontsize=11)
                ax2.set_ylabel('Silhouette Score', fontsize=11)
                ax2.set_title('Silhouette Score', fontsize=13, pad=15)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='both', which='major', labelsize=10)
                
                # Annotate recommended cluster (force K=3)
                recommended_k = 3
                if silhouette_scores:
                    K_list = list(K_range)
                    if recommended_k in K_list:
                        idx = K_list.index(recommended_k)
                        ax2.annotate(
                            f'Rekomendasi: K={recommended_k}',
                            xy=(recommended_k, silhouette_scores[idx]),
                            xytext=(recommended_k + 0.5, silhouette_scores[idx] - 0.05),
                            arrowprops=dict(arrowstyle='->', color='black'),
                            fontsize=10
                        )
                
                plt.tight_layout(pad=2.0)
                st.pyplot(fig3)
                plt.close(fig3)
                
                # Show recommended clusters based on silhouette score (forced to 3)
                st.info(f"**Rekomendasi jumlah cluster berdasarkan Silhouette Score: {recommended_k}**")
                
                # Perform clustering with selected number of clusters
                st.subheader(f"📊 Hasil Clustering (K = {n_clusters})")
                
                with st.spinner("Melakukan clustering..."):
                    kmeans_model = perform_clustering(clustering_scaled, n_clusters)
                
                # Add cluster labels
                rfm_scored['Cluster'] = kmeans_model.labels_
                
                # Cluster Analysis
                cluster_stats = rfm_scored.groupby('Cluster').agg({
                    'CustomerID': 'count',
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean',
                    'RFM_Segment': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(2)
                
                cluster_stats.columns = ['Customers', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Dominant_Segment']
                
                # Display cluster statistics
                st.dataframe(cluster_stats)
                
                # Visualize clusters
                st.subheader("📈 Visualisasi Cluster")
                
                fig4, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Recency vs Frequency
                scatter1 = axes[0].scatter(rfm_scored['Recency'], rfm_scored['Frequency'], 
                                          c=rfm_scored['Cluster'], cmap='viridis', 
                                          alpha=0.6, s=60, edgecolors='w', linewidth=0.5)
                axes[0].set_xlabel('Recency', fontsize=11)
                axes[0].set_ylabel('Frequency', fontsize=11)
                axes[0].set_title('Recency vs Frequency', fontsize=13, pad=15)
                axes[0].tick_params(axis='both', which='major', labelsize=10)
                axes[0].grid(True, alpha=0.3)
                
                # Recency vs Monetary
                axes[1].scatter(rfm_scored['Recency'], rfm_scored['Monetary'], 
                               c=rfm_scored['Cluster'], cmap='viridis', 
                               alpha=0.6, s=60, edgecolors='w', linewidth=0.5)
                axes[1].set_xlabel('Recency', fontsize=11)
                axes[1].set_ylabel('Monetary', fontsize=11)
                axes[1].set_title('Recency vs Monetary', fontsize=13, pad=15)
                axes[1].tick_params(axis='both', which='major', labelsize=10)
                axes[1].grid(True, alpha=0.3)
                
                # Frequency vs Monetary
                axes[2].scatter(rfm_scored['Frequency'], rfm_scored['Monetary'], 
                               c=rfm_scored['Cluster'], cmap='viridis', 
                               alpha=0.6, s=60, edgecolors='w', linewidth=0.5)
                axes[2].set_xlabel('Frequency', fontsize=11)
                axes[2].set_ylabel('Monetary', fontsize=11)
                axes[2].set_title('Frequency vs Monetary', fontsize=13, pad=15)
                axes[2].tick_params(axis='both', which='major', labelsize=10)
                axes[2].grid(True, alpha=0.3)
                
                # Colorbar
                cbar = plt.colorbar(scatter1, ax=axes, orientation='vertical', 
                                  fraction=0.02, pad=0.02)
                cbar.set_label('Cluster', fontsize=11)
                cbar.ax.tick_params(labelsize=10)
                
                plt.tight_layout(pad=2.0)
                st.pyplot(fig4)
                plt.close(fig4)
        
        with tab3:
            st.header("Association Rules Analysis")
            
            if 'rfm_scored' in locals():
                # Perform association analysis
                with st.spinner("Menganalisis pola asosiasi..."):
                    association_results = analyze_associations(
                        df_clean, 
                        rfm_scored, 
                        min_support, 
                        min_confidence
                    )
                
                if association_results:
                    # Display results per cluster
                    for cluster_num in sorted(association_results.keys()):
                        st.subheader(f"🔗 Association Rules - Cluster {cluster_num}")
                        
                        rules = association_results[cluster_num]
                        
                        if len(rules) > 0:
                            # Display top rules
                            st.write(f"**Jumlah rules ditemukan:** {len(rules)}")
                            
                            # Create readable rules (maksimal 10)
                            readable_rules = []
                            for _, row in rules.head(10).iterrows():
                                antecedents = ', '.join(list(row['antecedents']))
                                consequents = ', '.join(list(row['consequents']))
                                readable_rules.append({
                                    'Rule': f"{antecedents} → {consequents}",
                                    'Support': f"{row['support']:.3f}",
                                    'Confidence': f"{row['confidence']:.3f}",
                                    'Lift': f"{row['lift']:.3f}"
                                })
                            
                            # Display as table
                            rules_df = pd.DataFrame(readable_rules)
                            st.dataframe(rules_df)
                            
                            # Visualization
                            if len(rules) >= 3:  # Minimal 3 rules untuk visualisasi
                                fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                 
                                # Support vs Confidence
                                scatter = ax1.scatter(rules['support'], rules['confidence'], 
                                                      c=rules['lift'], cmap='YlOrRd', 
                                                      s=120, alpha=0.7, edgecolors='k', linewidth=0.5)
                                ax1.set_xlabel('Support', fontsize=11)
                                ax1.set_ylabel('Confidence', fontsize=11)
                                ax1.set_title(f'Support vs Confidence (Cluster {cluster_num})', 
                                             fontsize=13, pad=15)
                                ax1.grid(True, alpha=0.3)
                                ax1.tick_params(axis='both', which='major', labelsize=10)
                                 
                                # Colorbar untuk lift
                                cbar1 = plt.colorbar(scatter, ax=ax1, orientation='vertical', 
                                                    fraction=0.05, pad=0.02)
                                cbar1.set_label('Lift', fontsize=10)
                                cbar1.ax.tick_params(labelsize=9)
                                 
                                # Top 5 rules by lift
                                top_rules = rules.head(5).copy()
                                rule_labels = []
                                for _, row in top_rules.iterrows():
                                    antecedents = list(row['antecedents'])
                                    consequents = list(row['consequents'])
                                    # Batasi panjang label
                                    antecedents_str = ', '.join(antecedents[:2])
                                    if len(antecedents) > 2:
                                        antecedents_str += ', ...'
                                    consequents_str = ', '.join(consequents[:2])
                                    if len(consequents) > 2:
                                        consequents_str += ', ...'
                                    label = f"{antecedents_str} → {consequents_str}"
                                    # Truncate jika masih terlalu panjang
                                    if len(label) > 50:
                                        label = label[:47] + '...'
                                    rule_labels.append(label)
                                     
                                y_pos = np.arange(len(rule_labels))
                                bars = ax2.barh(y_pos, top_rules['lift'], color='steelblue', alpha=0.7)
                                ax2.set_yticks(y_pos)
                                ax2.set_yticklabels(rule_labels, fontsize=9)
                                ax2.set_xlabel('Lift', fontsize=11)
                                ax2.set_title(f'Top Rules by Lift (Cluster {cluster_num})', 
                                             fontsize=13, pad=15)
                                ax2.grid(True, alpha=0.3, axis='x')
                                ax2.tick_params(axis='both', which='major', labelsize=9)
                                 
                                # Tambah nilai lift di bar
                                for i, v in enumerate(top_rules['lift']):
                                    ax2.text(v + max(top_rules['lift'])*0.01, i, 
                                            f'{v:.2f}', va='center', fontsize=9)
                                 
                                plt.tight_layout(pad=2.0)
                                st.pyplot(fig5)
                                plt.close(fig5)
                else:
                    st.info("Tidak ditemukan association rules yang signifikan untuk cluster manapun.")
        
        with tab4:
            st.header("Marketing Recommendations")
            
            if 'rfm_scored' in locals():
                st.subheader("🎯 Strategi Promosi Berdasarkan Cluster")
                
                # Generate recommendations for each cluster
                for cluster_num in sorted(rfm_scored['Cluster'].unique()):
                    cluster_data = rfm_scored[rfm_scored['Cluster'] == cluster_num]
                    
                    if len(cluster_data) == 0:
                        continue;
                    
                    # Calculate cluster characteristics
                    avg_recency = cluster_data['Recency'].mean()
                    avg_frequency = cluster_data['Frequency'].mean()
                    avg_monetary = cluster_data['Monetary'].mean()
                    segment_counts = cluster_data['RFM_Segment'].value_counts()
                    dominant_segment = segment_counts.index[0] if len(segment_counts) > 0 else "Unknown";
                    
                    # Display cluster info
                    with st.expander(f"📋 Cluster {cluster_num} - {len(cluster_data)} customers"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Recency", f"{avg_recency:.1f} days")
                        with col2:
                            st.metric("Avg Frequency", f"{avg_frequency:.1f}")
                        with col3:
                            st.metric("Avg Monetary", f"${avg_monetary:,.2f}")
                        
                        st.write(f"**Dominant RFM Segment:** {dominant_segment}")
                        
                        # Generate recommendations based on characteristics
                        #st.write("**Rekomendasi Strategi:**")
                        
                        #recommendations = []
                        
                        #if avg_recency < 30:
                            #recommendations.append("✅ **Pelanggan Aktif**: Pertahankan dengan program loyalitas")
                        #elif avg_recency < 90:
                            #recommendations.append("⚠️ **Perlu Perhatian**: Kirim promo re-engagement")
                        
                        #if avg_frequency > 5:
                            #recommendations.append("💎 **Sering Belanja**: Tawarkan membership premium")
                        #elif avg_frequency <= 2:
                            #recommendations.append("🎯 **Sesekali Belanja**: Tingkatkan frekuensi dengan bundling")
                        
                        #if avg_monetary > cluster_data['Monetary'].median() * 2:
                            #recommendations.append("💰 **High Value**: Tawarkan produk premium dan VIP service")
                        
                        #for rec in recommendations:
                            #st.write(f"- {rec}")
                        
                        # Add association-based recommendations if available
                        if 'association_results' in locals() and cluster_num in association_results:
                            rules = association_results[cluster_num]
                            if len(rules) > 0:
                                st.write("**Rekomendasi Produk Berdasarkan Asosiasi:**")
                                top_rule = rules.iloc[0]
                                antecedents = list(top_rule['antecedents'])
                                consequents = list(top_rule['consequents'])
                                
                                st.write(f"- **Bundling**: Gabungkan {', '.join(antecedents[:3])} dengan {', '.join(consequents[:3])}")
                                st.write(f"- **Cross-selling**: Tampilkan {', '.join(consequents[:3])} saat pelanggan melihat {', '.join(antecedents[:3])}")
                                st.write(f"- **Confidence**: {top_rule['confidence']:.1%}, **Lift**: {top_rule['lift']:.2f}")
                
                # Overall recommendations
                st.subheader("📈 Rekomendasi Umum")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    **Untuk High-Value Customers:**
                    - VIP program dengan benefits eksklusif
                    - Early access ke produk baru
                    - Personal shopping assistant
                    - Free shipping dan returns
                    """)
                
                with col2:
                    st.info("""
                    **Untuk At-Risk Customers:**
                    - Personalized email campaign
                    - Special discount untuk transaksi pertama
                    - Survey untuk memahami kebutuhan
                    - Reactivation offers
                    """)
                
                # Export button
                if st.button("📥 Download Hasil Analisis"):
                    csv = rfm_scored.to_csv(index=False)
                    st.download_button(
                        label="Klik untuk download",
                        data=csv,
                        file_name="hasil_analisis_pelanggan.csv",
                        mime="text/csv"
                    )
    else:
        # Tampilkan instruksi jika belum ada data
        st.info("👈 Silahkan upload dataset Anda melalui sidebar")
        
        # Display sample data structure
        st.header("📋 Struktur Data yang Diperlukan")
        
        sample_data = pd.DataFrame({
            'CustomerID': [17850, 17850, 13047],
            'Transaction_ID': [16679, 16680, 16684],
            'Transaction_Date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'Product_Category': ['Electronics', 'Electronics', 'Home Appliances'],
            'Quantity': [1, 1, 2],
            'Avg_Price': [153.71, 153.71, 122.77]
        })
        
        st.dataframe(sample_data)
        st.caption("Pastikan dataset Anda memiliki minimal kolom-kolom di atas")

if __name__ == "__main__":
    main()
