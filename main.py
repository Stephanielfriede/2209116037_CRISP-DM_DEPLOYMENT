import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pickle 


# Load data
df_file = pd.read_csv('https://raw.githubusercontent.com/Stephanielfriede/Mini-Project-Data-Mining/main/Data%20Cleaned%20(Without%20Encoding).csv')

filtered_df_numeric = df_file.select_dtypes(include=['float64', 'int64'])


# Load KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load Hierarchical Clustering model
with open('hierarchical_model.pkl', 'rb') as f:
    hierarchical = pickle.load(f)

st.set_page_config(
    page_title="Manga Series Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Main Panel
with st.container():
    st.markdown('# Analisis Faktor-Faktor yang Mempengaruhi Penjualan Manga Best Seller')
   
with st.sidebar:
    st.title('📚 Manga Series Dashboard')

    st.subheader('Select Dashboard Section')
    section_option = st.selectbox('', ('Home', 'Distribution', 'Comparison', 'Composition', 'Relationship', 'Clustering'))

# Visualisasi di layar utama
# Distribusi Jumlah Volume Manga per Penerbit
if section_option == 'Home':
    # Menampilkan gambar di dashboard
    st.image('https://cdn.keepo.me/images/post/lists/2019/01/27/main-list-image-a60fba1f-5e7f-4a01-b432-bba0f57a9879-1.jpg', caption='Manga Best Seller', use_column_width=True)
    st.write("""
    Manga adalah istilah yang merujuk kepada komik atau novel grafis Jepang. Istilah ini digunakan secara luas untuk menggambarkan karya-karya seni visual yang sering kali memiliki narasi yang kompleks dan beragam genre, mulai dari aksi, petualangan, romansa, hingga fantasi dan fiksi ilmiah. Manga biasanya dibaca dari kanan ke kiri, berbeda dengan komik barat yang dibaca dari kiri ke kanan.

Sejarah manga dimulai pada abad ke-12 ketika rol gambar dikenal sebagai "chojugiga" muncul di Jepang. Namun, bentuk modern dari manga mulai berkembang pada abad ke-18 dan ke-19 dengan munculnya cetakan kayu dan majalah berilustrasi seperti "Hokusai Manga" karya Katsushika Hokusai. Pada abad ke-20, manga menjadi semakin populer dengan munculnya majalah-majalah seperti "Shonen Jump" dan "Shojo Club" yang memperkenalkan berbagai karya manga yang ikonik.

Berikut adalah top 10 manga terlaris sepanjang masa berdasarkan dataset [Best Selling Manga Dataset](https://www.kaggle.com/datasets/drahulsingh/best-selling-manga):
    """)
    # Data tentang manga
    manga_data = {
        "Golgo 13": {
            "description": "Golgo 13 adalah manga aksi dan thriller yang mengikuti petualangan Duke Togo, seorang pembunuh bayaran legendaris yang dikenal karena keahliannya yang luar biasa dalam menyelesaikan pekerjaannya.",
            "features": "Cerita-cerita dalam Golgo 13 sering kali menampilkan plot yang kompleks, adegan aksi yang menegangkan, dan elemen misteri yang kuat."
        },
        "Case Closed / Detective Conan": {
            "description": "Case Closed mengisahkan tentang Shinichi Kudo, seorang detektif SMA yang berubah menjadi seorang anak kecil bernama Conan Edogawa setelah diracuni oleh organisasi kriminal. Dia terus menyelidiki kasus sambil mencari cara untuk mengembalikan tubuhnya yang asli.",
            "features": "Manga ini menawarkan alur cerita detektif yang rumit, kasus-kasus misteri yang menarik, dan karakter-karakter yang kuat."
        },
        "KochiKame: Tokyo Beat Cops": {
            "description": "KochiKame adalah manga komedi yang berfokus pada kehidupan sehari-hari polisi di sebuah kantor polisi di Tokyo. Ceritanya sering kali menggambarkan situasi lucu dan konyol yang dihadapi oleh para polisi.",
            "features": "Manga ini menampilkan humor slapstick, karakter-karakter yang khas, dan penggambaran yang kreatif tentang kehidupan polisi."
        },
        "Oishinbo": {
            "description": "Oishinbo adalah manga kuliner yang mengeksplorasi berbagai aspek masakan Jepang, mulai dari bahan makanan hingga teknik memasak. Ceritanya mengikuti petualangan seorang wartawan makanan yang berusaha mencari makanan terbaik.",
            "features": "Manga ini menampilkan ilustrasi makanan yang menggugah selera, resep masakan autentik, dan informasi yang mendalam tentang budaya kuliner Jepang."
        },
        "JoJo's Bizarre Adventure": {
            "description": "JoJo's Bizarre Adventure adalah manga aksi dan petualangan yang mengikuti berbagai generasi keluarga Joestar dalam pertempuran melawan musuh-musuh supernatural. Setiap bagian memiliki protagonis dan musuh yang unik.",
            "features": "Manga ini terkenal karena gaya seni yang khas, pertarungan yang kreatif, dan alur cerita yang beragam dari satu bagian ke bagian lainnya."
        },
        "Hajime no Ippo": {
            "description": "Hajime no Ippo adalah manga olahraga yang mengisahkan tentang perjalanan seorang pemuda bernama Ippo Makunouchi dalam dunia tinju. Ia berusaha untuk menjadi petinju yang hebat dengan bimbingan pelatihnya, Kamogawa.",
            "features": "Manga ini menampilkan pertarungan tinju yang intens, penggambaran yang akurat tentang dunia tinju, dan pengembangan karakter yang mendalam."
        },
        "Captain Tsubasa": {
            "description": "Captain Tsubasa adalah manga olahraga yang mengikuti perjalanan seorang pemain sepak bola muda bernama Tsubasa Ozora dalam mencapai mimpinya menjadi pemain sepak bola terbaik di dunia.",
            "features": "Manga ini menampilkan aksi sepak bola yang mendebarkan, teknik-teknik sepak bola yang realistis, dan pesan-pesan inspiratif tentang kerja keras dan tekad."
        },
        "Dragon Quest Retsuden: Roto no Monshō": {
            "description": "Dragon Quest Retsuden mengadaptasi cerita dari seri game Dragon Quest. Manga ini mengikuti petualangan pahlawan legendaris Roto dalam menjelajahi dunia fantasi yang penuh dengan monster dan rintangan.",
            "features": "Manga ini menampilkan dunia fantasi yang kaya, pertempuran epik dengan monster, dan kisah kepahlawanan yang memikat."
        },
        "Ace of Diamond": {
            "description": "Ace of Diamond adalah manga olahraga yang berkisah tentang tim bisbol SMA Seidou yang berusaha untuk menjadi juara nasional di turnamen bisbol sekolah menengah.",
            "features": "Manga ini menampilkan strategi dan taktik dalam permainan bisbol, dinamika tim yang kompleks, dan penggambaran yang realistis tentang dunia bisbol sekolah."
        },
        "One-Punch Man": {
            "description": "One-Punch Man mengikuti petualangan Saitama, seorang pahlawan yang sangat kuat sehingga ia bisa mengalahkan musuh-musuhnya hanya dengan satu pukulan. Namun, ia merasa bosan karena kurangnya tantangan.",
            "features": "Manga ini menampilkan aksi pertarungan yang spektakuler, humor yang khas, dan parodi atas genre pahlawan super."
        }
    }

    # Tampilkan judul halaman
    st.title('Top 10 Best Selling Manga')

    # Pilih manga dari dropdown
    selected_manga = st.selectbox('Pilih Manga', list(manga_data.keys()))

    # Tampilkan informasi tentang manga yang dipilih
    if selected_manga:
        st.subheader(selected_manga)
        st.write("**Penjelasan**: " + manga_data[selected_manga]["description"])
        st.write("**Ringkasan**: " + manga_data[selected_manga]["features"])


elif section_option == 'Distribution':
    st.subheader('Distribution by Demographic or Publisher')
    distribution_option = st.selectbox('Select Distribution Type', ('Demographic', 'Publisher'))

    if distribution_option == 'Demographic':
        st.subheader('Distribution by Demographic')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Demographic', y='Approximate sales in million(s)', data=df_file, estimator=sum, ci=None)
        plt.title('Distribution of Manga Sales by Demographic')
        plt.xlabel('Demographic')
        plt.ylabel('Approximate Sales (in millions)')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        st.write("""
        Interpretasi:
        - Dari visualisasi tersebut, terlihat bahwa genre Shōnen dan Seinen mendominasi pasar manga dengan penjualan yang tinggi dan variasi yang signifikan. Hal ini menunjukkan bahwa manga-manga dengan genre tersebut cenderung lebih diminati oleh pembaca dan memiliki pengaruh yang kuat dalam pasar.
        - Demographic lainnya, meskipun tidak sepopuler Shōnen dan Seinen, masih memiliki pangsa pasar yang signifikan. Meskipun penjualan mereka tidak sebesar dua genre utama tersebut, mereka masih memiliki peluang untuk menarik pembaca dengan preferensi yang berbeda.

        Insight:
        - Shōnen dan Seinen adalah genre yang paling populer dalam industri manga, dengan penjualan yang dominan dan variasi yang signifikan. Hal ini menunjukkan bahwa fokus pada pengembangan dan pemasaran manga dengan genre tersebut dapat menjadi strategi yang efektif untuk mencapai kesuksesan dalam pasar manga.
        - Meskipun Shōnen dan Seinen mendominasi, tidak boleh diabaikan bahwa genre-genre lain masih memiliki pangsa pasar yang signifikan. Hal ini menunjukkan bahwa ada potensi untuk meningkatkan penjualan dan popularitas genre-genre yang kurang populer dengan strategi pemasaran yang tepat.

        Actionable Insight:
        - Perusahaan penerbit manga dapat mempertimbangkan untuk lebih fokus pada produksi dan promosi manga dengan genre Shōnen dan Seinen untuk meningkatkan penjualan dan keuntungan.
        - Meskipun demikian, tidak boleh mengabaikan genre-genre lainnya. Perusahaan dapat mengambil langkah-langkah untuk mengembangkan strategi pemasaran yang lebih cermat dan kreatif untuk meningkatkan popularitas genre-genre yang kurang populer.
        - Melakukan riset pasar lebih lanjut untuk memahami preferensi pembaca dan tren pasar akan membantu dalam mengidentifikasi peluang baru dan mengambil langkah-langkah yang sesuai untuk meningkatkan kinerja manga.
        """)
 
    elif distribution_option == 'Publisher':
        st.subheader('Distribution by Publisher')
        plt.figure(figsize=(12, 6))
        publisher_sales = df_file.groupby('Publisher')['Approximate sales in million(s)'].sum()
        publisher_sales.plot(kind='bar')
        plt.title('Distribution of Manga Sales by Publisher')
        plt.xlabel('Publisher')
        plt.ylabel('Approximate Sales (in millions)')
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.write("""
Interpretasi:
- Dari visualisasi tersebut, terlihat bahwa penerbit manga utama, yaitu Shueisha, Kodansha, dan Shogakukan, mendominasi pasar dengan penjualan yang tinggi. Hal ini menunjukkan bahwa ketiga penerbit ini memiliki pengaruh yang signifikan dalam industri manga dan mampu menarik minat pembaca dengan karya-karya mereka.
- Penjualan yang jauh lebih tinggi dari ketiga penerbit tersebut dibandingkan dengan penerbit lainnya menunjukkan bahwa mereka memiliki keunggulan kompetitif yang kuat dalam hal pemasaran, distribusi, dan portofolio karya.

Insight:
- Shueisha, Kodansha, dan Shogakukan memiliki pangsa pasar yang dominan dalam industri manga, menunjukkan pentingnya peran mereka sebagai pemimpin pasar. Keberhasilan mereka dalam menarik pembaca dan memonopoli penjualan menunjukkan bahwa strategi pemasaran dan kualitas karya mereka memiliki dampak yang signifikan.
- Penerbit manga lainnya mungkin perlu mengadopsi strategi yang lebih inovatif dan agresif untuk bersaing dengan ketiga penerbit utama. Mereka perlu memahami kekuatan dan kelemahan mereka sendiri serta tren pasar untuk mengembangkan strategi yang efektif.

Actionable Insight:
- Penerbit manga yang tidak termasuk dalam kategori utama seperti Shueisha, Kodansha, dan Shogakukan perlu melakukan evaluasi menyeluruh terhadap strategi pemasaran dan portofolio karya mereka. Seperti peningkatan kualitas konten, penetrasi pasar yang lebih agresif, dan kolaborasi dengan kreator dan platform distribusi untuk meningkatkan visibilitas dan daya tarik bagi pembaca.
- Mengembangkan kemitraan dengan platform distribusi digital dan mengadopsi strategi pemasaran digital yang efektif dapat membantu penerbit manga untuk mencapai audiens yang lebih luas dan meningkatkan penjualan mereka.
- Penerbit manga juga perlu berinvestasi dalam riset pasar untuk memahami preferensi pembaca dan tren industri, sehingga mereka dapat menyesuaikan strategi mereka sesuai dengan permintaan pasar yang terus berkembang.
        """)

# Perbandingan Jumlah Volume Manga dengan Penjualan, Komposisi Demografis Manga, dan Hubungan antara Jumlah Volume dan Penjualan per Demografis
if section_option == 'Comparison':
    st.subheader('Comparison')
    comparison_data = df_file.groupby('Publisher')['Volume Category'].value_counts().unstack()

    # Plot the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_data.plot(kind='bar', ax=ax)
    ax.set_xlabel('Publisher')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Volume Category by Publisher')
    ax.legend(title='Volume Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.grid(True)
    st.pyplot(fig)
    st.write("""
Interpretasi:
- Dari hasil perbandingan kategori volume manga berdasarkan penerbit, terlihat bahwa ada variasi yang cukup besar dalam jumlah volume manga yang diterbitkan oleh masing-masing penerbit. Mayoritas penerbit cenderung menerbitkan lebih banyak karya dengan volume yang lebih rendah, namun ada beberapa pengecualian di mana beberapa penerbit memiliki lebih banyak karya dengan volume besar, seperti Kodansha, Shogakukan, dan Shueisha.
- Perbedaan ini menunjukkan bahwa preferensi volume manga dapat berbeda-beda tergantung pada penerbitnya. Beberapa penerbit cenderung fokus pada menerbitkan karya dengan volume yang lebih besar, sementara yang lain mungkin lebih memilih untuk menerbitkan karya dengan volume yang lebih kecil.

Insight:
- Penerbit seperti Kodansha, Shogakukan, dan Shueisha memiliki kecenderungan untuk menerbitkan karya dengan volume yang lebih besar. Hal ini mungkin disebabkan oleh strategi pemasaran mereka yang bertujuan untuk menciptakan seri manga yang lebih panjang dan menarik bagi pembaca.
- Di sisi lain, penerbit lain seperti Enix, Gakken, dan Ushio Shuppan cenderung lebih fokus pada menerbitkan karya dengan volume yang lebih sedikit. Mereka mungkin memiliki strategi yang berbeda, mungkin lebih memilih untuk fokus pada kualitas daripada kuantitas, atau mungkin lebih memperhatikan preferensi pembaca yang lebih memilih karya dengan volume yang lebih singkat.

Actionable Insight:
- Untuk penerbit yang cenderung menerbitkan karya dengan volume besar, mereka dapat mempertimbangkan untuk terus mengembangkan seri manga yang menarik dan mengikuti tren, namun tetap memperhatikan kualitas cerita dan seni yang konsisten.
- Bagi penerbit yang cenderung menerbitkan karya dengan volume yang lebih sedikit, mereka dapat fokus pada pengembangan karya-karya yang berkualitas tinggi dan inovatif, serta mempertimbangkan untuk menargetkan segmen pembaca yang lebih spesifik dengan preferensi untuk karya dengan volume yang lebih pendek.
- Penerbit juga dapat melakukan riset pasar lebih lanjut untuk memahami preferensi pembaca dan tren pasar, sehingga mereka dapat menyesuaikan strategi penerbitan mereka dengan lebih baik untuk memenuhi permintaan pasar yang terus berubah.
    """)

if section_option == 'Composition':
    st.subheader('Composition of Manga Sales by Volume Category')
    sales_by_volume_category = df_file.groupby('Volume Category')['Approximate sales in million(s)'].sum()
    fig, ax = plt.subplots()
    ax.pie(sales_by_volume_category, labels=sales_by_volume_category.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Composition of Manga Sales by Volume Category')
    st.pyplot(fig)
    st.write("""Interpretasi:
- Hasil dari 'Composition of Manga Sales by Volume Category' menyoroti preferensi pembaca terhadap berbagai kategori volume manga. Ditemukan bahwa penjualan manga didominasi oleh kategori volume 'few', yang menyumbang sebagian besar (63.1%) dari total penjualan. Lalu diikuti oleh kategori 'moderate' dengan 19.5%, dan kategori 'many' dengan 17.4%.
- Mayoritas pembelian manga cenderung pada volume-volume yang lebih sedikit, menunjukkan bahwa pembaca mungkin lebih tertarik pada cerita-cerita yang lebih singkat atau seri manga dengan jumlah volume yang terbatas.

Insight:
- Dominasi kategori volume 'few' dalam penjualan manga menunjukkan bahwa ada kecenderungan pembaca untuk memilih karya dengan volume yang lebih pendek atau seri manga dengan jumlah volume yang terbatas. Hal ini mungkin karena preferensi pembaca untuk cerita yang lebih singkat atau karena faktor lain seperti keterbatasan waktu atau anggaran.

Actionable Insight:
- Berdasarkan insight ini, penerbit manga dapat mempertimbangkan untuk mengembangkan lebih banyak karya dengan volume yang lebih pendek atau seri manga dengan jumlah volume terbatas. Hal ini dapat meningkatkan daya tarik bagi pembaca yang mencari cerita yang lebih singkat atau yang memiliki keterbatasan waktu untuk membaca seri yang lebih panjang.
- Selain itu, penerbit juga dapat melakukan riset lebih lanjut untuk memahami faktor-faktor yang mendorong preferensi pembaca terhadap kategori volume manga tertentu. Informasi ini dapat membantu mereka menyusun strategi penerbitan yang lebih tepat dan efektif untuk memenuhi kebutuhan dan keinginan pembaca dengan lebih baik. """)

if section_option == 'Relationship':
     # Display correlation matrix in the "Relationship" section
    st.markdown('### Correlation Matrix between Numeric Features')
    corr = filtered_df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix between Numeric Features')
    st.pyplot(plt)
    st.write("""
Interpretasi:
1. **Korelasi antara Jumlah Volume**: Korelasi sempurna antara jumlah volume yang terkumpul dengan dirinya sendiri menunjukkan bahwa tidak ada perubahan relatif antara volume manga dari waktu ke waktu. Ini mengindikasikan konsistensi dalam penambahan volume dari suatu seri manga.
   
2. **Korelasi antara Penjualan Keseluruhan dan Jumlah Volume**: Korelasi positif menunjukkan bahwa penjualan manga secara keseluruhan cenderung meningkat seiring dengan peningkatan jumlah volume yang tersedia. Ini menggambarkan minat pembaca yang kuat dalam melanjutkan cerita, yang mendorong penjualan yang lebih tinggi saat volume baru dirilis.

3. **Korelasi antara Rata-rata Penjualan per Volume dan Penjualan Keseluruhan**: Korelasi positif yang kuat mengindikasikan bahwa kualitas konten yang konsisten atau popularitas seri tertentu dapat berkontribusi pada peningkatan penjualan manga secara keseluruhan. Jika rata-rata penjualan per volume tinggi, maka penjualan manga secara keseluruhan juga cenderung tinggi.

4. **Korelasi antara Jumlah Volume dan Rata-rata Penjualan per Volume**: Korelasi negatif menunjukkan bahwa semakin banyak volume yang terkumpul, rata-rata penjualan per volume cenderung lebih rendah. Hal ini bisa disebabkan oleh potensi pembaca yang kurang tertarik untuk memulai seri yang memiliki banyak volume, sehingga mempengaruhi penjualan per volume.

Insight:
- Penerbit manga dapat memanfaatkan korelasi positif antara jumlah volume yang terkumpul dan penjualan keseluruhan untuk merencanakan strategi penerbitan yang lebih efektif. Mereka dapat memperkirakan potensi penjualan baru dengan merilis lebih banyak volume dari seri yang sedang populer.
- Korelasi antara rata-rata penjualan per volume dan penjualan keseluruhan menunjukkan pentingnya konten berkualitas dan popularitas seri. Penerbit dapat fokus pada pengembangan konten yang menarik untuk meningkatkan penjualan secara keseluruhan.
- Korelasi negatif antara jumlah volume dan rata-rata penjualan per volume mengindikasikan bahwa penerbit harus berhati-hati dalam meningkatkan jumlah volume dari suatu seri. Mereka perlu mempertimbangkan keseimbangan antara memuaskan pembaca yang setia dengan melanjutkan cerita dan menarik pembaca baru dengan mengurangi risiko penurunan rata-rata penjualan per volume.

Actionable Insight:
- Penerbit manga dapat melakukan riset pasar lebih lanjut untuk memahami preferensi pembaca terhadap jumlah volume dan kualitas konten. Hal ini akan membantu mereka dalam menentukan strategi penerbitan yang lebih tepat dan efektif.
- Penggunaan data korelasi dapat membantu penerbit dalam membuat keputusan yang lebih berdasarkan bukti untuk merencanakan perilisan volume baru, mempromosikan seri yang sedang populer, dan mengoptimalkan penjualan manga secara keseluruhan.
    """)



# Clustering Analysis
if section_option == 'Clustering':
    st.subheader('Clustering Analysis of Manga Sales')
    st.write("For clustering analysis, we'll focus on the numerical features 'Approximate sales in million(s)' and 'No. of collected volumes'.")

    # Selecting numerical features for clustering
    clustering_data = df_file[['Approximate sales in million(s)', 'No. of collected volumes']]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Selecting number of clusters with slider
    num_clusters = st.slider("Select number of clusters (2-8):", min_value=2, max_value=8, value=4, step=1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)
    kmeans_cluster_labels = kmeans.labels_

    # Perform Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    hierarchical_cluster_labels = hierarchical.fit_predict(scaled_data)

    # Visualizing the clusters
    plt.figure(figsize=(16, 6))

    # Plot KMeans clustering
    plt.subplot(1, 2, 1)
    plt.scatter(clustering_data['No. of collected volumes'], clustering_data['Approximate sales in million(s)'],
                c=kmeans_cluster_labels, cmap='viridis', s=50)
    plt.title(f'KMeans Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('No. of collected volumes')
    plt.ylabel('Approximate Sales in Million(s)')
    plt.grid(True)

    # Plot Hierarchical clustering
    plt.subplot(1, 2, 2)
    plt.scatter(clustering_data['No. of collected volumes'], clustering_data['Approximate sales in million(s)'],
                c=hierarchical_cluster_labels, cmap='viridis', s=50)
    plt.title(f'Hierarchical Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('No. of collected volumes')
    plt.ylabel('Approximate Sales in Million(s)')
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation of clusters
    st.write(f"*Number of Clusters: {num_clusters}*")
    st.write("""
Output dari analisis clustering menampilkan visualisasi scatter plot yang membagi data manga menjadi beberapa kelompok berdasarkan jumlah volume yang terkumpul dan penjualan perkiraan dalam jutaan. Scatter plot tersebut menunjukkan titik-titik data yang mewakili setiap manga, di mana sumbu-x menunjukkan jumlah volume yang terkumpul dan sumbu-y menunjukkan penjualan perkiraan dalam jutaan. Dengan menggunakan dua jenis clustering, yaitu KMeans dan Hierarchical Clustering, data dibagi menjadi kelompok-kelompok yang berbeda. Setiap kelompok ditandai dengan warna yang berbeda pada scatter plot, memungkinkan kita untuk melihat pola-pola yang muncul dan perbedaan antara kluster yang dihasilkan oleh kedua metode clustering tersebut.    """)
