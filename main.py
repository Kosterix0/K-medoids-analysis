import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.datasets import make_blobs, fetch_california_housing
from sklearn.cluster import KMeans


def calculate_euclidean_distance(X, medoids):
    """Oblicza macierz odległości euklidesowej."""
    return np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)


def calculate_total_cost(X, labels, medoids, k):
    """Oblicza sumę odległości punktów do ich przypisanych medoid."""
    cost = 0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            cost += np.sum(np.linalg.norm(cluster_points - medoids[i], axis=1))
    return cost


def k_medoids(X, k, max_iters=100):
    """Główna implementacja algorytmu PAM (Partitioning Around Medoids) z historią."""
    m, n = X.shape
    np.random.seed(42)
    medoid_indices = np.random.choice(m, k, replace=False)
    medoids = X[medoid_indices]
    labels = np.zeros(m)
    history = []

    for iteration in range(max_iters):
        distances = calculate_euclidean_distance(X, medoids)
        new_labels = np.argmin(distances, axis=1)

        current_cost = calculate_total_cost(X, new_labels, medoids, k)
        history.append({
            'iteration': iteration + 1,
            'medoids': np.copy(medoids),
            'labels': np.copy(new_labels),
            'cost': current_cost
        })

        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = X[new_labels == i]
            if len(cluster_points) == 0: continue

            cost_matrix = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
            total_costs = np.sum(cost_matrix, axis=1)
            best_idx = np.argmin(total_costs)
            new_medoids[i] = cluster_points[best_idx]

        if np.array_equal(medoids, new_medoids):
            # Zapisz końcowy stan, gdy algorytm się zatrzymał
            final_cost = calculate_total_cost(X, new_labels, new_medoids, k)
            history.append({
                'iteration': iteration + 2,
                'medoids': np.copy(new_medoids),
                'labels': np.copy(new_labels),
                'cost': final_cost
            })
            break

        medoids = new_medoids
        labels = new_labels

    return medoids, labels, history


# ==================== INTERFEJS STREAMLIT ====================
st.set_page_config(page_title="Analiza Skupień: K-Medoids", layout="wide")

st.title("Analiza Skupień K-Medoids (PAM)")

st.markdown(r"""
### 📑 Wstęp
Analiza skupień (clustering) to technika uczenia nienadzorowanego, której celem jest podział zbioru danych na grupy (skupienia) podobnych do siebie obiektów.

#### Algorytm **K-Medoids**
Algorytm K-Medoids jest odporną na zniekształcenia modyfikacją algorytmu K-Means.
* W **K-Means** centrum (centroida) jest wirtualnym punktem wyliczanym jako średnia.
* W **K-Medoids** centrum (medoida) to **rzeczywisty punkt** należący do zbioru, minimalizujący sumę odległości do pozostałych obiektów w grupie.

#### 2. Sformułowanie Matematyczne
Jako miarę podobieństwa między dwoma punktami $A = (a_1, a_2, \dots, a_n)$ oraz $B = (b_1, b_2, \dots, b_n)$ przyjmujemy **odległość euklidesową**:
$$d(A, B) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

Funkcja celu, którą algorytm stara się zminimalizować w sposób iteracyjny, to całkowity koszt uogólniony (suma odległości):
$$J = \sum_{i=1}^{k} \sum_{X \in C_i} d(X, M_i)$$
gdzie $k$ to liczba skupień, $C_i$ to $i$-te skupienie, a $M_i$ to medoida przypisana do tego skupienia.

#### 3. Przebieg Algorytmu:
1. **Inicjalizacja:** Losowy wybór $k$ punktów jako początkowe medoidy.
2. **Przypisanie:** Obliczenie odległości każdego punktu i przypisanie do najbliższej medoidy.
3. **Aktualizacja:** Wyznaczenie nowej medoidy minimalizującej sumę odległości wewnątrz grupy.
4. **Warunek stopu:** Powtarzanie kroków do momentu stabilizacji (braku zmian).
""")

st.write("---")

st.sidebar.header("Parametry")

dataset_type = st.sidebar.selectbox("Wybierz zbiór danych", ("Syntetyczny", "Geolokalizacja (Kalifornia)"))
k_param = st.sidebar.slider("Liczba skupień (k)", min_value=2, max_value=8, value=3)

if dataset_type == "Syntetyczny":
    n_samples = st.sidebar.slider("Liczba punktów w klastrach", min_value=50, max_value=500, value=150)
    X, _ = make_blobs(n_samples=n_samples, centers=k_param, cluster_std=1.0, random_state=42)
    x_label, y_label = "Cecha 1", "Cecha 2"
    x_min, x_max, x_def = -25.0, 25.0, 15.0
    y_min, y_max, y_def = -25.0, 25.0, 15.0
else:
    california = fetch_california_housing()
    full_X = np.column_stack((california.data[:, 7], california.data[:, 6]))
    np.random.seed(42)
    indices = np.random.choice(full_X.shape[0], 300, replace=False)
    X = full_X[indices]
    x_label, y_label = "Długość geograficzna (Longitude)", "Szerokość geograficzna (Latitude)"
    x_min, x_max, x_def = -140.0, -110.0, -130.0
    y_min, y_max, y_def = 30.0, 50.0, 35.0

st.sidebar.write("---")
st.sidebar.subheader("Eksperymenty i Anomalie")

compare_kmeans = st.sidebar.checkbox("Porównaj z K-Means")
add_individual = st.sidebar.checkbox("Dodaj pojedyncze anomalie")
add_cloud = st.sidebar.checkbox("Dodaj chmurę szumu")

individual_outliers = []
cloud_outliers = []

if add_individual:
    n_outliers = st.sidebar.slider("Liczba pojedynczych anomalii", 1, 5, 1)
    for i in range(n_outliers):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            out_x = st.slider(f"X (Pkt {i + 1})", min_value=x_min, max_value=x_max, value=x_def, step=0.5,
                              key=f"ind_x_{i}")
        with col2:
            out_y = st.slider(f"Y (Pkt {i + 1})", min_value=y_min, max_value=y_max, value=y_def, step=0.5,
                              key=f"ind_y_{i}")
        individual_outliers.append([out_x, out_y])

if add_cloud:
    cloud_size = st.sidebar.slider("Liczba punktów w chmurze", 5, 50, 20)
    col3, col4 = st.sidebar.columns(2)
    with col3:
        cloud_x = st.slider("Chmura X", min_value=x_min, max_value=x_max, value=x_min + 5, step=0.5)
    with col4:
        cloud_y = st.slider("Chmura Y", min_value=y_min, max_value=y_max, value=y_min + 5, step=0.5)

    spread = 1.5 if dataset_type == "Syntetyczny" else 0.5
    np.random.seed(99)
    cloud_pts = np.random.randn(cloud_size, 2) * spread + np.array([cloud_x, cloud_y])
    cloud_outliers = cloud_pts.tolist()

all_outliers = individual_outliers + cloud_outliers
if all_outliers:
    X = np.vstack([X, np.array(all_outliers)])

with st.spinner('Trwa obliczanie skupień...'):
    medoids, labels, history = k_medoids(X, k=k_param)


def plot_outliers(ax):
    if individual_outliers:
        ax.scatter(np.array(individual_outliers)[:, 0], np.array(individual_outliers)[:, 1],
                   color='black', marker='*', s=200, label='Pojedyncze Anomalie', zorder=5)
    if cloud_outliers:
        ax.scatter(np.array(cloud_outliers)[:, 0], np.array(cloud_outliers)[:, 1],
                   color='grey', marker='x', s=50, label='Chmura Szumu', zorder=5)


if compare_kmeans:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # K-MEDOIDS
    ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    ax1.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoidy (Rzeczywiste)',
                edgecolors='black', zorder=6)
    plot_outliers(ax1)
    ax1.set_title(f"Wynik grupowania algorytmem K-Medoids (k={k_param})", fontsize=14)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # K-MEANS
    kmeans = KMeans(n_clusters=k_param, random_state=42, n_init="auto")
    kmeans_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, label='Centroidy (Wirtualne)',
                edgecolors='black', zorder=6)
    plot_outliers(ax2)
    ax2.set_title(f"Wynik grupowania algorytmem K-Means (k={k_param})", fontsize=14)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    ax.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoidy', edgecolors='black', zorder=6)
    plot_outliers(ax)
    ax.set_title(f"Końcowy wynik grupowania algorytmem K-Medoids (k={k_param})", fontsize=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# ==================== ANIMACJA ====================
st.write("---")
st.subheader("🎬 Animacja procesu uczenia dla K-Medoids")

if st.button("▶️ Odtwórz animację zbieżności"):
    plot_placeholder = st.empty()
    costs = []
    iterations = []

    for step in history:
        anim_fig, (ax_anim1, ax_anim2) = plt.subplots(1, 2, figsize=(16, 6))

        # Rozkład przestrzenny
        ax_anim1.scatter(X[:, 0], X[:, 1], c=step['labels'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
        ax_anim1.scatter(step['medoids'][:, 0], step['medoids'][:, 1], c='red', marker='X', s=200, label='Medoidy',
                         edgecolors='black', zorder=6)
        plot_outliers(ax_anim1)
        ax_anim1.set_title(f"Przebieg algorytmu - Iteracja: {step['iteration']}", fontsize=14)
        ax_anim1.set_xlabel(x_label)
        ax_anim1.set_ylabel(y_label)
        ax_anim1.grid(True, linestyle='--', alpha=0.5)
        ax_anim1.legend()

        # Wykres funkcji kosztu
        costs.append(step['cost'])
        iterations.append(step['iteration'])
        ax_anim2.plot(iterations, costs, marker='o', color='red', linewidth=2)
        ax_anim2.set_title("Wartość funkcji celu (Zbieżność)", fontsize=14)
        ax_anim2.set_xlim(0.5, len(history) + 0.5)
        # Zabezpieczenie przed błędem skalowania osi Y
        y_min_cost, y_max_cost = min([h['cost'] for h in history]), max([h['cost'] for h in history])
        if y_min_cost == y_max_cost:
            ax_anim2.set_ylim(y_min_cost - 10, y_max_cost + 10)
        else:
            ax_anim2.set_ylim(y_min_cost * 0.95, y_max_cost * 1.05)

        ax_anim2.set_xlabel("Numer iteracji")
        ax_anim2.set_ylabel("Całkowity koszt uogólniony (J)")
        ax_anim2.grid(True, linestyle='--', alpha=0.5)

        plot_placeholder.pyplot(anim_fig)
        plt.close(anim_fig)
        time.sleep(0.7)

st.write("---")
st.subheader("Zwracane wartości")
col1, col2 = st.columns(2)

with col1:
    st.write("**Lista baz (Medoidy):**")
    df_medoids = pd.DataFrame(medoids, columns=["Oś X", "Oś Y"])
    df_medoids.index = df_medoids.index + 1
    df_medoids.index.name = "ID"
    st.dataframe(df_medoids, use_container_width=True)

with col2:
    st.write("**Przypisanie punktów do baz:**")
    df_points = pd.DataFrame(X, columns=["Oś X", "Oś Y"])
    df_points["Baza"] = labels.astype(int)
    df_points.index = df_points.index + 1
    df_points.index.name = "Nr punktu"
    st.dataframe(df_points, use_container_width=True)