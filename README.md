# K-Medoids Clustering Analysis 

This project is a complete, from-scratch implementation of the **K-Medoids (PAM - Partitioning Around Medoids)** unsupervised machine learning algorithm. It is built strictly using core mathematical operations, without relying on pre-built clustering functions from ML libraries.

The project features a highly interactive GUI designed to visually demonstrate:
* The algorithm's mathematical robustness against data noise and outliers (compared side-by-side with K-Means).
* A step-by-step animation of the algorithm's learning process and the monotonic convergence of its cost function.

## Technical Requirements

The project is written in Python. To run the dashboard and perform the mathematical operations, you need the following libraries:

* `streamlit` (GUI framework)
* `numpy` (vectorized calculations and distance matrices)
* `matplotlib` (generating static plots and animation frames)
* `pandas` (tabularizing the final output)
* `scikit-learn` (used exclusively for dataset generation and the K-Means comparison)

### Quick Installation
Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

Start the local server by running:

```bash
streamlit run main.py
```

Your default web browser will automatically open the application (usually is http://localhost:8501).
