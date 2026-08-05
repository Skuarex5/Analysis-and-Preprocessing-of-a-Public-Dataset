# 📊 Sensor Data Analysis and Preprocessing

This project demonstrates a complete data analysis and preprocessing workflow using Python.

The program generates synthetic data representing the distance between a rock and a monitoring point. It simulates measurements from both a real sensor and a fictitious sensor with a systematic offset, exports the generated data to CSV files, compares both datasets, calculates a correction model, and visualizes the results.

The project also includes Tkinter interfaces for displaying the comparison data and the mathematical equations used during the analysis.

---

## 🎯 Project Objective

The main objective is to simulate an inaccurate distance sensor and calculate a mathematical correction that makes its measurements approximate the real values.

The fictitious sensor produces measurements approximately `100 km` below the real distance. A linear regression model is used to estimate and correct this systematic error.

---

## ✨ Features

- Generates deterministic synthetic sensor data.
- Simulates real and inaccurate distance measurements.
- Exports the generated data to CSV files.
- Compares the first 20 days of both datasets.
- Applies cubic interpolation for smooth visualization.
- Fits third-degree polynomial models to the data.
- Calculates a linear sensor correction model.
- Computes the coefficient of determination, R².
- Displays comparative graphs with Matplotlib.
- Displays the data in a Tkinter table.
- Displays the mathematical equations used in the analysis.

---

## 📁 Project Structure

```text
Analysis-and-Preprocessing-of-a-Public-Dataset/
├── Sensores.py
├── sensor_ficticio.csv
├── sensor_real.csv
└── README.md
```

### File Description

| File | Description |
|---|---|
| `Sensores.py` | Main Python program for data generation, analysis, visualization, and graphical interfaces |
| `sensor_ficticio.csv` | Measurements generated for the fictitious sensor |
| `sensor_real.csv` | Reference measurements representing the real distance |
| `README.md` | Project documentation |

---

## 🧠 How It Works

### 1. Data Generation

The program generates data for 30 days using:

```python
def generar_datos(n=30, ruido=2):
    np.random.seed(42)

    dias = np.arange(1, n + 1)

    distancia_real = (
        np.linspace(800, 200, n)
        + np.random.normal(0, ruido, n)
    )

    velocidad_real = np.linspace(5, 25, n)

    distancia_sensor = (
        distancia_real
        - 100
        + np.random.normal(0, ruido, n)
    )

    return dias, distancia_real, velocidad_real, distancia_sensor
```

The generated variables are:

- `dias`: days from 1 to 30.
- `distancia_real`: simulated real distance.
- `velocidad_real`: simulated rock velocity.
- `distancia_sensor`: fictitious sensor measurement with an approximate error of `-100 km`.

A fixed random seed is used so that the program generates the same values every time it runs.

---

### 2. CSV Generation

The program creates two Pandas DataFrames and exports them as CSV files.

#### Fictitious sensor dataset

```text
Dia,Distancia_Sensor_Ficticio,Velocidad_Roca
```

#### Real sensor dataset

```text
Dia,Distancia_Real,Velocidad_Roca
```

The files are generated automatically as:

```text
sensor_ficticio.csv
sensor_real.csv
```

---

### 3. Data Comparison

The first 20 days are selected for analysis:

```python
n_comparacion = 20
```

A new DataFrame combines the real and fictitious distance measurements:

```python
df_comparativa = df_sensor_ficticio[
    ["Dia", "Distancia_Sensor_Ficticio"]
].iloc[:n_comparacion].copy()

df_comparativa["Distancia_Real"] = (
    df_sensor_real["Distancia_Real"].iloc[:n_comparacion]
)
```

---

### 4. Polynomial Fitting

A third-degree polynomial is defined as:

```text
y = ax³ + bx² + cx + d
```

It is implemented with:

```python
def polinomio_tercer_grado(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
```

SciPy's `curve_fit` function estimates the polynomial parameters for both datasets.

---

### 5. Linear Sensor Correction

The correction model uses a linear equation:

```text
y = mx + b
```

Where:

- `x` is the fictitious sensor measurement.
- `y` is the estimated real distance.
- `m` is the slope.
- `b` is the intercept.

The function is defined as:

```python
def funcion_lineal(x, m, b):
    return m * x + b
```

The parameters are estimated with:

```python
popt_correccion, _ = curve_fit(
    funcion_lineal,
    df_comparativa["Distancia_Sensor_Ficticio"],
    df_comparativa["Distancia_Real"]
)
```

For the current generated dataset, the approximate correction equation is:

```text
Corrected distance = 0.9974 × sensor distance + 101.8706
```

---

## 📐 Coefficient of Determination

The coefficient of determination evaluates how well the correction model represents the real data.

It is calculated as:

```text
R² = 1 - (SSres / SStot)
```

Where:

- `SSres` is the residual sum of squares.
- `SStot` is the total sum of squares.

For the current dataset, the result is approximately:

```text
R² = 0.9997
```

A value close to `1` indicates that the linear correction model closely matches the real measurements.

---

## 📈 Visualizations

The program generates a figure containing two graphs.

### Real and Fictitious Sensor Comparison

The first graph displays:

- Real distance measurements.
- Fictitious sensor measurements.
- Cubic spline curves.
- Original measurement points.

### Corrected Sensor Comparison

The second graph displays:

- Real distance.
- Fictitious sensor distance.
- Corrected sensor distance.

The corrected values should closely follow the real measurements.

---

## 🖥️ Graphical Interfaces

The project uses Tkinter to create two additional windows.

### Comparative Data Table

The first window displays:

- Day.
- Fictitious sensor distance.
- Real distance.

### Equation Table

The second window displays the equations used in the project:

| Equation | Purpose |
|---|---|
| `y = ax³ + bx² + cx + d` | Third-degree polynomial fitting |
| `y = mx + b` | Linear sensor correction |
| `R² = 1 - (SSres / SStot)` | Evaluation of the correction model |

---

## 📋 Requirements

Make sure you have the following installed:

- Python 3.8 or newer.
- Tkinter.
- NumPy.
- Pandas.
- Matplotlib.
- SciPy.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Skuarex5/Analysis-and-Preprocessing-of-a-Public-Dataset.git
cd Analysis-and-Preprocessing-of-a-Public-Dataset
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install numpy pandas matplotlib scipy
```

Tkinter is normally included with Python on Windows.

On Ubuntu or Debian, install it with:

```bash
sudo apt update
sudo apt install python3-tk
```

---

## 📦 Requirements File

You can create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
scipy
```

Then install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main Python file:

```bash
python Sensores.py
```

On systems where Python 3 uses a separate command:

```bash
python3 Sensores.py
```

---

## 🔄 Execution Sequence

When the program runs, it performs the following operations:

1. Generates 30 days of synthetic data.
2. Creates the real and fictitious sensor DataFrames.
3. Exports both datasets to CSV files.
4. Selects the first 20 days for comparison.
5. Calculates the polynomial fitting parameters.
6. Calculates the linear correction model.
7. Computes the R² score.
8. Displays the Matplotlib graphs.
9. Prints the correction equation and R² value.
10. Displays the comparative data table.
11. Displays the equation table.

> [!NOTE]
> The graphical windows are displayed sequentially. After closing the Matplotlib graph, the comparative table opens. After closing that table, the equation table opens.

---

## 💻 Expected Console Output

The output should be similar to:

```text
Ecuación de corrección: y = 0.9974 * x + 101.8706
Coeficiente de determinación (R²): 0.9997
```

---

## 📄 Dataset Columns

### `sensor_ficticio.csv`

| Column | Description |
|---|---|
| `Dia` | Measurement day |
| `Distancia_Sensor_Ficticio` | Distance measured by the fictitious sensor |
| `Velocidad_Roca` | Simulated rock velocity |

### `sensor_real.csv`

| Column | Description |
|---|---|
| `Dia` | Measurement day |
| `Distancia_Real` | Simulated real distance |
| `Velocidad_Roca` | Simulated rock velocity |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and synthetic data generation |
| Pandas | Dataset creation, manipulation, and CSV export |
| Matplotlib | Data visualization |
| SciPy | Curve fitting and spline interpolation |
| Tkinter | Graphical tables and interfaces |
| CSV | Storage of generated sensor data |

---

## ⚠️ Current Limitations

- The data is synthetic and does not come from a physical sensor.
- The random seed is fixed, so every execution generates the same dataset.
- Only the first 20 days are used for fitting and comparison.
- The sensor error is intentionally simulated as approximately `-100 km`.
- The program overwrites the CSV files every time it runs.
- The graphical windows must be closed sequentially.
- The third-degree polynomial parameters are calculated but are not directly displayed or used in the final graphs.
- The project title mentions a public dataset, but the current implementation generates its own synthetic dataset.

---

## 🚀 Possible Improvements

- Load a real public dataset instead of generating synthetic data.
- Allow the user to select CSV files through the interface.
- Add input fields for the number of days and noise level.
- Save the generated graphs as image files.
- Display the correction equation directly in the interface.
- Display the R² value in the graphical interface.
- Add error metrics such as MAE, MSE, and RMSE.
- Separate data generation, analysis, visualization, and interface code into modules.
- Add unit tests.
- Use command-line arguments for configuration.
- Prevent CSV files from being overwritten without confirmation.
- Display all results in a single Tkinter application window.

---

## 👤 Author

Developed by [Skuarex5](https://github.com/Skuarex5).

---
