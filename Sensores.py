import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import ttk

def generar_datos(n=30, ruido=2):
    np.random.seed(42)
    dias = np.arange(1, n + 1)
    distancia_real = np.linspace(800, 200, n) + np.random.normal(0, ruido, n)
    velocidad_real = np.linspace(5, 25, n)
    distancia_sensor = distancia_real - 100 + np.random.normal(0, ruido, n)
    return dias, distancia_real, velocidad_real, distancia_sensor

dias, distancia_real, velocidad_real, distancia_sensor = generar_datos()

df_sensor_ficticio = pd.DataFrame({
    'Dia': dias,
    'Distancia_Sensor_Ficticio': distancia_sensor,
    'Velocidad_Roca': velocidad_real
})

df_sensor_real = pd.DataFrame({
    'Dia': dias,
    'Distancia_Real': distancia_real,
    'Velocidad_Roca': velocidad_real
})

df_sensor_ficticio.to_csv('sensor_ficticio.csv', index=False)
df_sensor_real.to_csv('sensor_real.csv', index=False)

n_comparacion = 20
df_comparativa = df_sensor_ficticio[['Dia', 'Distancia_Sensor_Ficticio']].iloc[:n_comparacion].copy()
df_comparativa['Distancia_Real'] = df_sensor_real['Distancia_Real'].iloc[:n_comparacion]

def mostrar_tabla():
    root = tk.Tk()
    root.title("Tabla comparativa de los primeros 20 días")
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(frame, columns=('Dia', 'Distancia_Sensor_Ficticio', 'Distancia_Real'), show='headings')
    tree.heading('Dia', text='Día')
    tree.heading('Distancia_Sensor_Ficticio', text='Distancia Sensor Ficticio (km)')
    tree.heading('Distancia_Real', text='Distancia Real (km)')

    for index, row in df_comparativa.iterrows():
        tree.insert('', tk.END, values=(row['Dia'], row['Distancia_Sensor_Ficticio'], row['Distancia_Real']))

    tree.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

def mostrar_ecuaciones():
    ecuaciones = [
        {'Ecuación': 'y = a*x^3 + b*x^2 + c*x + d', 'Descripción': 'Polinomio de tercer grado para ajustar datos de distancia.'},
        {'Ecuación': 'y = m*x + b', 'Descripción': 'Ajuste lineal para la corrección de los datos del sensor.'},
        {'Ecuación': 'R² = 1 - (SS_res / SS_tot)', 'Descripción': 'Coeficiente de determinación para evaluar la calidad del ajuste.'}
    ]

    root = tk.Tk()
    root.title("Ecuaciones utilizadas")
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(frame, columns=('Ecuación', 'Descripción'), show='headings')
    tree.heading('Ecuación', text='Ecuación')
    tree.heading('Descripción', text='Descripción')

    for eq in ecuaciones:
        tree.insert('', tk.END, values=(eq['Ecuación'], eq['Descripción']))

    tree.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

def polinomio_tercer_grado(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

popt_real, _ = curve_fit(polinomio_tercer_grado, df_comparativa['Dia'], df_comparativa['Distancia_Real'])
popt_ficticio, _ = curve_fit(polinomio_tercer_grado, df_comparativa['Dia'], df_comparativa['Distancia_Sensor_Ficticio'])

def funcion_lineal(x, m, b):
    return m * x + b

popt_correccion, _ = curve_fit(funcion_lineal, df_comparativa['Distancia_Sensor_Ficticio'], df_comparativa['Distancia_Real'])

y_fit = funcion_lineal(df_comparativa['Distancia_Sensor_Ficticio'], *popt_correccion)
ss_res = np.sum((df_comparativa['Distancia_Real'] - y_fit) ** 2)
ss_tot = np.sum((df_comparativa['Distancia_Real'] - np.mean(df_comparativa['Distancia_Real'])) ** 2)
r_squared = 1 - (ss_res / ss_tot)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
x_new = np.linspace(1, n_comparacion, 300)
spl_ficticio = make_interp_spline(df_comparativa['Dia'], df_comparativa['Distancia_Sensor_Ficticio'], k=3)
spl_real = make_interp_spline(df_comparativa['Dia'], df_comparativa['Distancia_Real'], k=3)

plt.plot(x_new, spl_ficticio(x_new), color='blue', label='Distancia Sensor Ficticio (Curva)')
plt.plot(x_new, spl_real(x_new), color='orange', label='Distancia Real (Curva)')
plt.scatter(df_comparativa['Dia'], df_comparativa['Distancia_Sensor_Ficticio'], color='blue')
plt.scatter(df_comparativa['Dia'], df_comparativa['Distancia_Real'], color='orange')

plt.xlabel('Día')
plt.ylabel('Distancia (km)')
plt.title('Comparación de Distancia Real y Sensor Ficticio (20 días)')
plt.legend()
plt.grid()

distancia_sensor_corregida = funcion_lineal(df_comparativa['Distancia_Sensor_Ficticio'], *popt_correccion)

plt.subplot(1, 2, 2)
plt.plot(x_new, spl_real(x_new), color='orange', label='Distancia Real (Curva)')
plt.plot(x_new, spl_ficticio(x_new), color='blue', label='Distancia Sensor Ficticio (Curva)')
plt.plot(df_comparativa['Dia'], distancia_sensor_corregida, color='green', linestyle='--', label='Distancia Corregida (Línea Recta)')
plt.scatter(df_comparativa['Dia'], distancia_sensor_corregida, color='green')

plt.xlabel('Día')
plt.ylabel('Distancia (km)')
plt.title('Corrección del Sensor Ficticio (20 días)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print(f"Ecuación de corrección: y = {popt_correccion[0]:.4f} * x + {popt_correccion[1]:.4f}")
print(f"Coeficiente de determinación (R²): {r_squared:.4f}")

mostrar_tabla()
mostrar_ecuaciones()