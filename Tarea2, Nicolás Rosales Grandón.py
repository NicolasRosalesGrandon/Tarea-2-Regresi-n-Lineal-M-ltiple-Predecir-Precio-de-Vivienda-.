

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Datos entregados
datos = {
    'Superficie_m2': [50, 70, 65, 90, 45],
    'Num_Habitaciones': [1, 2, 2, 3, 1],
    'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
    'Precio_UF': [2500, 3800, 3500, 5200, 2100]
}
df = pd.DataFrame(datos)

# X (features) e y (target)
X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df['Precio_UF']

# Train/Test split (quedan 3 para entrenar y 2 para probar)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Modelo y entrenamiento
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción en el set de prueba
y_pred = model.predict(X_test)

# ---- Salidas ----
print("Coeficientes (UF por unidad):")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:>20s}: {coef:.3f}")
print("Intercepto (UF):", round(model.intercept_, 3))

print("\nPredicciones vs Reales (TEST):")
for real, pred in zip(y_test.values, y_pred):
    print(f"  Real: {real:7.1f} | Pred: {pred:7.1f} | Error abs: {abs(pred-real):6.1f}")

# Métricas pedidas en la semana 4: MAE y R²
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\nMAE: {mae:.2f} UF")
print(f"R² : {r2:.3f} ({r2*100:.1f}%)")
