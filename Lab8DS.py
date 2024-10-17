import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar datos
data = pd.read_csv('houses_to_rent_v2.csv')

# Preprocesamiento de datos
numeric_features = ['area', 'rooms', 'bathroom', 'parking spaces', 'hoa (R$)', 'property tax (R$)',
                    'fire insurance (R$)']
categorical_features = ['city', 'animal', 'furniture']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# División de los datos
X = data.drop('total (R$)', axis=1)
y = data['total (R$)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición de los modelos
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}


# Función para mostrar las métricas de cada modelo
def print_metrics(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{model_name}:')
    print(f'  RMSE: {rmse}')
    print(f'  MAE: {mae}')
    print(f'  R2: {r2}\n')

    return rmse, mae, r2


# Lista para almacenar las métricas de cada modelo
metrics = []

# Entrenar, guardar y evaluar los modelos
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f'{name.replace(" ", "_")}_model.pkl')

    # Predicción
    y_pred = pipeline.predict(X_test)

    # Mostrar métricas
    rmse, mae, r2 = print_metrics(y_test, y_pred, name)

    # Guardar las métricas
    metrics.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

    # Gráfico de dispersión (Predicción vs Real)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Línea ideal
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'{name} - Valores Reales vs Predicciones')
    plt.show()

# Crear un DataFrame con las métricas
metrics_df = pd.DataFrame(metrics)

# Gráfico de barras comparando las métricas
metrics_df.set_index('Model', inplace=True)
metrics_df[['RMSE', 'MAE', 'R2']].plot(kind='bar', figsize=(10, 6))
plt.title('Comparación de Métricas entre Modelos')
plt.ylabel('Valor de la Métrica')
plt.show()


