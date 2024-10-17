import streamlit as st
import pandas as pd
import joblib

# Título de la aplicación
st.title('Predicción de Precios de Alquiler de Viviendas en Brasil')

# Cargar los modelos entrenados
model_options = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

# Opción para seleccionar el modelo
selected_model = st.selectbox('Seleccione el modelo de Machine Learning', model_options)

# Cargar el modelo seleccionado
modelo = joblib.load(f'{selected_model.replace(" ", "_")}_model.pkl')

# Ingreso de datos por el usuario
area = st.number_input('Área (m²)', value=50)
rooms = st.number_input('Número de habitaciones', value=2)
bathroom = st.number_input('Número de baños', value=1)
parking_spaces = st.number_input('Número de espacios de parqueo', value=1)
hoa = st.number_input('Impuesto HOA (R$)', value=500)
property_tax = st.number_input('Impuesto sobre bienes inmuebles (R$)', value=100)
fire_insurance = st.number_input('Seguro contra incendios (R$)', value=50)

# Selección de opciones
city = st.selectbox('Ciudad', ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte'])
animal = st.selectbox('¿Se permiten animales?', ['Yes', 'No'])
furniture = st.selectbox('¿Está amueblado?', ['Yes', 'No'])

# Botón para predecir
if st.button('Predecir'):
    # Crear un dataframe con los datos de entrada
    input_data = pd.DataFrame({
        'area': [area],
        'rooms': [rooms],
        'bathroom': [bathroom],
        'parking spaces': [parking_spaces],
        'hoa (R$)': [hoa],
        'property tax (R$)': [property_tax],
        'fire insurance (R$)': [fire_insurance],
        'city': [city],
        'animal': [animal],
        'furniture': [furniture]
    })

    # Realizar la predicción
    prediccion = modelo.predict(input_data)

    # Mostrar el resultado
    st.write(f'El precio de alquiler estimado es: R${prediccion[0]:,.2f}')
