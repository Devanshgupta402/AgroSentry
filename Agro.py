import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

def predict_condition(measurements, model):
    prediction = model.predict(measurements)
    return prediction[0]

def main():
    st.title("Plant Condition Detector")
    st.write("Welcome to the Plant Condition Detector!")

    # Load the trained model
    model = train_model()

    # Buttons on the home page
    option = st.sidebar.radio("Navigation", ["Home", "AI", "Manual"])

    if option == "Home":
        st.write("This is the home page.")
    elif option == "AI":
        ai_page(model)
    elif option == "Manual":
        manual_page()

def ai_page(model):
    st.title("AI Page")
    st.write("Welcome to the AI Page!")

    # Buttons on the AI page
    if st.button("Check List"):
        checklist()

    if st.button("Let AI show its wonders"):
        ai_wonders(model)

def manual_page():
    st.title("Manual Page")
    st.write("Welcome to the Manual Page!")

    # Buttons on the Manual page
    if st.button("DAP"):
        dap_info()

    if st.button("Water"):
        water_info()

    if st.button("Epsom Salt"):
        epsom_salt_info()

    if st.button("Fertilizer"):
        fertilizer_info()

    if st.button("Manure"):
        manure_info()

    if st.button("Pesticides"):
        pesticides_info()

def checklist():
    st.title("Check List")
    st.write("This is the checklist page.")

def ai_wonders(model):
    st.title("AI Wonders")
    st.write("This is the AI wonders page.")

    st.write("Let's predict the plant condition!")
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

    measurements = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    condition = predict_condition(measurements, model)
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.write(f"Predicted plant condition: {species[condition]}")

def dap_info():
    st.title("DAP Information")
    st.write("This is the DAP information page.")

def water_info():
    st.title("Water Information")
    st.write("This is the water information page.")

def epsom_salt_info():
    st.title("Epsom Salt Information")
    st.write("This is the Epsom salt information page.")

def fertilizer_info():
    st.title("Fertilizer Information")
    st.write("This is the fertilizer information page.")

def manure_info():
    st.title("Manure Information")
    st.write("This is the manure information page.")

def pesticides_info():
    st.title("Pesticides Information")
    st.write("This is the pesticides information page.")


if __name__ == "main":
    main()
