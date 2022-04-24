##Imports
from turtle import position
import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import plotly.express as px


## Importing the datasets

@st.cache(ttl=600)
def get_data():
    return pd.read_csv("hotel_bookings.csv")

## Importing the model
dnn = tf.keras.models.load_model("Dnn")

## Display Functions:

def view_dataset():
    st.markdown("""
    ---
    ## View Dataset
    """)
    st.subheader("Data Plot")
    df = get_data()
    fig = px.scatter(df[df.adr < 1000].adr)
    st.plotly_chart(fig)
    st.subheader("Data Table")
    st.dataframe(df)
    st.subheader("Data Statistics")
    st.write(df.describe())

def run_ai():
    st.markdown("""
    ---
    ## Run Artificial Intelligence
    """)
    st.subheader("Time to delve into the Future")

    models = ["ARIMA Model", "Deep Neural Network Model", "Recurrent Neural Network Model", "Long Short Term Memory Model", "Convolutional+LSTM Model"]

    model = st.selectbox('Select Model', models, index=0)

    if model == models[0]:
        st.write("Arima model selected")
    elif model == models[1]:
        st.write("DNN model selected")
    elif model == models[2]:
        st.write("RNN Model Selected")
    elif model == models[3]:
        st.write("LSTM Model Selected")
    elif model == models[4]:
        st.write("Convo+LSTM Model Selected")


if __name__ == '__main__':
    st.session_state.page = 0
    ## Title
    st.title("Philomath: Dive into the future")

    ## Styles
    st.write('''
        <style>
            div.row-widget.stRadio > div {
                flex-direction:row; 
                justify-content:space-between;
            }
        </style>
        ''', 
        unsafe_allow_html=True)
    ## Radio Buttons
    options =  ["View Dataset", "Run Artificial Intelligence"]
    page = st.radio("Select", options, index=0)

    if page == options[0]:
        st.session_state.page = 0  
    elif page == options[1]:
        st.session_state.page = 1

    if st.session_state.page == 0:
        view_dataset()
    elif st.session_state.page == 1:
        run_ai()
