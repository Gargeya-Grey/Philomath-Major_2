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

## Losses of the models
dnn_loss = 10.7261
rnn_loss = 4.7291
lstm_loss = 4.6761
conv_lstm_loss = 4.7544


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
    st.subheader("ADR Data Distribution")
    st.image("Hist.jpg", "Histogram of ADR")
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
        st.subheader("Seasonal ARIMA Model Summary")
        st.image("ARIMA/arima_m_summary.JPG", "Arima model summary")
        st.subheader("365 Days of Prediction")
        st.image("ARIMA/Arima_365.jpg", "365 Days of Prediction")
        st.subheader("30,000+ Days of Prediction")
        st.image("ARIMA/Arima_full.jpg", "30,000+ Days of Prediction")
    elif model == models[1]:
        st.write("DNN model selected")
        st.subheader("Deep Neural Network Model Summary")
        st.image("Dnn/dnn_m_summary.JPG", "DNN model summary")
        if st.button("Evaluation Loss"):
            st.subheader(f"{dnn_loss}")
        else:
            st.write("Click the Button for Model Evaluation Loss")
        st.subheader("Prediction on Validation Dataset with Window size: 1")
        st.image("Dnn/Dnn_pred_1.jpg", "Prediction with No Moving average")
        st.subheader("Prediction on Validation Dataset with Window size: 2")
        st.image("Dnn/Dnn_pred_2.jpg", "Prediction with Moving average window")
    elif model == models[2]:
        st.write("RNN model selected")
        st.subheader("Recurrent Neural Network Model Summary")
        st.image("Rnn/rnn_m_summary.JPG", "RNN model summary")
        if st.button("Evaluation Loss"):
            st.subheader(f"{rnn_loss}")
        else:
            st.write("Click the Button for Model Evaluation Loss")
        st.subheader("Prediction on Validation Dataset with Window size: 1")
        st.image("Rnn/Rnn_pred_1.jpg", "Prediction with No Moving average")
        st.subheader("Prediction on Validation Dataset with Window size: 2")
        st.image("Rnn/Rnn_pred_2.jpg", "Prediction with Moving average window")
    elif model == models[3]:
        st.write("LSTM Model Selected")
        st.subheader("Long Short Term Memory Network Model Summary")
        st.image("lstm/lstm_m_summary.JPG", "LSTM model summary")
        if st.button("Evaluation Loss"):
            st.subheader(f"{lstm_loss}")
        else:
            st.write("Click the Button for Model Evaluation Loss")
        st.subheader("Prediction on Validation Dataset with Window size: 1")
        st.image("lstm/Lstm_pred_1.jpg", "Prediction with No Moving average")
        st.subheader("Prediction on Validation Dataset with Window size: 2")
        st.image("lstm/Lstm_pred_2.jpg", "Prediction with Moving average window")
    elif model == models[4]:
        st.write("Convo+LSTM Model Selected")
        st.subheader("Convolutions added Long Short Term Memory Network Model Summary")
        st.image("conv-lstm/conv-lstm_m_summary.JPG", "LSTM model summary")
        if st.button("Evaluation Loss"):
            st.subheader(f"{conv_lstm_loss}")
        else:
            st.write("Click the Button for Model Evaluation Loss")
        st.subheader("Prediction on Validation Dataset with Window size: 1")
        st.image("conv-lstm/Conv-Lstm_pred_1.jpg", "Prediction with No Moving average")
        st.subheader("Prediction on Validation Dataset with Window size: 2")
        st.image("conv-lstm/Conv-Lstm_pred_2.jpg", "Prediction with Moving average window")


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
        view_dataset()
    elif page == options[1]:
        run_ai()

## Note: This analysis clearly shows that model complexity doesn't necessarily mean better results. It depends completely on the type of problem and data itself.
##       Also, realize that increasing the size and complexity of the models deoptimize the time and space complexity of the solution. These models in the experiments
##       might would have given completely different results and efficieny with either another problem statement or data or both.