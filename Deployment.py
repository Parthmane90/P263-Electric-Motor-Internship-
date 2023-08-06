import pandas as pd
import streamlit as st
import numpy as np
from sklearn import *
import pickle



with open("D:\Data science\Projects\lightgbm_model.pkl", "rb") as file:
        model=pickle.load(file)

#image="https://previews.123rf.com/images/nosua/nosua1609/nosua160900599/63815755-electric-motor-in-disassembled-state-3d-illustration-on-a-white-background.jpg"
image="data:image/png;base64,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"
def add_bg_from_url(link):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({link});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url(image)



st.title("Motor Speed Prediction :racing_motorcycle:")

st.header("Enter Input to Predict Motor Speed")
st.subheader("Data must be standardized")

ambient=st.number_input("Enter Ambient temprature of Motor")

coolant=st.number_input("Enter Coolant temprature for Motor")

u_d= st.number_input("Enter Voltage d Component (u_d)")

u_q= st.number_input("Enter Voltage q Component (u_q)")

torque=st.number_input("Enter Torque of Motor")

i_d=st.number_input("Enter Current d Component (i_d)")

i_q=st.number_input("Enter Current q Component (i_q)")

pm=st.number_input("Enter Permanent Magnet value of Motor (PM)")

stator_yoke=st.number_input("Enter Stator yoke Temprature")

stator_tooth=st.number_input("Enter Stator Tooth Temprature")

stator_winding=st.number_input("Enter Stator Winding Temprature")

profile_id=st.selectbox("Select Profile id",(4,  6, 10, 11, 20, 27, 29, 30, 31, 32, 36, 41, 42, 43, 44, 45, 46,
       47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
       64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81,
       72), help="Profile is Unique Measurement session Select Appropriate session for Accuracy")


if st.button("Predict"):
    data=pd.DataFrame({"ambient":ambient,"coolant":coolant,"u_d":u_d,"u_q":u_q,"torque":torque,"i_d":i_d,"i_q":i_q,"pm":pm,"stator_yoke":stator_yoke,"stator_tooth":stator_tooth,"stator_winding":stator_winding, "profile_id":profile_id}, index=[0])
    st.write(data)
    st.write(model.predict(data))

new_d = st.file_uploader("Choose a csv file", help="Columns names must be 'ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id'")

column=('ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id')


if st.button("Read and Predict"):
    try:
        prediction=pd.read_csv(new_d)
        if "Unnamed: 0" in prediction.columns:
            prediction.drop("Unnamed: 0", axis=1, inplace=True)
        if prediction.shape[1]!=12:
            st.write(f"Uploaded File has {prediction.shape[1]} file must have 12 columns ")
            st.write(f"Columns should be 'ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id'")
        else:
            st.write(prediction)
            st.write(f"FIle uploaded Successfully. File has {prediction.shape[0]} Rows and {prediction.shape[1]} Columns")
            prediction["motor_speed"]=model.predict(prediction)
            st.write("Model has Successfully predicted Motor Speed for given input dataset Please download file to view it")
            #st.download_button("Click to download file",prediction,file_name="Predicted Motor Speed.csv",mime='text/csv')
    except:
        #prediction=pd.read_csv(new_d)
        st.write("Unable to read file please check the file format...")

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(prediction)

    st.download_button(
        label="Click to download file",
        data=csv,
        file_name='Predicted Motor Speed.csv',
        mime='text/csv',
    )




#st.download_button("Click to download file",data=prediction,file_name="Predicted Motor Speed.csv")