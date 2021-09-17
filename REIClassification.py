import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,Input,Activation,BatchNormalization,MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler, CSVLogger
import streamlit as st

st.title("Real Estate Image Classification")

X_test=pd.read_pickle('X_test_image_input_data_latest.pkl')
json_file = open("EfficientNetB5_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("EfficientNetB5_model.h5")

def final_fun_1(selected_image='bedroom (419)'):
  image_data=X_test[X_test['image_name'].values==selected_image]['image_data'].values[0]
  pred_prob=loaded_model.predict(image_data)  
  return predict_class(np.argmax(pred_prob))

def predict_class(class_index):
  if(class_index==0):
    return 'backyard'
  elif(class_index==1):
    return 'bathroom'
  elif(class_index==2):
    return 'bedroom'
  elif(class_index==3):
    return 'frontyard'
  elif(class_index==4):
    return 'kitchen'
  else:
     return 'livingRoom'  



image_select = st.selectbox('Select Image: ',X_test['image_name'].values)
image_selection_msg=st.text('You have selected image: '+image_select)
st.image(X_test[X_test['image_name'].values==image_select]['image_data'].values[0])
if st.button('Predict Image Class'):
    image_predicted_class=final_fun_1(image_select)
    st.write('Image Name: '+image_select)
    st.write("Image Predicted Class: "+str(image_predicted_class))
