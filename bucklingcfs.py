import streamlit as st
import pandas as pd
import sklearn
import numpy as np




#ML MODEL

data=pd.read_csv(r"C:\Users\User\Desktop\ml project\training1.csv")

data.drop(index=0,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=data
dfle.BOUNDARYCONDITIONS=le.fit_transform(dfle.BOUNDARYCONDITIONS)
print(dfle)
X= dfle.drop(columns=['BucklingLoad'])
y= dfle['BucklingLoad'] 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
num_splits=5
kfold=KFold(n_splits=num_splits,shuffle=True,random_state=42)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
model=GradientBoostingRegressor()
model=GradientBoostingRegressor(learning_rate=0.1,max_depth=3,n_estimators=200,subsample=0.6,random_state=42)
model.fit(X,y)

#WEB APP 

st.title("Buckling Load Predictor ")
st.write("This app predicts the buckling load of cold formed steel channel sections")
st.subheader("Geometrical Properties of the column")
#image=Image.open(r'C:\\Users\\User\\Desktop\\training data\\cfs.jpg')
#st.image(image,caption='CROSS SECTION OF CFS CHANNEL SECTION',use_column_width=True)
length=st.number_input("Height of the column(mm)",min_value=0.00,max_value=10000.00)
webdepth=st.number_input("WEB DEPTH(mm)",min_value=0.00,max_value=10000.00)
flangewidth=st.number_input("FLANGE WIDTH(mm)",min_value=0.00,max_value=10000.00)
thickness=st.number_input("THICKNESS(mm)",min_value=0.00,max_value=10000.00)
liplength=st.number_input("LIP LENGTH (mm)",min_value=0.00,max_value=10000.00)
options=['Fixed','Hinged']
boundary=st.selectbox("BOUNDARY CONDITIONS",options)
if boundary=='Fixed' :
   bc=0
else:
   bc=1
#hbyt=st.number_input("h/t  ratio",min_value=1.00,max_value=140.00)

#bbyt=st.number_input("b/t ratio",min_value=6.445183,max_value=126.666667)

features=pd.DataFrame({'Length':[length],'WEB DEPTH':[webdepth],'FLANGE WIDTH':[flangewidth],'THICKNESS':[thickness],'LIP LENGTH':[liplength],'BOUNDARYCONDITIONS':[bc]})

predictions=model.predict(features)
#if all(features == 0):
 #  predictions=0
  # buckling=st.write(f'Buckling load(KN):{predictions}KN')
#elseif :
 #  {
  # predictions=model.predict(features)
   #hbyt=webdepth/thickness
   #bbyt=flangewidth/thickness
   #st.write("web depth/thickness ratio",hbyt)
   #st.write("flange width/thickness ratio",bbyt)
buckling=st.write(f'Buckling load(KN):{predictions}KN')
   

    
    
   
