
import numpy as np
import pickle


heart_disease_model = pickle.load(open('C:/Users/hp/Desktop/Heartpredictor/heart_disease_model.sav','rb'))


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)


input_data_as_numpy_array= np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = heart_disease_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
