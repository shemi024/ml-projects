from flask import Flask,render_template,request
import joblib
import numpy as np
app=Flask(__name__)
#model=pickle.load(open('model (1).pkl','rb'))
model=joblib.load('str.h5')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
     int_features = [int(x) for x in request.form.values()]
     final_features = np.array(int_features).reshape(1,10)
     print(final_features)
     prediction = model.predict(final_features)
     L_collection = {0: "No strok", 1: "strok"}
     result=L_collection[prediction[0]]
     print(result)

     return render_template("result.html", prediction_text=f"YOU HAVE :  {result} ")
if __name__ =="__main__":
     app.run(port=8000)

