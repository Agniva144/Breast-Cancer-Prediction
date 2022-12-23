from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('BreastCancerPredictionModel.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

    


class run_model():
    def __init__(self) -> None:
        pass
    
    def model_predict(val1,val2,val3,val4,val5):
        arr=np.array([val1,val2,val3,val4,val5])
        arr=arr.astype(np.float64)
        result=model.predict([arr])

        if result==1:
            return "    Patient predicted with Cancer possiblity."
        else:
            return "    Result is normal."


# First

@app.route('/predict',methods=['GET','POST'])
def predict():
    val1=request.form['radius']
    val2=request.form['texture']
    val3=request.form['perimeter']
    val4=request.form['area']
    val5=request.form['smoothness']

    # arr=np.array([val1,val2,val3,val4,val5])
    # arr=arr.astype(np.float64)
    # pred=model.predict([arr])

    pred=run_model.model_predict(val1,val2,val3,val4,val5)

    return render_template('index.html',data=pred)

if __name__=='__main__':
    app.run(debug=True)