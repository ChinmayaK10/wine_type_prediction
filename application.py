from flask import Flask,request,render_template
import numpy as np

import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData
application=Flask(__name__)

app=application




@app.route('/')
def home_page():
    return render_template('home_page.html')

@app.route("/predictdata",methods=['POST',"GET"])
def predict_data_point():
  
    if request.method=='POST':
        data = CustomData(
            fixed_acidity=float(request.form.get('fixed_acidity')),
            volatile_acidity=float(request.form.get('volatile_acidity')),
            citric_acid=float(request.form.get('citric_acid')),
            residual_sugar=float(request.form.get('residual_sugar')),
            chlorides=float(request.form.get('chlorides')),
            freesulfur_dioxide=float(request.form.get('free_sulfur_dioxide')),
            totalsulfur_dioxide=float(request.form.get('total_sulfur_dioxide')),
            density=float(request.form.get('density')),
            pH=float(request.form.get('pH')),
            sulphates=float(request.form.get('sulphates')),
            alcohol=float(request.form.get('alcohol')),
            quality=float(request.form.get('quality'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictionPipeline()
        print("mid prediction")
        pred=predict_pipeline.predict(pred_df)
        if pred[0]==0:
            results='white wine'
        else :
            results='red wine'
        return render_template('form.html',results=results)
    return render_template('form.html',results=None)
    
if __name__=='__main__':
    app.run(debug=True)
