from flask import Flask,jsonify,render_template,request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return {"message": "Wine Prediction API is running"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = CustomData(
            fixed_acidity=float(data.get('fixed_acidity')),
            volatile_acidity=float(data.get('volatile_acidity')),
            citric_acid=float(data.get('citric_acid')),
            residual_sugar=float(data.get('residual_sugar')),
            chlorides=float(data.get('chlorides')),
            freesulfur_dioxide=float(data.get('free_sulfur_dioxide')),
            totalsulfur_dioxide=float(data.get('total_sulfur_dioxide')),
            density=float(data.get('density')),
            pH=float(data.get('pH')),
            sulphates=float(data.get('sulphates')),
            alcohol=float(data.get('alcohol')),
            quality=float(data.get('quality'))
        )

        pred_df = input_data.get_data_as_data_frame()

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(pred_df)

        result = "white wine" if prediction[0] == 0 else "red wine"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8000)