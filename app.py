from flask import Flask, jsonify, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('home_page.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            required_fields = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid',
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates',
                'alcohol', 'quality'
            ]

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

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

            # Return JSON if request is from API, HTML if from browser
            if request.accept_mimetypes.best == 'application/json':
                return jsonify({"result": result}), 200

            return render_template('form.html', results=result)

        except ValueError as e:
            return jsonify({"error": f"Invalid data type: {str(e)}"}), 422

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # GET — render the empty form
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)