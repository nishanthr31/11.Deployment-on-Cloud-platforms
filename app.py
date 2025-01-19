from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
# Load the pre-trained model
filename = 'FinalizedModel_For_SalaryPrediction.sav'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        yoe = float(request.form['yoe'])

        # Format the input data into a numpy array
        input_data = np.array([[yoe]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        print(prediction)
        # Map prediction to result
        result = prediction
        #result=prediction
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


