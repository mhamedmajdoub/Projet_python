from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

model=joblib.load('model.joblib')
le=joblib.load('le.joblib')
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

# To run open the directory in cmd and excute this command '''flask --app file_name.py --debug run'''

@app.route('/predict', methods=['POST'])
def predict():
    features =[] #get inputs from user
    # On transforme les entrées de l'utilisateur en nombres tout en acceptant les entrées majuscules et miniscules
    features.append(le.fit_transform([request.form['Road']])[0])
    features.append(le.fit_transform([request.form['County']])[0])
    features.append(le.fit_transform([request.form['City']])[0])
    features.append(request.form["housing_median_age"])
    features.append(request.form["total_rooms"])
    features.append(request.form["total_bedrooms"])
    features.append(request.form["population"])
    features.append(request.form["households"])
    features.append(request.form["median_income"])
    near_by_ocean_feature=request.form["ocean_proximity"]
    if near_by_ocean_feature=="1H OCEAN":
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif near_by_ocean_feature=="INLAND":
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif near_by_ocean_feature=="NEAR BAY":
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    elif near_by_ocean_feature=="NEAR OCEAN":
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
    features = np.array(features) #convert to numpy array
    prediction = model.predict([features]) #make prediction
    prediction = round(int(prediction[0]), 2) # round the output up to 2 and print the first element of the list which is the prediction 
    print(prediction) #print prediction to the terminal
# Redirect to result page with prediction as query parameter
    return redirect(url_for('result', prediction=prediction))

@app.route('/result')
def result():
    # Get prediction from query parameter
    prediction = request.args.get('prediction')
    # Render result page with prediction
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
