from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

#Loading Model
model = pickle.load(open('crop_model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
ms = pickle.load(open('mx.pkl','rb'))

#Webpage Routes
app = Flask(__name__)

@app.route('/')  # Route for Home_page.html
def home():
    return render_template('Home_page.html')

@app.route('/contact_us.html')
def contact_us():
    return render_template('contact_us.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/crop_predict', methods = ['GET'])
def crop_predict():
   return render_template("crop_predict.html", result = None)

@app.route("/predict", methods = ['GET','POST'])
def predict():
    result = None
    if request.method == 'POST':
        
        N = request.form.get('nitrogen')   # Use .get() and lowercase
        P = request.form.get('phosphorus') # Use .get() and lowercase
        K = request.form.get('potassium') # Use .get() and lowercase
        temp = request.form.get('temperature') # Use .get() and lowercase
        humidity = request.form.get('humidity') # Use .get() and lowercase
        ph = request.form.get('ph') # Use .get() and lowercase
        rainfall = request.form.get('rainfall')

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange", 
                    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana", 
                    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 
                    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        #Result Handling
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated.".format(crop)
            return render_template('crop_predict.html', result=result,crop = crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            return render_template('crop_predict.html', result=None)
    return render_template('crop_predict.html', result=result)

#main
if __name__ == "__main__":
    app.run(debug = True)
