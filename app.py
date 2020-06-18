from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle

# load the model from disk
predictor_model=pickle.load(open('random_forest_regression_model_Sunanda.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # df=pd.read_csv('real_2018.csv')
    # my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    # my_prediction=my_prediction.tolist()

    if request.method == 'POST':
        avg_temp = float(request.form['avgtemp'])
        maximum_temp = float(request.form['maxtemp'])
        minimum_temp = float(request.form['mintemp'])
        at_pressure_sea_lvl = float(request.form['atmospresealvl'])
        avg_rel_humidity = float(request.form['avgrelhumid'])
        average_visibility =  float(request.form['avgvis'])
        average_wind_speed =  float(request.form['avgwindspeed'])
        maximum_sustained_wind_speed = float(request.form['maxsustwindspeed'])

        data = np.array([[avg_temp, maximum_temp, minimum_temp, at_pressure_sea_lvl, avg_rel_humidity, average_visibility, average_wind_speed, maximum_sustained_wind_speed]])
        my_prediction = predictor_model.predict(data)

        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)