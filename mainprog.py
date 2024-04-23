
from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/service')
def servicepage():
    return render_template('services.html')

@app.route('/coconut')
def coconutpage():
    return render_template('Coconut.html')

@app.route('/cocoa')
def cocoapage():
    return render_template('cocoa.html')

@app.route('/arecanut')
def arecanutpage():
    return render_template('arecanut.html')

@app.route('/paddy')
def paddypage():
    return render_template('paddy.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')





@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("pregnant.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO user(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("pregnant.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM user where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home1.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')



@app.route('/info')
def predictin():
   return render_template('info.html')

# Load data and train the model
data = pd.read_csv('updated_dataset.csv')
X = data.drop('Diseased', axis=1)
y = data['Diseased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=42)
rf_classifier.fit(X_train, y_train)



# Define route for prediction
@app.route('/info', methods=['POST', 'GET'])
def predcrop():
    if request.method == 'POST':
        # Retrieve form data
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        PH = float(request.form['PH'])
        Water_content = float(request.form['Water_content'])
        LDR = float(request.form['LDR'])
        CO2 = float(request.form['CO2'])



        # Predict target based on user input
        prediction = predict_target(N, P, K, Temperature, Humidity, PH, Water_content, LDR, CO2)

        # Calculate accuracy and generate classification report
        y_pred = rf_classifier.predict(X_test)
        accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred) * 100)
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Retrieve precision, recall, and f1-score for class '0' if available, otherwise set them to 0
        precision_0 = "{:.2f}".format(class_report_dict.get('0', {'precision': 0})['precision'] * 100)
        recall_0 = "{:.2f}".format(class_report_dict.get('0', {'recall': 0})['recall'] * 100)
        f1_score_0 = "{:.2f}".format(class_report_dict.get('0', {'f1-score': 0})['f1-score'] * 100)

        # Retrieve precision, recall, and f1-score for class '1' if available, otherwise set them to 0
        precision_1 = "{:.2f}".format(class_report_dict.get('1', {'precision': 0})['precision'] * 100)
        recall_1 = "{:.2f}".format(class_report_dict.get('1', {'recall': 0})['recall'] * 100)
        f1_score_1 = "{:.2f}".format(class_report_dict.get('1', {'f1-score': 0})['f1-score'] * 100)

        # Pass prediction, accuracy, and classification report to the template for rendering
        return render_template('resultpred.html', prediction=prediction, accuracy=accuracy,
                               precision_0=precision_0, recall_0=recall_0, f1_score_0=f1_score_0,
                               precision_1=precision_1, recall_1=recall_1, f1_score_1=f1_score_1)

    # Render the form if request method is GET
    return render_template('info.html')

# Function to predict the target based on user input
def predict_target(N, P, K, Temperature, Humidity, PH, Water_content, LDR, CO2):
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'Temperature': [Temperature],
        'Humidity': [Humidity],
        'PH': [PH],
        'Water_content': [Water_content],
        'LDR': [LDR],
        'CO2': [CO2]

    })

    # Predict target for user input
    prediction = rf_classifier.predict(user_data)
    print("Predicted label : ",prediction)

    # Determine if the prediction is close to 0 or 1
    if prediction == 0:
        return "Plant is Normal"
    else:
        return "Plant is Diseased"



@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)


