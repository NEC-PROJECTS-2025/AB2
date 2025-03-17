from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pickle
import numpy as np
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'a9f8b3d1c4e6g7h5i2j0k8l6m3n9p4q7r2s1t5u8v3w0x9y4z6'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1)

# Initialize Database & Bcrypt for Password Hashing
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Initialize Database
with app.app_context():
    db.create_all()

# Load the Model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features)
        predicted_height = float(prediction[0])

        # Pass both inches and cm to the template
        return render_template(
            'predictions/result.html',
            prediction_inches=round(predicted_height, 2),
            prediction_cm=round(predicted_height * 2.54, 2)  # Convert to cm
        )

    return render_template('predictions/index.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics/metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart/flowchart.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists!"

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        print(f"New user registered: {username}")  # Debugging output

        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            return "<h1>Invalid credentials!</h1>"
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
