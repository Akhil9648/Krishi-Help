from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests # For making HTTP requests

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.secret_key = 'your_secret_key'  # Change this to something secure

# OpenWeatherMap API Key - TODO: Store this securely as an environment variable
OPENWEATHERMAP_API_KEY = '66fab11b925c16a93768990b7e2336e9'

# MySQL configuration
app.config['MYSQL_HOST'] = 'sql12.freesqldatabase.com'
app.config['MYSQL_USER'] = 'sql12774305'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'sql12774305'
mysql = MySQL(app)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['user_id']
            session['email'] = user['email']
            # Handle remember me checkbox
            if 'remember' in request.form:
                session.permanent = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        if user:
            flash('Email already registered', 'warning')
        else:
            cursor.execute(
                'INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)',
                (full_name, email, hashed_password)
            )
            mysql.connection.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        return render_template('dashboard.html')
    else:
        flash("Please log in first", "warning")
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Add the routes that appear in your navbar
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact.html')

def get_weather_data(city_name, api_key):
    """Fetches weather data from OpenWeatherMap API."""
    api_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # For Celsius
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        
        # Extract relevant information
        weather_info = {
            'city': data.get('name'),
            'temperature': data.get('main', {}).get('temp'),
            'description': data.get('weather', [{}])[0].get('description'),
            'main_condition': data.get('weather', [{}])[0].get('main'),
            'error': None
        }
        # Check for essential missing data that wouldn't raise KeyError/IndexError due to .get()
        if weather_info['temperature'] is None or weather_info['description'] is None or weather_info['main_condition'] is None:
            return {'error': "Could not parse weather data due to missing fields."}
        return weather_info
    except requests.exceptions.RequestException as e:
        # Handle network errors or API errors
        return {'error': f"API request error: {e}"}
    except (KeyError, IndexError, TypeError): # Should be less likely with .get() but good for safety
        # Handle issues with expected JSON structure
        return {'error': "Could not parse weather data due to unexpected structure."}
    except Exception as e:
        # Catch any other unexpected errors
        return {'error': f"An unexpected error occurred: {e}"}

@app.route('/weather')
def weather():
    # For now, use a default city "London"
    city = "London"
    weather_data_dict = get_weather_data(city, OPENWEATHERMAP_API_KEY)
    
    if weather_data_dict and weather_data_dict.get('error'):
        flash(weather_data_dict['error'], 'danger') # Flash the error message
        weather_data_dict = None # Ensure template gets None if there's an error

    return render_template('weather.html', weather_data=weather_data_dict)

if __name__ == '__main__':
    app.run(debug=True)