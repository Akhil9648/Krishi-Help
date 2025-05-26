# Krishi-Help: Empowering Farmers Through Technology 🌾🤖📈

**Krishi-Help** is a comprehensive web platform designed to assist farmers by providing essential agricultural resources, cutting-edge predictive analytics for crop recommendation and price forecasting, an AI-powered chatbot for instant support, and real-time weather updates. Our mission is to leverage technology to simplify decision-making, enhance productivity, and promote sustainable agricultural practices.

**Live Demo/Deployed Link:** (If you have one, add it here! e.g., `https://krishi-help.herokuapp.com`)
**Team:** Binary_Brains

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Framework Flask](https://img.shields.io/badge/Framework-Flask-blue.svg)](https://flask.palletsprojects.com/)
[![Frontend HTML-CSS-JS](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-orange.svg)](https://developer.mozilla.org/)
[![Powered by Gemini](https://img.shields.io/badge/AI%20Chatbot-Gemini%20API-4285F4.svg)](https://ai.google.dev/)
[![Weather by OpenWeatherMap](https://img.shields.io/badge/Weather%20API-OpenWeatherMap-77A7D3.svg)](https://openweathermap.org/api)

---

## 🌟 Key Features

*   **Information Hub:**
    *   Access to resources on Modern Equipment, Workshops & Training, and Financial Support.
    *   Latest agricultural news from official sources (e.g., Press Information Bureau, harvest records).
*   **Krishi-Bot (AI Assistant):**
    *   Intelligent chatbot powered by Google's Gemini API.
    *   Provides answers to farming queries and information about the Krishi-Help platform.
    *   Supports interaction in both English and Hindi.
*   **Intelligent Crop Recommendation:**
    *   Recommends the most suitable crop based on soil parameters (Nitrogen, Phosphorus, Potassium, pH) and weather conditions (Temperature, Humidity, Rainfall).
    *   Also suggests an appropriate fertilizer based on N, P, K inputs.
*   **Predictive Crop Price Analysis:**
    *   Forecasts average modal prices for various commodities.
    *   Inputs include Month, Commodity Name, State Name, District Name, and Calculation Type.
    *   Helps farmers make informed decisions about selling and market timing.
*   **Real-Time Weather Forecast:**
    *   Provides current weather conditions (temperature, description, main condition) for any specified city.
    *   Integrated with the OpenWeatherMap API.

---

## 📸 Screenshots

*(Consider adding 2-3 key screenshots here directly in the README if your GitHub plan supports it, or link to an imgur album. For now, users can refer to your PPT.)*

*   **Homepage:**
    ![Homepage Screenshot](<link_to_your_homepage_screenshot_if_available_else_remove_this_line>)
    *(Your Screenshot 1)*
*   **Crop Recommendation:**
    ![Crop Recommendation Screenshot](<link_to_your_crop_recommendation_screenshot_if_available_else_remove_this_line>)
    *(Your Screenshot 4)*
*   **Krishi-Bot:**
    ![Chatbot Screenshot](<link_to_your_chatbot_screenshot_if_available_else_remove_this_line>)
    *(Your Screenshot 3)*

---

## 🛠️ Technology Stack

*   **Backend:**
    *   **Python:** Core programming language.
    *   **Flask:** Micro web framework for routing, request handling, and serving HTML.
*   **Frontend:**
    *   **HTML5:** Structure and content of web pages.
    *   **Tailwind CSS:** Utility-first CSS framework for styling.
    *   **JavaScript:** Client-side interactivity (e.g., chatbot API calls, dynamic form submissions for price prediction).
*   **Machine Learning:**
    *   **Scikit-learn:** For building and training predictive models.
    *   **Pandas:** For data manipulation and preprocessing (e.g., one-hot encoding, feature scaling).
    *   **NumPy:** For numerical operations.
    *   **Models:**
        *   Crop Recommendation Model (`crop_model.pkl`)
        *   Fertilizer Recommendation Model (`fertilizer.pkl`)
        *   Crop Price Prediction Model (`crop_price.pkl`)
    *   **Model Persistence:** `pickle` and `joblib` for saving and loading trained models.
*   **External APIs:**
    *   **Google Gemini API:** Powers the Krishi-Bot.
    *   **OpenWeatherMap API:** Provides real-time weather data.
*   **Environment Management:**
    *   `python-dotenv`: For managing API keys and sensitive configurations via `.env` files.
*   **Version Control:**
    *   Git & GitHub.

---

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Akhil9648/Binary_Brains.git
    cd Binary_Brains
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file. See below.)*

4.  **Set up Environment Variables:**
    *   Create a `.env` file in the root directory of the project.
    *   Add your API keys to this file:
        ```env
        OPENWEATHERMAP_API_KEY="YOUR_OPENWEATHERMAP_API_KEY"
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        ```
    *   Replace `"YOUR_..."` with your actual API keys.

5.  **Ensure Model Files are Present:**
    *   The trained model files (`crop_model.pkl`, `sc.pkl`, `mx.pkl`, `fertilizer.pkl`, `crop_price.pkl`, `min_max_scaler.pkl`, `model_columns.pkl`, `original_numerical_cols.pkl`, `original_categorical_cols.pkl`) should be present in the `models/` directory. These are typically generated during the model training phase which is not part of this run-time setup.

6.  **Run the Flask application:**
    ```bash
    python app.py
    ```

7.  Open your web browser and navigate to `http://127.0.0.1:5000/`.

---

### Creating `requirements.txt`

If you don't have a `requirements.txt` file, you can generate one after installing all necessary packages in your virtual environment:

```bash
pip freeze > requirements.txt
