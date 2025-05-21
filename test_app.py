import unittest
from unittest.mock import patch, MagicMock
import sys
import requests # For requests.exceptions
import os

# Add the parent directory to the Python path to allow importing 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app # Import the Flask app instance

class WeatherAppTests(unittest.TestCase):

    def setUp(self):
        """Set up test client and other test variables."""
        app.testing = True
        self.client = app.test_client()
        # It's good practice to disable WTF_CSRF_ENABLED for tests if you use Flask-WTF
        app.config['WTF_CSRF_ENABLED'] = False 
        app.config['TESTING'] = True


    def test_weather_route_status_code(self):
        """Test that the /weather route returns a 200 OK status."""
        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200)

    @patch('app.requests.get')
    def test_weather_information_display_mocked_success(self, mock_get):
        """Test that weather information is displayed correctly with a mocked successful API call."""
        # Configure the mock to return a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'London',
            'main': {'temp': 15.0},
            'weather': [{'description': 'scattered clouds', 'main': 'Clouds'}]
        }
        mock_get.return_value = mock_response

        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200)
        
        response_data = response.data.decode('utf-8')
        self.assertIn('Weather in London', response_data)
        self.assertIn('15.0°C', response_data)
        self.assertIn('Clouds', response_data) # Main condition
        self.assertIn('Scattered clouds', response_data) # Description (capitalized by template filter)

    @patch('app.requests.get')
    def test_weather_api_failure_mocked(self, mock_get):
        """Test that a user-friendly error message is shown if the weather API fails."""
        # Configure the mock to simulate an API failure (e.g., raise an exception)
        mock_get.side_effect = requests.exceptions.RequestException("API is down")

        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200) # Page should still load

        response_data = response.data.decode('utf-8')
        # Check for the flashed message content
        self.assertIn('API request error: API is down', response_data)
        # Check that the fallback message in the weather-data div is NOT there if a flash message is shown
        # Or, if no specific weather_data.error is shown, check for the generic one.
        # Based on current app.py, the flashed message takes precedence.

    @patch('app.requests.get')
    def test_weather_api_failure_bad_status_mocked(self, mock_get):
        """Test that a user-friendly error message is shown if the weather API returns a bad status."""
        # Configure the mock to return a non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_get.return_value = mock_response

        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200) # Page should still load

        response_data = response.data.decode('utf-8')
        self.assertIn('API request error', response_data) # Check for the flashed message
        self.assertIn('Server Error', response_data) 


    @patch('app.requests.get')
    def test_weather_api_malformed_response_mocked(self, mock_get):
        """Test handling of malformed JSON response from the weather API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = { # Missing 'main' or 'weather' keys
            'name': 'London',
            # 'main': {'temp': 15.0}, # Missing
            # 'weather': [{'description': 'scattered clouds', 'main': 'Clouds'}] # Missing
        }
        mock_get.return_value = mock_response

        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200)
        response_data = response.data.decode('utf-8')
        # Check for the flashed error message
        self.assertIn('Could not parse weather data due to missing fields.', response_data)
        # Check for the generic message in the weather-data div because weather_data will be None
        self.assertIn('Could not retrieve weather data at this time. Please try again later.', response_data)


if __name__ == '__main__':
    # Need to import requests for the side_effect in test_weather_api_failure_mocked
    import requests 
    unittest.main()
