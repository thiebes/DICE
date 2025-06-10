import sys
import os

# Add the project root directory to sys.path
# This allows importing modules from the root directory (e.g., the main dice.py)
# os.path.abspath(__file__) gives the absolute path of the current script
# os.path.dirname() gets the directory of that script (webapp)
# os.path.dirname() again gets the parent of that directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from flask import Flask, send_from_directory, render_template, request

from dicecore.utils import fft_cnr

app = Flask(__name__)

@app.route('/robots.txt')
def robots_txt():
    static_dir = app.static_folder or 'static'
    return send_from_directory(static_dir, 'robots.txt')

@app.route('/sitemap.xml')
def sitemap_xml():
    static_dir = app.static_folder or 'static'
    return send_from_directory(static_dir, 'sitemap.xml')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        values = request.form.get('values', '').strip()
        if values:
            try:
                # Try splitting by commas
                values_list = list(map(float, values.split(',')))
            except ValueError:
                try:
                    # If that fails, try splitting by spaces
                    values_list = list(map(float, values.split()))
                except ValueError:
                    return render_template('error.html', error_message='Please enter a valid list of comma-separated or space-separated numbers.')

            try:
                result = fft_cnr(np.array(values_list))
                return render_template('result.html', result=result)
            except Exception as e:
                return render_template('error.html', error_message=f'Error processing data: {str(e)}')
        else:
            return render_template('error.html', error_message='No data provided. Please enter a valid list of numbers.')
    return render_template('home.html')

if __name__ == '__main__':
    # When running directly, ensure the werkzeug reloader also knows about the path
    # This is often handled by running flask from the project root or using a proper package structure,
    # but for a direct script run, this explicit path addition is important.
    # However, Flask's `app.run()` in debug mode might spawn a new process where sys.path modifications
    # at the top level might not be consistently available without further configuration.
    # For production, a WSGI server (like Gunicorn) would be used, and its configuration
    # would handle the Python path.
    app.run(debug=False)