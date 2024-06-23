from flask import Flask, send_from_directory, render_template, request
from dice import fft_cnr

app = Flask(__name__)

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt')

@app.route('/sitemap.xml')
def sitemap_xml():
    return send_from_directory(app.static_folder, 'sitemap.xml')

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
                result = fft_cnr(values_list)  # Use the imported function
                return render_template('result.html', result=result)
            except Exception as e:
                return render_template('error.html', error_message=f'Error processing data: {str(e)}')
        else:
            return render_template('error.html', error_message='No data provided. Please enter a valid list of numbers.')
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=False)