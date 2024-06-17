from flask import Flask, request, render_template_string
from dice import fft_cnr

app = Flask(__name__)

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
                    return render_template_string('''
                    <div style="margin: auto; width: 50%; padding: 10px;">
                        <h3>Error: Please enter a valid list of comma-separated or space-separated numbers.</h3>
                        <p><a href="/">Try again</a></p>
                    </div>
                    ''')

            try:
                result = fft_cnr(values_list)  # Use the imported function
                return render_template_string('''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Estimate Contrast-to-Noise Ratio (CNR)</title>
                    <link rel="icon" href="./static/favicon.ico" type="image/x-icon">
                    <link rel="icon" href="./static/favicon-16x16.png" sizes="16x16" type="image/png">
                    <link rel="icon" href="./static/favicon-32x32.png" sizes="32x32" type="image/png">
                    <link rel="icon" href="./static/favicon-64x64.png" sizes="64x64" type="image/png">
                    <link rel="apple-touch-icon" href="./static/apple-touch-icon.png" sizes="180x180">
                    <link rel="icon" href="./static/android-chrome-192x192.png" sizes="192x192" type="image/png">
                    <link rel="icon" href="./static/android-chrome-512x512.png" sizes="512x512" type="image/png">
                    <link rel="icon" href="./static/mstile-144x144.png" sizes="144x144" type="image/png">
                    <meta name="msapplication-TileImage" content="./static/mstile-144x144.png">
                    <link rel="manifest" href="./static/site.webmanifest">
                </head>
                <body>
                    <div style="margin: auto; width: 50%; padding: 10px;">
                        <h3>Contrast-to-noise estimate: {{result}}</h3>
                        <p><a href="/">Try again</a></p>
                        <p>Download the open-source code for the full DICE application here:<br />
                        <a href="https://github.com/thiebes/DICE" target="_blank">https://github.com/thiebes/DICE</a></p>
                        <p>Citation for this code:</br>
                        <textarea rows="4" cols="50" readonly>Joseph J. Thiebes. (2023). thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191</textarea></p>
                        <p>Citation for the paper about the research that this code was used to perform:</p>
                        <textarea rows="4" cols="50" readonly>Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. <i>J. Chem. Phys.</i> 28 March 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347</textarea></p>
                    </div>
                </body>
                </html>
                ''', result=result)
            except Exception as e:
                return render_template_string(f'''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Estimate Contrast-to-Noise Ratio (CNR)</title>
                    <link rel="icon" href="./static/favicon.ico" type="image/x-icon">
                    <link rel="icon" href="./static/favicon-16x16.png" sizes="16x16" type="image/png">
                    <link rel="icon" href="./static/favicon-32x32.png" sizes="32x32" type="image/png">
                    <link rel="icon" href="./static/favicon-64x64.png" sizes="64x64" type="image/png">
                    <link rel="apple-touch-icon" href="./static/apple-touch-icon.png" sizes="180x180">
                    <link rel="icon" href="./static/android-chrome-192x192.png" sizes="192x192" type="image/png">
                    <link rel="icon" href="./static/android-chrome-512x512.png" sizes="512x512" type="image/png">
                    <link rel="icon" href="./static/mstile-144x144.png" sizes="144x144" type="image/png">
                    <meta name="msapplication-TileImage" content="./static/mstile-144x144.png">
                    <link rel="manifest" href="./static/site.webmanifest">
                </head>
                <body>
                    <div style="margin: auto; width: 50%; padding: 10px;">
                        <h3>Error processing data: {str(e)}</h3>
                        <p><a href="/">Try again</a></p>
                    </div>
                </body>
                </html>
                ''')
        else:
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Estimate Contrast-to-Noise Ratio (CNR)</title>
                <link rel="icon" href="./static/favicon.ico" type="image/x-icon">
                <link rel="icon" href="./static/favicon-16x16.png" sizes="16x16" type="image/png">
                <link rel="icon" href="./static/favicon-32x32.png" sizes="32x32" type="image/png">
                <link rel="icon" href="./static/favicon-64x64.png" sizes="64x64" type="image/png">
                <link rel="apple-touch-icon" href="./static/apple-touch-icon.png" sizes="180x180">
                <link rel="icon" href="./static/android-chrome-192x192.png" sizes="192x192" type="image/png">
                <link rel="icon" href="./static/android-chrome-512x512.png" sizes="512x512" type="image/png">
                <link rel="icon" href="./static/mstile-144x144.png" sizes="144x144" type="image/png">
                <meta name="msapplication-TileImage" content="./static/mstile-144x144.png">
                <link rel="manifest" href="./static/site.webmanifest">
            </head>
            <body>
                <div style="margin: auto; width: 50%; padding: 10px;">
                    <h3>Error: No data provided. Please enter a valid list of numbers.</h3>
                    <p><a href="/">Try again</a></p>
                </div>
            </body>
            </html>
            ''')
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Estimate Contrast-to-Noise Ratio (CNR)</title>
            <link rel="icon" href="./static/favicon.ico" type="image/x-icon">
            <link rel="icon" href="./static/favicon-16x16.png" sizes="16x16" type="image/png">
            <link rel="icon" href="./static/favicon-32x32.png" sizes="32x32" type="image/png">
            <link rel="icon" href="./static/favicon-64x64.png" sizes="64x64" type="image/png">
            <link rel="apple-touch-icon" href="./static/apple-touch-icon.png" sizes="180x180">
            <link rel="icon" href="./static/android-chrome-192x192.png" sizes="192x192" type="image/png">
            <link rel="icon" href="./static/android-chrome-512x512.png" sizes="512x512" type="image/png">
            <link rel="icon" href="./static/mstile-144x144.png" sizes="144x144" type="image/png">
            <meta name="msapplication-TileImage" content="./static/mstile-144x144.png">
            <link rel="manifest" href="./static/site.webmanifest">
        </head>
        <body>
            <div style="margin: auto; width: 50%; padding: 10px;">
                <form method="post">
                    <h1>Estimate Contrast-to-Noise Ratio (CNR)</h1>
                    <p>Use this application to estimate the contrast-to-noise ratio from a list of comma-separated values representing a one-dimensional excited state population profile.</p>
                    
                    <hr />

                    <p>First, <strong>before using this tool</strong>, you must:<br />
                        <ol><li>subtract your background (baseline), and</li>
                        <li>normalize so the peak height is set to unity.</li></ol>
                    </p>

                    <p><img src="/static/favicon-16x16.png" /> Enter your noisy Gaussian profile as a list of comma-separated values here: <br />
                    <input type="text" name="values">
                    <input type="submit" value="Calculate CNR estimate"></p>

                    <hr />

                    <p>The CNR estimation algorithm works as follows:</p>
                    <ol>
                        <li>The data undergoes a unitary (i.e., scaled by <i>n</i><sup>&#189;</sup>), single sided, one-dimensional discrete Fourier transform (FT).
                        <ul>
                            <li>See <a href="https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html" target="_blank">One-dimensional Discrete Fourier Transform (numpy.fft.rfft)</a>.</li>
                            <li>Within this transformed data set, the signal is positioned at zero frequency, whereas the noise predominantly occupies higher spatial frequencies.</li>
                        </ul>
                        <li>The modulus of each complex value in the transformed data is calculated.</li>
                        <li>The first local minimum subsequent to the signal peak is found.</li>
                        <li>The noise level is determined by calculating the root mean squared (RMS) amplitude for frequencies above the initial local minimum</li>
                        <li>Since the normalized peak amplitude was unity, the CNR is simply the reciprocal of the noise level.</li>
                        <li>The estimate is rounded to the nearest integer value.</li>
                    </ol>

                    <p>Download the open-source code for the full DICE application here:<br />
                    <a href="https://github.com/thiebes/DICE" target="_blank">https://github.com/thiebes/DICE</a></p>
                    <p>Citation for this code:</br>
                    <textarea rows="4" cols="50" readonly>Joseph J. Thiebes. 2023. thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191</textarea></p>
                    <p>Citation for the paper about the research that this code was used to perform:</p>
                    <textarea rows="4" cols="50" readonly>Joseph J. Thiebes, Erik M. Grumstrup; Quantifying noise effects in optical measures of excited state transport. J. Chem. Phys. 28 March 2024; 160 (12): 124201. https://doi.org/10.1063/5.0190347</textarea></p>
                    <p>Joseph Thiebes: <a href="http://thiebes.org" target="_blank">Website</a> | <a href="https://www.linkedin.com/in/thiebes/" target="_blank">LinkedIn</a> | <a href="https://www.buymeacoffee.com/leufebkxkn" target="_blank">Buy me a coffee</a>
                </form>
            </div>
        </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
