from flask import Flask, request, render_template_string
from dice import fft_cnr

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        values = request.form['values']
        values_list = list(map(float, values.split(',')))
        result = fft_cnr(values_list)  # Use the imported function
        return render_template_string('''
        <div style="margin: auto; width: 50%; padding: 10px;">
            <h3>Contrast-to-noise estimate: {{result}}</h3>
            <p><a href="/">Try again</a></p>
            <p>Download the open-source code for the full DICE application here:<br />
            <a href="https://github.com/thiebes/DICE" target="_blank">https://github.com/thiebes/DICE</a></p>
            <p>Citation:</br>
            <textarea rows="4" cols="50" readonly>Joseph J. Thiebes. (2023). thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191</textarea></p>
            <p>This is based on a paper that is currently in peer review.</p>
        </div>
        ''', result=result)
    return '''
        <div style="margin: auto; width: 50%; padding: 10px;">
            <form method="post">
                <h1>Estimate Contrast-to-Noise Ratio (CNR)</h1>
                <p>Use this application to estimate the contrast-to-noise ratio
                from a list of comma-separated values representing a one-dimensional
                excited state population profile.</p>

                <p><em><strong>You must first subtract your background (baseline), and
                then normalize so the peak heinght is set to unity, before
                using this tool.</strong></em></p>

                <p>Enter your profile as a list of comma-separated values: <input type="text" name="values">
                <input type="submit" value="Calculate"></p>

                <p>The CNR estimation algorithm works as follows:</p>

                <ol>
                    <li>The data undergoes a unitary (scaled by <i>n</i>^-1/2), single sided, one-dimensional discrete Fourier transform (FT).
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
                <p>Citation:</br>
                <textarea rows="4" cols="50" readonly>Joseph J. Thiebes. (2023). thiebes/DICE. Zenodo. https://doi.org/10.5281/zenodo.10258191</textarea></p>
                <p>This is based on a paper that is currently in peer review.</p>
                <p>Joseph Thiebes: <a href="http://thiebes.org" target="_blank">Website</a> | <a href="https://www.linkedin.com/in/thiebes/" target="_blank">LinkedIn</a> | <a href="https://www.buymeacoffee.com/leufebkxkn" target="_blank">Buy me a coffee</a>
            </form>
        </div>
    '''

if __name__ == '__main__':
    app.run(debug=True)
