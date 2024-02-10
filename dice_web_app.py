#########################################################################
# Diffusion Insight Computation Engine (DICE) simulates optical         #
# measures of diffusion in optoelectronic semiconducting materials      #
# using experimental parameters, and evaluates the precision of         #
# composite fitting methods of estimating the diffusion coefficient.    #
#                                                                       #
# Copyright (C) 2023-2024 Joseph J. Thiebes                             #
#                                                                       #
# This material is based upon work supported by the National Science    #
# Foundation under Grant No. 2154448. Any opinions, findings, and       #
# conclusions or recommendations expressed in this material are those   #
# of the author(s) and do not necessarily reflect the views of the      #
# National Science Foundation.                                          #
#                                                                       #
# This work is licensed under the Creative Commons Attribution 4.0      #
# International License. To view a copy of this license, visit          #
# http://creativecommons.org/licenses/by/4.0/ or send a letter to       #
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.          #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  #
#                                                                       #
# You should include a copy of the license or a link to it with         #
# every copy of the work you distribute. You can do this by             #
# including a link to the license in your README.md file or             #
# documentation.                                                        #
#########################################################################

# This is a web app to serve the DICE program, using Flask.
#
# Presently it only uses the DICE function to estimate CNR from an 
# excited state population profile as a list of comma-separated values.
# The app uses the Flask framework to serve the function to web users.

from flask import Flask, request, render_template_string
from dice import fft_cnr

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        values = request.form['values']
        values_list = list(map(float, values.split(',')))
        result = fft_cnr(values_list)  # Use the imported function
        return render_template_string('''<h1>Result: {{result}}</h1>
                                         <a href="/">Try again</a>''', result=result)
    return '''
        <form method="post">
            Comma-separated values: <input type="text" name="values">
            <input type="submit" value="Calculate">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)

