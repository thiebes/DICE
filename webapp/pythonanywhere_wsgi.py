# WSGI configuration file for PythonAnywhere
#
# Copy the contents of this file to your WSGI configuration file in the
# PythonAnywhere Web tab.
#
# Replace YOUR-USERNAME with your actual PythonAnywhere username.

import sys
import os

# Add your project directory to the sys.path
project_home = '/home/YOUR-USERNAME/DICE'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Add the webapp directory
webapp_path = os.path.join(project_home, 'webapp')
if webapp_path not in sys.path:
    sys.path.insert(0, webapp_path)

# Import Flask app
from dice_web_app import app as application

# Optional: Set Flask configuration
# application.config['DEBUG'] = False
