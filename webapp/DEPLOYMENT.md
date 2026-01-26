# PythonAnywhere Git-Based Deployment

This guide describes how to deploy the DICE CNR webapp to PythonAnywhere using git-based deployment.

## Initial Setup

### 1. Clone Repository on PythonAnywhere

Open a Bash console on PythonAnywhere and run:

```bash
cd ~
git clone https://github.com/thiebes/DICE.git
cd DICE
pip install --user -r requirements.txt
```

### 2. Configure Web App

In the PythonAnywhere Web tab:

1. Click "Add a new web app"
2. Choose "Manual configuration" and Python 3.10 (or latest available)
3. Click through the wizard

### 3. Configure WSGI File

In the Web tab, click on the WSGI configuration file link and replace its contents with:

```python
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
```

Replace `YOUR-USERNAME` with your actual PythonAnywhere username.

### 4. Configure Static Files

In the Web tab, add static file mappings:

- URL: `/static/`
- Directory: `/home/YOUR-USERNAME/DICE/webapp/static/`

Add additional mappings for files in the webapp root:

- URL: `/favicon.ico`
- Directory: `/home/YOUR-USERNAME/DICE/webapp/favicon.ico`

- URL: `/sitemap.xml`
- Directory: `/home/YOUR-USERNAME/DICE/webapp/sitemap.xml`

### 5. Configure Virtual Environment (Optional)

If you want to use a virtual environment:

```bash
cd ~/DICE
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then in the Web tab, set the virtualenv path to:
`/home/YOUR-USERNAME/DICE/venv`

### 6. Reload the Web App

Click the "Reload" button in the Web tab.

## Deployment Workflow

### Deploying Updates

When you have changes to deploy:

1. **Push changes to GitHub** (from your local machine):

   ```bash
   git add .
   git commit -m "your commit message"
   git push origin main
   ```

2. **Pull changes on PythonAnywhere** (in PythonAnywhere Bash console):

   ```bash
   cd ~/DICE
   git pull origin main
   ```

3. **Reload the web app**:
   - Go to the Web tab in PythonAnywhere
   - Click the green "Reload" button

   Or use the API from the command line:

   ```bash
   # Install the pythonanywhere helper if not already installed
   pip install --user pythonanywhere

   # Reload the app
   pa_reload_webapp.py YOUR-USERNAME.pythonanywhere.com
   ```

### Checking Deployment Status

To verify what version is deployed:

```bash
cd ~/DICE
git log -1 --oneline
git status
```

### Rollback Procedure

If you need to rollback to a previous version:

```bash
cd ~/DICE
git log --oneline  # Find the commit hash you want to rollback to
git checkout COMMIT_HASH
```

Then reload the web app.

To return to the latest version:

```bash
git checkout main
git pull
```

## Troubleshooting

### Check Error Logs

In the Web tab, check the error log link to see any Python errors.

### Test Locally First

Before deploying, always test locally:

```bash
cd webapp
python dice_web_app.py
# Visit http://127.0.0.1:5000
```

### Dependencies Issues

If you add new dependencies:

1. Update `requirements.txt` locally
2. Push to GitHub
3. On PythonAnywhere:

   ```bash
   cd ~/DICE
   git pull
   pip install --user -r requirements.txt
   # Reload web app
   ```

### Import Errors

If you see import errors, verify:

- The WSGI file has the correct paths
- The dice package is properly structured with `__init__.py` files
- All dependencies are installed

### Permission Issues

All files should be readable. If you encounter permission issues:

```bash
cd ~/DICE
chmod -R 755 .
```

## Notes

- PythonAnywhere free tier has a daily quota for outbound internet access
- The free tier web app goes to sleep after inactivity and takes a moment to wake up
- Always test changes locally before deploying
- Keep your local repository in sync with GitHub before making changes
