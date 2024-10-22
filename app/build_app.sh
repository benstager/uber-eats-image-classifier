# build requirements
pip install -r requirements.txt

# build app in prod environment, listens on port 10000
gunicorn -b 0.0.0.0:10000 app:app