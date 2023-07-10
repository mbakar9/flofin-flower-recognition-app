from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from flask_wtf import Form
from wtforms import TextField
from werkzeug import secure_filename
import os
import proccess

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'bucokzorbirsifre'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['PROPAGATE_EXCEPTIONS'] = False

cors = CORS(app, resources={r"*": {"origins": "http://localhost:5000"}})
URL = 'http://localhost:5000'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def file_upload():

    message = ""
    error = 0
    
    if request.method == 'POST':
        f = request.files['file']
        if f.filename.split('.')[len(f.filename.split('.')) - 1] == 'jpg' or f.filename.split('.')[len(f.filename.split('.')) - 1] == 'png':
            f.save("static/test.jpg")
            error = 0
            message = 'success'
        else:
            error = 1
            message = 'error'

        return render_template('api.html', message=message)
    
@app.route('/analytics')
def analytics():

    data_dir = os.path.realpath(os.path.dirname(__file__))
    model_path = os.path.join(data_dir, "model.pt")

    message = "error"
    if not os.path.exists(model_path):
        message = "error"
    else:
        message = proccess.classify()

    return render_template('analytics.html', message=message)
    
app.run(debug=True)