from flask import Flask, jsonify, g, Response, request, flash, redirect, url_for, send_from_directory, render_template
from werkzeug.exceptions import HTTPException
from waitress import serve
from PIL import Image
from werkzeug.utils import secure_filename
from matplotlib import pyplot as plt
from tabulate import tabulate

import os
import io
import cv2
import numpy as np
import datetime 
import time
import socket
import requests
import pandas as pd

#visualizing images in output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

UPLOAD_FOLDER = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\upload"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
INDEX = r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\templates\index.html"
TEMPLATE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(INDEX)))
TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HOST_NAME = 'TM Vision HTTP Server'
HOST_PORT = 80

nu = 0.75



# ========================================================== SYSTEM =================================================================
@app.errorhandler(HTTPException) 
#Handle an exception that did not have an error handler associated with it, or that was raised from an error handler. 
#This always causes a 500 InternalServerError.
def handleException(e):
    '''Return HTTP errors.'''
    TRIMessage(e)
    return e

@app.errorhandler(400)
def bad_request(e):
    return render_template("400.html"), 400

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return render_template("405.html"), 405

#---------------------------- Utility functions--------------------------------------------------------------------------------------
def TRIMessage(message):
    print(f'\n[{datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().isoformat(timespec="milliseconds")}] {message}')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def jsonresponse(cx,cy,box_w, box_h, task, theta, message, output):
    #type check
    if type(cx) != int:
        print("cx has to be an integer number")
        cx = int(cx)
    if type(cy) != int:
        print("cy has to be an integer number")
        cy = int(cy)
    if type(box_w) != float:
        print("box_w has to be a float number")
        box_w = float(box_w)
    if type(box_h) != int:
        print("box_h has to be a float number")
        box_h = float(box_h)
    if type(task) != str :
        print("task has to be a string")
        task = str(task)
    if type(theta) != float:
        print("theta has to be a float number")
        theta = float(theta)
    if message != "success" and message != "fail":
        print("message has to be a string containing either success or fail")
        return redirect(request.url)
    if cx == 0 and cy == 0 and box_w == 0 and box_h == 0: #used when the json contains a failure
        result = { 
                    "message": message,
                    "result" : output
                }
        return result
    result = {
            "message": "success", #This message has to be either "success" or "fail"...
            "annotations":[
                {
                    "box_cx": float(str(cx)),
                    "box_cy": float(str(cy)),
                    "box_w": float(str(box_w)),
                    "box_h": float(str(box_h)),
                    "label": str(task),
                    "score": float(str(1.000)),
                    "rotation": float(str(theta))

                }
            ],
            "result": str(output) #"Image if success, None if fail...."
        }
    return jsonify(result)

# =========================================================== GET ===================================================================

@app.route('/') 
#by deafult the app.route expects a get request. Here I put an index page for the user to open on the pc side.
def index():
    return render_template('index.html')

@app.route('/api/<string:m_method>', methods=['GET']) 
#dummy GET to try the connection over the TM robot side in the vision node settings
def get(m_method):
    # user defined method
    #result = dict()
    
    if m_method == 'status':
        result = jsonresponse(0,0,0,0,0,0,"success",None)
    else:
        result = jsonresponse(0,0,0,0,0,0,"fail",None)
    return result

# ============================================================== POST ===============================================================
@app.route('/api/<string:m_method>', methods=['POST'])
def post(m_method):
    #get key/value
    parameters = request.args
    model_id = parameters.get('model_id')
    TRIMessage(f'model_id: {model_id}')
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    #check key/value
    if model_id is None:
        TRIMessage('model_id is not set')
        result=jsonresponse(0,0,0,0,0,0,"fail",None)
        return result
    
    #--------------------------------------------SAVING IMAGE ON PC------------------------------------------------------------------
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        now = datetime.datetime.now()
        now = now.strftime("%H_%M_%S")
        filename = secure_filename(file.filename)
        name = filename.rsplit('.')
        if model_id == "dx":
            filename = name[0] + "_DX" + now + "." + name[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('File saved succesfully')
            
        if model_id == "sx":
            filename = name[0] + "_SX" + now + "." + name[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('File saved succesfully')
            
 #-------------------------------------------------PILING RESULTS IN JSON FORMAT TO SEND BACK TO ROBOT-----------------------------------------
    # Classification
    if m_method == 'CLS':
        TRIMessage("No Classification method implemented, yet")
        result = jsonresponse(0,0,0,0,0,0,"fail",None)
        
    # Detection
    elif m_method == 'DET':
        
        result = jsonresponse(0,0,0,0,"photo saved",0,"success","Image")
            
    # no method
    else:
        TRIMessage("no HTTP method")
        result = jsonresponse(0,0,0,0,0,0,"fail",None)
        with open('json.txt', 'a') as f:
            f.write('\n')
            f.write((str(result)))
            f.close()
    
    return result 
    

# ============================== ENTRY POINT =========================================================================================================================================================================================================================
if __name__ == '__main__':
    check=False
    try:
        host_addr = ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) 
        #host_addr = "10.10.10.180"
        check = True if len(host_addr) > 0 else False
    except Exception as e:
        TRIMessage(e)
    if check == True:
        host_addr = host_addr[-1] if len(host_addr) > 1 else host_addr[0]
        TRIMessage(f'serving on http://{host_addr}:{HOST_PORT}')
    else:
        TRIMessage(f'serving on http://127.0.0.1:{HOST_PORT}')
    serve(app, port=HOST_PORT, ident=HOST_NAME, _quiet=True)