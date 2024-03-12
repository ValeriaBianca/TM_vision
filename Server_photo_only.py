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
            

    # -------------------------------------------IMAGE PROCESSING--------------------------------------------------------------------
    #-- Undistort and rectify images before giving them to the ORB algorithm to process
    #-- Use camera parameters PREVIOUSLY evaluated by stereo calibration script
    #cv_file = cv2.FileStorage()
    #cv_file.open('C:\\Users\\bianc\\TMvision_TmHttp_server_sample_code\\python_example\\stereoMap.xml', cv2.FILE_STORAGE_READ)

    #stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    #stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    #stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    #stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    #cv_file.release()
    #-- The algorithm needs to be trained to track the desired object so I place 
    # the training image in the pc and let the algorithm pick it
    trainim = cv2.imread(r"C:\Users\bianc\TMvision_TmHttp_server_sample_code\python_example\images\train_im.jpg")
    #trainim = cv2.imread(r"C:\Users\bianc\images\chess.jpg")
    
    #-- The query image is picked from the robot camera 
    if model_id == "dx":
        queryim = cv2.imread(UPLOAD_FOLDER + r"\image_DX.jpg")
    #    queryim = UndistAndRect(queryim, stereoMapR_x, stereoMapR_y)
    if model_id == "sx":
        queryim = cv2.imread(UPLOAD_FOLDER + r"\image_SX.jpg")
    #    queryim = UndistAndRect(queryim, stereoMapL_x, stereoMapL_y)
    

    #===========ORB==================================================================================================================
    
    #IMPORTANT_NOTE: avoid irrelevant corners in query pictures at all costs!!
    outputimg = queryim.copy() #i will make a copy of the query image to transform into output

    width, height = Image.open(UPLOAD_FOLDER + r"\image_DX.jpg").size #widthxheight pixels of query img

    # 1 = train, 2 = query

    # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB.
    # So if I use plt then uncomment this conversion
    # trainim = cv2.cvtColor(trainim, cv2.COLOR_BGR2RGB)
    # queryim = cv2.cvtColor(queryim, cv2.COLOR_BGR2RGB)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with ORB
    kp1 = orb.detect(trainim,None)
    kp2 = orb.detect(queryim,None)

    kp1, des1 = orb.compute(trainim, kp1)
    kp2, des2 = orb.compute(queryim, kp2)

    # create BFMatcher object
    bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors, knn method.
    matches = bf.knnMatch(des1,des2,k=2)

    # I can also mask the keypoints by "filtering" only the best 
    # ones; a.k.a. the keypoints whose descriptor have low distance

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x: x[:][1].distance)

    good = []
    for m,n in matches:
        if m.distance < nu*n.distance:
            good.append([m])

    Matched = cv2.drawMatchesKnn(trainim,kp1,queryim,kp2,
                                good,outImg=None,matchColor=(0, 155, 0),singlePointColor=(0, 255, 255),matchesMask=None,flags=0)
    
    kp3 = [] #creating an empty keypoint object
    
    for i in range(len(good)):
        a = good[i][0].trainIdx 
        idx=kp2[a].pt
        key = cv2.KeyPoint(idx[0],idx[1],1)
        kp3.append(key)
       
    
    output_img = cv2.drawKeypoints(outputimg, kp3 ,0,(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
    
    

    #===========Getting coordinates for box=====================================================================================================
    #-- Localize object
    obj = np.empty((len(good),2), dtype = np.float32)
    scene = np.empty((len(good),2), dtype = np.float32)
    output = np.empty((len(good),2), dtype=np.float32)

    for i in range(len(good)):
        obj[i,0] = kp1[good[i][0].queryIdx].pt[0] #coordinata x del keypoint delle train image che corrisponde 
        # all'i esimo keypoint nella query image- sfrutto il match object che mappa le corrispondenze tra i due set di kypoints
        # nell' obj metto quindi le coordinate in pixel della train image con corrispondenza alla query
        # nell'oggetto scene faccio l'opposto: salvo le coodinate in pixel della query image che corrispondono ai keypoint della train image
        obj[i,1] = kp1[good[i][0].queryIdx].pt[1]
        scene[i,0] = kp2[good[i][0].trainIdx].pt[0]
        scene[i,1] = kp2[good[i][0].trainIdx].pt[1]
    # questo kp3 serve semplicemente a creare l'immmagine di output singola da quella dove ho l'immagine di train e di query affiancate
    for i in range(len(kp3)):    
        output[i,0] = kp3[i].pt[0] 
        output[i,1] = kp3[i].pt[1] 

    try:
        H, _ = cv2.findHomography(obj,scene,cv2.RANSAC)
        H2, _ = cv2.findHomography(obj, output, cv2.RANSAC)
    except:
        print("Not enough matches or zero matches found!")
        result = {            
            "message": "Not enough matches or zero matches found!",
            "result": None            
        }
        return result #with this return the server can respond to more post/get requests
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0

    obj_corners[1,0,0] = trainim.shape[1]
    obj_corners[1,0,1] = 0

    obj_corners[2,0,0] = trainim.shape[1]
    obj_corners[2,0,1] = trainim.shape[0]

    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = trainim.shape[0]

    #-- Here I get the corners of the train object "mapped" on to the query image coordinates through the homography matrix
    # previously evaluated
    scene_corners = cv2.perspectiveTransform(obj_corners, H)
    output_corners = cv2.perspectiveTransform(obj_corners, H2) 

    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv2.line(Matched, 
            (int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])),\
            (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),
            (0,255,0), 4)
    cv2.line(Matched, (int(scene_corners[1,0,0] + trainim.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv2.line(Matched, (int(scene_corners[2,0,0] + trainim.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv2.line(Matched, (int(scene_corners[3,0,0] + trainim.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + trainim.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    cv2.line(output_img, 
        (int(output_corners[0,0,0] ), int(output_corners[0,0,1])),\
        (int(output_corners[1,0,0] ), int(output_corners[1,0,1])),
        (0,255,0), 4) 
    # (coordinata x 0,0,0 , coordinata y 0,0,1); le altre due coordinate dopo il backslash sono x,y del punto successivo
    cv2.line(output_img, (int(output_corners[1,0,0] ), int(output_corners[1,0,1])),\
    (int(output_corners[2,0,0] ), int(output_corners[2,0,1])), (0,255,0), 4)
    cv2.line(output_img, (int(output_corners[2,0,0] ), int(output_corners[2,0,1])),\
    (int(output_corners[3,0,0] ), int(output_corners[3,0,1])), (0,255,0), 4)
    cv2.line(output_img, (int(output_corners[3,0,0] ), int(output_corners[3,0,1])),\
    (int(output_corners[0,0,0]), int(output_corners[0,0,1])), (0,255,0), 4)


    #-- Get coordinates, height and width of square box
    cx = (output_corners[0,0,0]+output_corners[1,0,0]+output_corners[2,0,0]+output_corners[3,0,0])/4
    cy = (output_corners[0,0,1]+output_corners[1,0,1]+output_corners[2,0,1]+output_corners[3,0,1])/4
    box_h = np.sqrt(np.square(output_corners[0,0,0]-output_corners[3,0,0])+np.square(output_corners[0,0,1]-output_corners[3,0,1]))
    box_w = np.sqrt(np.square(output_corners[0,0,0]-output_corners[1,0,0])+np.square(output_corners[0,0,1]-output_corners[1,0,1]))

    #-- Get rotation of square box
    # acos((tr(R)-1)/2), Rodrigues formula inverted
    #print((np.trace(H2)-1)/2)
    #theta = np.arccos((np.trace(H2)-1)/2) #answer is in radians
    theta = - np.arctan2(H2[0,1], H2[0,0]) 
    theta = np.rad2deg(theta)
    
    if model_id == 'dx':
        label = 'DX_Image'
    else:
        label = 'SX_Image'

 #-------------------------------------------------PILING RESULTS IN JSON FORMAT TO SEND BACK TO ROBOT-----------------------------------------
    # Classification
    if m_method == 'CLS':
        TRIMessage("No Classification method implemented, yet")
        result = jsonresponse(0,0,0,0,0,0,"fail",None)
        
    # Detection
    elif m_method == 'DET':
        
        result = jsonresponse(float(cx),float(cy),float(box_w),float(box_h),str(label),float(theta),"success","Image")
            
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