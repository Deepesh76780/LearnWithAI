from flask import Flask, make_response, render_template, Response
import cv2
import side_face_detection as sfd
# import pose_detection as ps
import Drowsyness as dwsy
from PIL import Image as im
from flask import Markup
import os
import mediapipe as mp
import rebs_estimation as rebe
import aug
import iris_detection as ird
import brightness_control as bc
import air_canvas as acc
from collections import deque
import numpy as np


app = Flask(__name__)
camera = cv2.VideoCapture(0)

picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
iris_list = [face_mesh]


def side_face_iris():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = sfd.side_face_detector(frame)
            frame1 = cv2.flip(frame1, 1)
            frame2 = ird.iris_estimation(frame1, iris_list)
            # data = im.fromarray(frame)

            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

drowsy_list = [sleep, drowsy, active, status, color]


def drowsy_face():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = dwsy.drowsyness_detector(frame, drowsy_list)
            # data = im.fromarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


sleepbc = 0
drowsybc = 0
activebc = 0
statusbc = ""
colorbc = (0, 0, 0)

brightness_list = [sleepbc, drowsybc, activebc, statusbc, colorbc]


def brightness_control():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = bc.brightness_contoller(frame, brightness_list)
            # data = im.fromarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


# def pose():
#     while True:

#         # read the camera frame
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             frame1 = ps.pose_estimation(frame)
#             # data = im.fromarray(frame)
#             ret, buffer = cv2.imencode('.jpg', frame1)
#             frame2 = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_list = [pose, counter, stage]


def pose_rebs():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = rebe.pose_estimation(frame, pose_list)
            # data = im.fromarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)
augment_list2 = [MIN_MATCHES, detector,
                 FLANN_INDEX_KDTREE, index_params, search_params, flann]

input_image = cv2.imread('LearnWithAI\\static\\pics\\vk.jpg')
augment_image = cv2.imread('LearnWithAI\\static\\pics\\mask.jpg')

input_image = cv2.resize(input_image, (300, 400), interpolation=cv2.INTER_AREA)
augment_image = cv2.resize(augment_image, (300, 400))
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# find the keypoints with ORB
keypoints, descriptors = detector.detectAndCompute(gray_image, None)
augment_list = [gray_image, augment_image, keypoints, descriptors]


def aug_mask():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = aug.augment_detector(frame, augment_list, augment_list2)
            # data = im.fromarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
kernel = np.ones((5, 5), np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

air_list = [bpoints, gpoints, rpoints, ypoints, blue_index,
            green_index, red_index, yellow_index, kernel, colors, colorIndex]


def air_canva():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            frame1 = acc.colour_detector(frame, air_list)
            # data = im.fromarray(frame)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame2 = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


@app.route('/')
def index():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'head.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'head2.svg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    return render_template('index.html', head=img, footer=img2, symbol=img3, head2=img1)


@app.route('/AIbox')
def AIbox():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'head.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'head2.svg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    return render_template('AIbox.html', head=img, footer=img2, symbol=img3, head2=img1)


@app.route('/Quizes')
def Quizes():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'head.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'head2.svg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    return render_template('Quizes.html', head=img, footer=img2, symbol=img3, head2=img1)


@app.route('/quiz')
def quiz():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'head.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'head2.svg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    return render_template('quizPage.html', head=img, footer=img2, symbol=img3, head2=img1)


@app.route('/Practice')
def Practice():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'head.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'head2.svg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    return render_template('Practice.html', head=img, footer=img2, symbol=img3, head2=img1)


@app.route('/Courses')
def Courses():
    img = os.path.join(app.config['UPLOAD_FOLDER'], 'c1.webp')
    img0 = os.path.join(app.config['UPLOAD_FOLDER'], 'symbol.svg')
    img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'c2.jpg')
    img2 = os.path.join(app.config['UPLOAD_FOLDER'], 'c3.jpg')
    img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'c4.png')
    img4 = os.path.join(app.config['UPLOAD_FOLDER'], 'c5.png')
    img5 = os.path.join(app.config['UPLOAD_FOLDER'], 'c6.png')
    imgf = os.path.join(app.config['UPLOAD_FOLDER'], 'footer.svg')

    return render_template('Courses.html', footer=imgf, symbol=img0, c1=img, c2=img2, c3=img3, c4=img1, c5=img4, c6=img5)


@app.route('/interview')
def interview():
    return render_template('interview.html')


@app.route('/nightstudy')
def nightstudy():
    return render_template('nightstudy.html')


@app.route('/videoforInterview')
def videoforInterview():
    frame = Response(
        side_face_iris(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


@app.route('/videoforNightstudy')
def videoforNightstudy():
    frame = Response(
        drowsy_face(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


# @app.route('/videoforPose')
# def videoforPose():
#     frame = Response(
#         pose(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     return frame


@app.route('/videoforPoseRebs')
def videoforPoseRebs():
    frame = Response(
        pose_rebs(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


@app.route('/videoforAug')
def videoforAug():
    frame = Response(
        aug_mask(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


@app.route('/videoforBrightness')
def videoforBrightness():
    frame = Response(
        brightness_control(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


@app.route('/videoforAircanvas')
def videoforAircanvas():
    frame = Response(
        air_canva(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


if __name__ == "__main__":
    app.run(debug=True)
