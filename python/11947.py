from IPython.display import YouTubeVideo
import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

get_ipython().magic('matplotlib inline')
# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()

def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()
    
class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |                     cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else                     cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord
    
class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print self.video.isOpened()

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (150, 150, 0), 8)

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)


# <img style="margin-top: 50px; margin-right: 30px; float: right; width: 40%" src="images/face_recon_example.png">
# 
# <h1 style="align: center; color: ">Recognize Faces in a Live Video Feed</h1>
# <br>
# 
# We have learned:
# - How to detect faces
# - How to normalize faces images
# - How to train a face recognition model in OpenCV
# - How to recognize from still images
# 
# What is left?
# - Recognize faces in a live video feed
# - Apply threshold to flag unknown faces

# ### Load Images, load labels and train models
# 

images, labels, labels_dic = collect_dataset()

rec_eig = cv2.face.createEigenFaceRecognizer()
rec_eig.train(images, labels)

# needs at least two people 
rec_fisher = cv2.face.createFisherFaceRecognizer()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.createLBPHFaceRecognizer()
rec_lbph.train(images, labels)

print "Models Trained Succesfully"


detector = FaceDetector("xml/frontal_face.xml")
webcam = VideoCamera(0)


cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame, True) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.MinDistancePredictCollector()
            rec_lbph.predict(face, collector)
            conf = collector.getDist()
            pred = collector.getLabel()
            threshold = 140
            print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
            cv2.putText(frame, labels_dic[pred].capitalize(),
                        (faces_coord[i][0], faces_coord[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        clear_output(wait = True)
        draw_rectangle(frame, faces_coord) # rectangle around face
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("PyData Tutorial", frame) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


# ### How is the threshold defined?
# 
# <img style="width: 30%; float: right; margin-right: 80px" src="http://www.programering.com/images/remote/ZnJvbT01MWN0byZ1cmw9Y0djcTVpTXhBVFVQMUNPWEZEUjRKVVFCSmxUeEJqYTJCblE1RVZNTTlXYUxkM0xFWnpMRFJ6THdBVFR2SURNelpXZTM5U2J2Tm1MdlIzWXhVakx6TTNMdm9EYzBSSGE.jpg">
# <br>
# Is the actual distance between the sample image and the closest face in the training set.
# 

# ### Apply threshold
# 

cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame, False) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.MinDistancePredictCollector()
            rec_lbph.predict(face, collector)
            conf = collector.getDist()
            pred = collector.getLabel()
            threshold = 140
            print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
            clear_output(wait = True)
            if conf < threshold: # apply threshold
                cv2.putText(frame, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            else:
                cv2.putText(frame, "Unknown",
                            (faces_coord[i][0], faces_coord[i][1]),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        draw_rectangle(frame, faces_coord) # rectangle around face
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("PyData Tutorial", frame) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


del webcam
webcam = VideoCamera(1)


cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
while True:
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame, False) # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.MinDistancePredictCollector()
            rec_lbph.predict(face, collector)
            conf = collector.getDist()
            pred = collector.getLabel()
            threshold = 140
            print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
            clear_output(wait = True)
            if conf < threshold: # apply threshold
                cv2.putText(frame, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            else:
                cv2.putText(frame, "Unknown",
                            (faces_coord[i][0], faces_coord[i][1]),
                            cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
        draw_rectangle(frame, faces_coord) # rectangle around face
    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("PyData Tutorial", frame) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break


del webcam


def draw_label(image, text, coord, conf, threshold):
    if conf < threshold: # apply threshold 
        cv2.putText(image, text.capitalize(),
                    coord,
                    cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
    else:
        cv2.putText(image, "Unknown",
                    coord,
                    cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)


def live_recognition(index, webcam):
    global double_frame
    detector = FaceDetector("xml/frontal_face.xml")
    while True:
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame, False) # detect more than one face
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord) # norm pipeline
            for i, face in enumerate(faces): # for each detected face
                collector = cv2.face.MinDistancePredictCollector()
                rec_lbph.predict(face, collector)
                conf = collector.getDist()
                pred = collector.getLabel()
                threshold = 140
                draw_label(frame, labels_dic[pred], 
                           (faces_coord[i][0], faces_coord[i][1] - 10), 
                           conf, threshold)
            draw_rectangle(frame, faces_coord) # rectangle around face
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, 
                    cv2.LINE_AA)
        if index == 0:
            cv2.putText(frame, "Laptop", (frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "External", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
        double_frame[0 : 481, index * 640 : (index +1 ) * 640] = frame # copy new frame to FS
        cv2.imshow("PyData Tutorial", double_frame) # live feed in external
        if cv2.waitKey(30) & 0xFF == 27:
            break


from threading import Thread
cv2.namedWindow("PyData Tutorial", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("PyData Tutorial", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.namedWindow("PyData Tutorial 1", cv2.WINDOW_AUTOSIZE)

webcam_0 = VideoCamera(0)
webcam_1 = VideoCamera(1)

single_frame = np.zeros_like(webcam_0.get_frame())
double_frame = np.hstack((single_frame, single_frame))

thread_0 = Thread(target = live_recognition, args = (0, webcam_0))
thread_1 = Thread(target = live_recognition, args = (1, webcam_1))
thread_0.start()
thread_1.start()
thread_1.join()
thread_0.join()
del webcam_0
del webcam_1
cv2.destroyAllWindows()


# <div style="float: left; width: 50%; height: 200px; padding-bottom:40px">
#     <img src="http://pydata.org/amsterdam2016/static/images/pydata-logo-amsterdam-2016.png" alt="PyData Amsterdam 2016 Logo">
# </div>
# <div style="float: right; width: 50%; height: 200px; padding-bottom:40px">
#     <img style="height: 100%; float:right" src="http://pydata.org/amsterdam2016/media/sponsor_files/qualogy_logo_350px.png" alt="Qualogy Logo">
# </div>
# 
# <h1 align="center">Thank you</h1>
# 
# <div style="float: left; width: 50px">
# <img style="width: 50px" src="https://pmcdeadline2.files.wordpress.com/2014/06/twitter-logo.png?w=970">
# </div>
# <br>
# <div style="float: left; margin-left: 10px">
#    @rodagundez 
# </div>
# 

