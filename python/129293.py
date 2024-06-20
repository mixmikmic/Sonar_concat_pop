# # OpenCV Tutorial Sample_01: ocv_info
# 
# [Sample 01](ocv_info.py) is a simple diagnostic program that queries the development environment and ensures that all the prerequisites have been met and displays the version information. It also checks to see if Environment variables have been set and displays the path for diagnostics if necessary.
# 
# >*Note:* If you are using pre-built OpenCV, you need to use Python 2.7.x to run the samples. If you have built OpenCV from source, then you can use either Python 3.x or Python 2.x.
# 
# First the Standard Python Shebang
# 

#!/usr/bin/env python2


# Next import print_function for Python 2/3 comptibility. This allows use of print like a function in Python 2.x
# 

from __future__ import print_function


# OpenCV uses Numpy arrays to manipulate image and video data and is a mandatory requirement. So import numpy module first.
# 

import numpy as np


# Now print the version. In the script this is done at the end in the Program Outputs block
# 

print('Numpy Version:', np.__version__)


# Next import the OpenCV python module
# 

import cv2


# Now print the version. In the script this is done at the end in the Program Outputs block
# 

print('OpenCV Version:', cv2.__version__)


# Import the other required python modules for this script. Matplotlib is used in a number of tutorials found in the OpenCV package and OS and sys are needed to test whether the OpenCV environment variables are properly setup.
# 

import matplotlib as mpl
import os
import sys


# Now print the versions here. In the script this is done at the end in the Program Outputs block
# 

print('Matplotlib Version:', mpl.__version__)
print(sys.version)


# Now we check to see if the OpenCV environment variables have been properly set. We need to do this in a safe way to prevent the script from crashing in case no variable has been set. So use standard python exception handling...
# 

try:
    pyth_path = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    pyth_path = []


# Now print the environment variable. In the script this is done at the end in the Program Outputs block.
# 

print('Python Environment Variable - PYTHONPATH:', pyth_path)


# Next check the OpenCV environemnt variables
# 

try:
    ocv2_path = os.environ['OPENCV_DIR']
except KeyError:
    ocv2_path = []
    
try:
    ocv2_vers = os.environ['OPENCV_VER']
except KeyError:
    ocv2_path = []


# Now print the environment variable. In the script this is done at the end in the Program Outputs block
# 

print('OpenCV Environment Variable - OPENCV_DIR:', ocv2_path)
print('OpenCV Environment Variable - OPENCV_VER:', ocv2_vers)


# Finally check the FFMPEG environment variable
# 

try:
    ffmp_path = os.environ['FFMPEG_BIN']
except KeyError:
    ffmp_path = []


# Now print the environment variable. In the script this is done at the end in the Program Outputs block
# 

print('FFMPEG Environment Variable - FFMPEG_BIN:', ffmp_path)


# If you did not see any errors, you are to be congratulated on setting up your OpenCV environment correctly.
# 

# ### Congratulations!
# 

# # OpenCV Tutorial Sample 10: ocv_face_img
# 
# [Sample 10](ocv_face_img.py) is a basic Face and Eye Detection program that uses OpenCV to analyze an image and detect human faces and eyes. The detected areas or Regions of Interest (ROI) are demarcated with rectangles. The program uses the OpenCV built-in pre-trained Haar feature-based cascade classifiers in order to perform this task.
# 
# ### What are Cascade Classifiers?
# 
# Cascade Classifiers are a form of ensemble learning systems. Such systems use a collection of a large number of simple classifiers in a cascade. This leads to accurate yet computationally efficient detection systems.
# 
# ### What are Haar feature-based Cascade Classifiers?
# 
# Haar features are named after Haar wavelets in mathematics. The are patterns in the pixel values of an image such as edges, lines and neighbors that are used with a windowing technique to extract features from an image. Since the features could be different, a collection of specialized but simple pattern classifiers are used in a cascade to perform the feature detection.
# 
# ### References:
# 
# 1. Rapid Object Detection using a Boosted Cascade of Simple Features [pdf](http://wearables.cc.gatech.edu/paper_of_week/viola01rapid.pdf) 
#  [_This is the original paper by Prof. Viola and Prof. Jones_]
# 2. An Analysis of the Viola-Jones Face Detection Algorithm [pdf](http://www.ipol.im/pub/art/2014/104/article.pdf)
# 3. A review on Face Detection and study of Viola Jones method [pdf](http://www.ijcttjournal.org/2015/Volume25/number-1/IJCTT-V25P110.pdf)
# 4. Explaining AdaBoost [pdf](http://rob.schapire.net/papers/explaining-adaboost.pdf)
# 5. Face detection using Haar Cascades [Tutorial link](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)
# 
# ## Sample Code
# 
# First we do the usual initializations ...
# 

from __future__ import print_function
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import the Numby and OpenCV2 Python modules
import numpy as np
import cv2


# Select the pre-trained Haar Cascade Classifier file to use for face and eye detection respectively and pass it to the OpenCV API [cv2.CascadeClassifier()](http://docs.opencv.org/3.0-last-rst/modules/objdetect/doc/cascade_classification.html#cv2.CascadeClassifier)
# 

# This section selects the Haar Cascade Classifer File to use
# Ensure that the path to the xml files are correct
# In this example, the files have been copied to the local folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# Next load an image to analyze. Several examples are provided. Make sure that only one cv2.imread() command is active and all the rest are commented out. The example images have all been copied to the local folder.
# 

img = cv2.imread('brian-krzanich_2.jpg')
#img = cv2.imread('Intel_Board_of_Directors.jpg')
#img = cv2.imread('bmw-group-intel-mobileye-3.jpg')


# Now convert the image to Grayscale to make it easier to process
# 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# The [detectMultiScale](http://docs.opencv.org/3.0-last-rst/modules/objdetect/doc/cascade_classification.html#cv2.CascadeClassifier.detectMultiScale) method of the OpenCV Cascade Classifier API detects features of different sizes in the input image. The detected objects are returned as a list of rectangles.
# 
# cv2.CascadeClassifier.detectMultiScale(image[,scaleFactor[,minNeighbors[,flags[,minSize[,maxSize]]]]]) -> objects
# 

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


# Draw the rectangles around detected Regions of Interest [ROI], namely faces amd eyes using [cv2.rectangle()](http://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/drawing_functions.html#cv2.rectangle) for all detected objects in the image returned by the classifiers.
# 
# cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
# 
# >Note: Since the eyes are a part of the face, we nest the classifier for the eyes. So we only look for eyes in areas identified as the face. This improves the accuracy.
# 

# Draw the rectangles around detected Regions of Interest [ROI] - faces
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # Since eyes are a part of face, limit eye detection to face regions to improve accuracy
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # Draw the rectangles around detected Regions of Interest [ROI] - eyes
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# Finally display the result until dismissed and release all reseources used.
# 

# Display the result        
cv2.imshow('img',img)
# Show image until dismissed using GUI exit window
cv2.waitKey(0)
# Release all resources used
cv2.destroyAllWindows()


# # OpenCV Tutorial Sample_05: ocv_ocl_info
# 
# [Sample 05](ocv_ocl_info.py) is a simple diagnostic program that determines whether OpenCL is available for use within OpenCV, Enables OpenCL, checks whether it has been enabled and then disables it. The program then checks if OpenCL has been disabled and exits.
# 
# ### What is OpenCL?
# OpenCL™ (Open Computing Language) is the open standard for parallel programming. Using OpenCL, one can use the GPU for parallel computing tasks other than just for graphics programming. Once can also use DSP's, FPGA's and other types of processors using OpenCL.
# 
# ###  How does OpenCV use OpenCL?
# In Computer Vision many algorithms can run on a GPU much more effectively than on a CPU: e.g. image processing, matrix arithmetic, computational photography, object detection etc. OpenCV 3.x is able to accelerate and optimize performaance by using an architectural concept called Transparent API (T-API) to transparently speed up certain tasks if supported by the underlying hardware.
# 
# ## Sample Diagnostic Code
# 
# Start with the usual initialization
# 

#!/usr/bin/env python2

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import the OpenCV2 module
import cv2


# Using the OpenCV API cv2.ocl.haveOpenCL() returns True if OpenCL is supported. If it is supported, OpenCL can be enabled using cv2.ocl.setUseOpenCL(True) and disabled using cv2.ocl.setUseOpenCL(False). To check if OpenCL has been enabled or disabled, use cv2.ocl.useOpenCL() which will return True or False as the case may be.
# 
# >Note: OpenCV Python module does not currently support enumeration of OpenCL devices.
# 
# The enable OpenCL with exception handling and check whether it was enabled, run the following code:
# 

try:
    # Returns True if OpenCL is present
    ocl = cv2.ocl.haveOpenCL()
    # Prints whether OpenCL is present
    print("OpenCL Supported?: ", end='')
    print(ocl)
    print()
    # Enables use of OpenCL by OpenCV if present
    if ocl == True:
        print('Now enabling OpenCL support')
        cv2.ocl.setUseOpenCL(True)
        print("Has OpenCL been Enabled?: ", end='')
        print(cv2.ocl.useOpenCL())

except cv2.error as e:
    print('Error:')


# The disable OpenCL with exception handling and check whether it has been disabled, run the following code:
# 

try:
    ocl_en = cv2.ocl.useOpenCL()
    if ocl_en ==True:
        print('Now disabling OpenCL support')
        cv2.ocl.setUseOpenCL(False)

    print("Checking - Is OpenCL still Enabled?: ", end='')
    print(cv2.ocl.useOpenCL())
    print()

except cv2.error as e:
    print('Error:')


# # OpenCV Tutorial Sample 8: ocv_dog_img
# 
# [Sample 08](sample_08/ocv_dog_img.py) is a program that overlays a Digital On-Screen Graphic (DOG) or logo onto a still image. DOG is a form of digital watermarking routinely used on broadcast TV to show the TV channel logo. It can also be used on digital signage to watermark content. 
# 
# In previous samples, we have seen how to overlay text on images and video. This sample shows how to overlay and image on another image.
# 
# The logo image (DOG) is usually a PNG file that is capable of preserving transparency information, in other words, the alpha channel.
# 
# In the interactive tutorial, we will use matplotlib to display some of the intermediate results.
# 
# First we start off with the usual initializations...
# 

#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and other needed Python modules
import numpy as np
import cv2


# Next load the image to be watermarked. We will call this the source image. For illustrative purposes, we will display this image in a named window called "Source Image". Remember the window will remained grayed out until the event handler cv2.waitkey() is called.
# 

# Load the source image
img = cv2.imread('Intel_Wall.jpg')
# Create a named window to show the source image
cv2.namedWindow('Source Image', cv2.WINDOW_NORMAL)
# Display the source image
cv2.imshow('Source Image',img)


# Next load the logo image with which the source image will be watermarked. A second named window called "Result Image" will help serve as a placeholder to handle intermediate outputs, resizing and the final image.
# 

# Load the logo image
dog = cv2.imread('Intel_Logo.png')
# Create a named window to handle intermediate outputs and resizing
cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)


# The Logo image and source image are not te same size. So we need to first find the size of the logo. We do this using the numpy shape object.
# 

# To put logo on top-left corner, create a Region of Interest (ROI)
rows,cols,channels = dog.shape
roi = img[0:rows, 0:cols ]
# Print out the dimensions of the logo...
print(dog.shape)


# Now convert the logo image to grayscale for faster processing... 
# Only in the interactive tutorial, we will use matplotlib to display the result.
# 

# Convert the logo to grayscale
dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
# The code below in this cell is only to display the intermediate result and not in the script
from matplotlib import pyplot as plt
plt.imshow(dog_gray)
plt.show()


# Next create a mask and inverse mask of the logo image ...
# 

# Create a mask of the logo and its inverse mask
ret, mask = cv2.threshold(dog_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# The code below in this cell is only to display the intermediate result and not in the script
plt.imshow(mask_inv)
plt.show()


# Now we blackout the logo within the ROI so that we can extract it from its background.
# 

# Now blackout the area of logo in ROI
img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)


# Perform the extraction
# 

# Now just extract the logo
dog_fg = cv2.bitwise_and(dog,dog,mask = mask)
# The code below in this cell is only to display the intermediate result and not in the script
plt.imshow(dog_fg)
plt.show()


# Now we add the logo to the source image. We can use the OpenCV [cv2.add()](http://docs.opencv.org/3.0-last-rst/modules/core/doc/operations_on_arrays.html#cv2.add) function.
# 

# Next add the logo to the source image
dst = cv2.add(img_bg,dog_fg)
img[0:rows, 0:cols ] = dst


# Time to display the result
# 

# Display the Result
cv2.imshow('Result Image',img)
# Wait until windows are dismissed
cv2.waitKey(0)


# Now release all resources used
# 

# Release all resources used
cv2.destroyAllWindows()


# In this example, we used some OpenCV image processing API's to extract the logo from its background. Using the alpha channel or transaprency of the PNG can also be exploited to produce the same effect. You can also reduce the opacity of the logo itself.
# 

# # OpenCV Tutorial Sample 4: ocv_video_test
# 
# [Sample 04](ocv_video_test.py) is a sanity test that uses OpenCV to connect to a WebCam and display the video stream. This test serves to ensure that OpenCV WebCam installation is working and further validates the development environment. It also shows how to overlay text on video streams.
# 
# We start by performing the basic initializations
# 

#!/usr/bin/env python2

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and Numpy modules
import numpy as np
import cv2


# First we need to initialize a Video Web Camera for capturing video with OpenCV. We do this transparently by using an OpenCV API [cv2.VideoCapture()](http://docs.opencv.org/3.0-last-rst/modules/videoio/doc/reading_and_writing_video.html#cv2.VideoCapture)
# 
# ```
#     cv2.VideoCapture(Parameters)
# ```
# 
#     Parameters:
# 
#     filename – Name and path of file to be loaded.
#     device_id - Id of the opened video capturing device (i.e. a camera index).
# 
#     Device Id:
#     
#     The default camera is 0 (usually built-in).The second camera would be 1 and so on
# 
# >Note: On the Nuc which has no camera, the default Id of "0" should work. On a Laptop, you may need to try "0" or "1" if you have two cameras for front and back.
# 

webcam = cv2.VideoCapture(0)


# cv2.videoCapture() method has many calls and isOpened() returns (True) if the device is opened sucessfully
# 

# Check if Camera was initialized correctly
success = webcam.isOpened()
if success == False:
    print('Error: Camera could not be opened')
else:
    print('Success: Grabbed the camera')


# Next we use the read() function from cv2.VideoCapure to read a video frame while this is (True)
# 

# Read each frame in video stream
ret, frame = webcam.read()        


# Once the frame is read, it is usually converted to grayscale before performing further operations. This avoids having to process color information in real-time. For this we use the same cv2.cvtColor() method from our previous example with just a different color space conversion code.
# 

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Overlay Text on the video frame with Exit instructions
# 

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gray, "Type q to Quit:",(50,50), font, 1,(0,0,0),2,cv2.LINE_AA)


# Now display the captured frame with overlay text in a GUI window
# 

cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Output',gray)


# Next comes the event handler where we wait for the q key and then release the devices and resources used
# 

# Wait for exit key "q" to quit
if cv2.waitKey(1) & 0xFF == ord('q'):
    print('Exiting ...')


# >Note: Since the interactive tutorial mode is not well suited for handling video, the While(True) loop has been omited and so you will only see a still image. But you can see this working for video in the consolidated example and script.
# 
# Next we release the devices and all resources used.
# 

webcam.release()
cv2.destroyAllWindows()


# >Note: Ensure that the camera was released in the previous step. The camera light should go off. If not restart the kernel before continuing to the next step.
# 
# Now putting it all together with exception handling:
# 

#!/usr/bin/env python2

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and Numpy modules
import numpy as np
import cv2
try:
    webcam = cv2.VideoCapture(0)
    # Check if Camera initialized correctly
    success = webcam.isOpened()
    if success == False:
        print('Error: Camera could not be opened')
    else:
        print('Success: Grabbed the camera')


    while(True):
        # Read each frame in video stream
        ret, frame = webcam.read()
        # Perform operations on the frame here
        # For example convert to Grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Overlay Text on the video frame with Exit instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray, "Type q to Quit:",(50,50), font, 1,(0,0,0),2,cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        # Wait for exit key "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exiting ...')
            break
    # Release all resources used
    webcam.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print('Please correct OpenCV Error')


# # OpenCV Tutorial Sample 2: ocv_build_info
# 

# [Sample 02](sample_01/ocv_build_info.py) is a simple diagnostic program that displays the detailed OpenCV build information.
# 

# First the standard Python shebang 
# 

#!/usr/bin/env python2


# Next import print_function for Python 2/3 compatibility. This allows use of print like a function in Python 2.x
# 

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x


# Import the OpenCV Python module
# 

# Import OpenCV Python module
import cv2


# Now obtain and print OpenCV Build Configuration. The function getBuildInformation() returns the full configuration time cmake output
# 

# This function returns the full configuration time cmake output
try:
	buildinfo = cv2.getBuildInformation()
	print(buildinfo)

except cv2.error as e:
	print('Error:')


# # OpenCV Tutorial Sample 9: ocv_dog_vid
# 
# [Sample 09](ocv_dog_vid.py) is a program that overlays a Digital On-Screen Graphic (DOG) on the video display stream. This program uses the same principles as used for the previous example for still images.
# 
# In fact, you can mash sample_04 and sample_08 together to create this sample. It's so simple! The procedure to load and process the image and to extract it from the background is only done once outside of the while loop. This is so you don't slow down the frame rate of the video. 
# 
# Inside the while loop, all that is done is blacking out the logo area and adding the logo to each frame. Replacing the camera device id with a filename and path in cv2.VideoCapture() function allows you to watermark any video file from disk. You can write the resulting video back to disk with the watermark added using the write() method from cv2.VideoCapture().
# 
# Since it is so simple we will just run the program:
# 

#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and Numpy modules
import numpy as np
import cv2

 
try:
    # Create a named window to display video output
    cv2.namedWindow('Watermark', cv2.WINDOW_NORMAL)
    # Load logo image
    dog = cv2.imread('Intel_Logo.png')
    # 
    rows,cols,channels = dog.shape
    # Convert the logo to grayscale
    dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
    # Create a mask of the logo and its inverse mask
    ret, mask = cv2.threshold(dog_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now just extract the logo
    dog_fg = cv2.bitwise_and(dog,dog,mask = mask)
    # Initialize Default Video Web Camera for capture.
    webcam = cv2.VideoCapture(0)
    # Check if Camera initialized correctly
    success = webcam.isOpened()
    if success == False:
        print('Error: Camera could not be opened')
    else:
        print('Sucess: Grabbing the camera')
        webcam.set(cv2.CAP_PROP_FPS,30);
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024);
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768);

    while(True):
        # Read each frame in video stream
        ret, frame = webcam.read()
        # Perform operations on the video frames here
        # To put logo on top-left corner, create a Region of Interest (ROI)
        roi = frame[0:rows, 0:cols ] 
        # Now blackout the area of logo in ROI
        frm_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # Next add the logo to each video frame
        dst = cv2.add(frm_bg,dog_fg)
        frame[0:rows, 0:cols ] = dst
        # Overlay Text on the video frame with Exit instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Type q to Quit:",(50,700), font, 1,(255,255,255),2,cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Watermark',frame)
        # Wait for exit key "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting ...')
            break

    # Release all resources used
    webcam.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print('Please correct OpenCV Error')


# # OpenCV Tutorial Sample 11: ocv_face_vid
# 
# [Sample 11](sample_11/ocv_face_vid.py) is a basic Face and Eye Detection program that uses OpenCV to analyze real-time video and detect human faces and eyes. The detected areas or Regions of Interest (ROI) are demarcated with rectangles. The program uses the OpenCV built-in pre-trained Haar feature-based cascade classifiers in order to perform this task.
# 
# This sample uses the same basic procedures from the previous samples to detect faces and eyes in real-time video. The detection is performed on every video frame.
# 
# The OS and SYS modules are also loaded in this sample in order to automatically locate the OpenCV libraries and use the Haar Cascade Classifier files.
# 
# Here is the initialization code.
# 

#!/usr/bin/env python2

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import Python modules
import numpy as np
import cv2
import sys
import os
print('Face and Eyes Tracker for Real-Time Video')
print('Type Esc to Exit Program ...')
try:
    # Checks to see if OpenCV can be found
    ocv = os.getenv("OPENCV_DIR")
    print(ocv)
except KeyError:
    print('Cannot find OpenCV')
# This automatically locates the cascade files within OpenCV
pri_cascade_file = os.path.join(ocv,'build\etc\haarcascades\haarcascade_frontalface_default.xml')
sec_cascade_file = os.path.join(ocv,'build\etc\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

# Uncomment for Debug if needed
#print(pri_cascade_file)
#print(sec_cascade_file)


# Setup the classifiers to use. We are still using the pre-trained classifiers provided as part of OpenCV.
# 

face_cascade = cv2.CascadeClassifier(pri_cascade_file)
eye_cascade = cv2.CascadeClassifier(sec_cascade_file)


# Now we grab the webcam and configure it
# 

# Initialize Default Camera
webcam = cv2.VideoCapture(0)
# Check if Camera initialized correctly
success = webcam.isOpened()
if success == True:
    print('Grabbing Camera ..')
        # Uncomment and adjust according to your webcam capabilities
        #webcam.set(cv2.CAP_PROP_FPS,30);
        #webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024);
        #webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768);
elif success == False:
    print('Error: Camera could not be opened')


# This section is a mashup of the video camera test sample_04 and the previous sample_10 for face and eye detection on a still image. Only difference is that it is done on each video frame within the while loop.
# 
# Video is converted to grayscale and histogram equalization filter is applied to improve the contrast. This helps the Haar Cascade Classifiers. Everything else stays the same.
# 

while(True):
    # Read each frame in video stream
    ret, frame = webcam.read()
    # Perform operations on the frame here
    # First convert to Grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Next run filters
    gray = cv2.equalizeHist(gray)
    # Uncomment for Debug if needed
    #cv2.imshow('Grayscale', gray)
    # Face detection using Haar Cascades
    # Detects objects of different sizes in the input image which are returned as a list of rectangles.
    # cv2.CascadeClassifier.detectMultiScale(image[,scaleFactor[,minNeighbors[,flags[,minSize[,maxSize]]]]])
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw the rectangles around detected Regions of Interest [ROI] - faces
    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    out = frame.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = out[y:y+h, x:x+w]
        # Since eyes are a part of face, limit eye detection to face regions to improve accuracy
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # Draw the rectangles around detected Regions of Interest [ROI] - eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Facetracker', out)
    # Wait for Esc Key to quit
    if cv2.waitKey(5) == 27:
        print('Quitting ...')
        break
# Release all resources used
webcam.release()
cv2.destroyAllWindows()





# # OpenCV Tutorial Sample 12: ocv_face_cnt_vid
# 
# [Sample 12](sample_12/ocv_face_cnt_vid.py) is a basic People Counter using the previous Face and Eye Detection program that uses OpenCV to analyze real-time video and detect human faces and eyes. In addition to detecting Faces and Eyes, the program also returns the number of faces detected to the console.
# 
# This program counts the number of faces seen in a frame and sends the output to console. It does not perform  a cumulative count. This because the detection is done on every frame of the video and unless faces were recognized and ignored in the following frame, each face would be counted multiple times per frame and produce erroneous results.
# 
# As such a people counter that counts faces needs to also have face recognition capabilities to be robust and perform a cumulative count.
# 
# In the context of digital signage, this example can be used to detect whether a sign is being seen by a single individual on more than one individual.
# 
# This sample is identical to the previous sample with the following exception:
# It uses the numpy len() function to count the number of elements in the array of rectangles for the faces detected after the cascade classifier is run.
# 
# ```
# print('Number of faces detected: ' + str(len(faces)))
# ```
# 

#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import Python modules
import numpy as np
import cv2
import sys
import os
print('Face and Eyes Tracker for Real-Time Video')
print('Type Esc to Exit Program ...')
try:
    # Checks to see if OpenCV can be found
    ocv = os.getenv("OPENCV_DIR")
    print(ocv)
except KeyError:
    print('Cannot find OpenCV')
# This automatically locates the cascade files within OpenCV
pri_cascade_file = os.path.join(ocv,'build\etc\haarcascades\haarcascade_frontalface_default.xml')
sec_cascade_file = os.path.join(ocv,'build\etc\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

# Uncomment for Debug if needed
#print(pri_cascade_file)
#print(sec_cascade_file)

# Setup Classifiers
face_cascade = cv2.CascadeClassifier(pri_cascade_file)
eye_cascade = cv2.CascadeClassifier(sec_cascade_file)

try:
    # Initialize Default Camera
    webcam = cv2.VideoCapture(0)
    # Check if Camera initialized correctly
    success = webcam.isOpened()
    if success == True:
        print('Grabbing Camera ..')
        # Uncomment and adjust according to your webcam capabilities
        #webcam.set(cv2.CAP_PROP_FPS,30);
        #webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024);
        #webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768);
    elif success == False:
        print('Error: Camera could not be opened')

    while(True):
        # Read each frame in video stream
        ret, frame = webcam.read()
        # Perform operations on the frame here
        # First convert to Grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Next run filters
        gray = cv2.equalizeHist(gray)
        # Uncomment for Debug if needed
        #cv2.imshow('Grayscale', gray)
        # Face detection using Haar Cascades
        # Detects objects of different sizes in the input image which are returned as a list of rectangles.
        # cv2.CascadeClassifier.detectMultiScale(image[,scaleFactor[,minNeighbors[,flags[,minSize[,maxSize]]]]])
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # Print to console the number of faces detected
        print('Number of faces detected: ' + str(len(faces)))
        # Draw the rectangles around detected Regions of Interest [ROI] - faces
        # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        out = frame.copy()
        for (x,y,w,h) in faces:		
            cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = out[y:y+h, x:x+w]
            # Since eyes are a part of face, limit eye detection to face regions to improve accuracy
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                # Draw the rectangles around detected Regions of Interest [ROI] - faces
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('Facetracker', out)
        # Wait for Esc Key to quit
        if cv2.waitKey(5) == 27:
            break
    # Release all resources used
    webcam.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print('Please correct OpenCV Error')


# # OpenCV Tutorial Sample 6: ocv_vid_cap
# 
# [Sample 06](ocv_vid_cap.py) is a simple program that uses OpenCV to connect to a WebCam in order to capture and save an image. This example is the basic first step for most video analytics programs. The video output of the WebCam is displayed and when the user inputs a keystroke, the frame is captured and written to an image file.
# 
# Perform the usual initialization
# 

#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import Numpy and OpenCV modules
import numpy as np
import cv2
# Print Debug Info
print('OpenCV Version:', cv2.__version__)
print('Numpy Version:', np.__version__)
print('OpenCV Video Capture Sample')
print('Type c to Capture and q to Quit')


# Next, open a named GUI window for displaying the webcam video in real-time. Initialize a counter to keep track of captures and initialize the webcam. These are the same steps taken in sample_04 to test the camera.
# 

# Initialize GUI window to grab keystrokes when it has focus.
cv2.namedWindow("Capture")
# Initialize Capture Counter
cap_cnt = 0
# Initialize Video Web Camera for capture. The default camera is 0 (usually built-in) 
# The second camera would be 1 and so on
webcam = cv2.VideoCapture(0)
# Check if Camera initialized correctly
success = webcam.isOpened()
if success == False:
    print('Error: Camera could not be opened')
else:
    print('Success: Grabbed the camera')


# Next we setup a loop that reads each frame and then displays it. We also setup an event handler that monitors the keyboard for the c and q keys to capture a framegrab or quit the program respectively. If the c key is pressed, we use the OpenCV API [cv2.imwrite()](http://docs.opencv.org/3.0-last-rst/modules/imgcodecs/doc/reading_and_writing_images.html#cv2.imwrite) to write the frame as an image file to disk. The filename is incremented with the counter we initialized before.
# 

while True:
    # Read each frame in video stream
    ret, frame = webcam.read()
    # Display each frame in video stream
    cv2.imshow("Capture", frame)
    if not ret:
        break
# Monitor keystrokes
    k = cv2.waitKey(1)

    if k & 0xFF == ord('q'):
        # q key pressed so quit
        print("Quitting...")
        break
    elif k & 0xFF == ord('c'):
        # c key pressed so capture frame to image file
        cap_name = "capture_{}.png".format(cap_cnt)
        cv2.imwrite(cap_name, frame)
        print("Saving {}!".format(cap_name))
        # Increment Capture Counter for next frame to capture
        cap_cnt += 1


# Now release all devices and resources used before exiting.
# 

# Release all resources used
webcam.release()
cv2.destroyAllWindows()


