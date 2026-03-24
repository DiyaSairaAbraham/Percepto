# Percepto
#Mini Project done in sxth Semester of B.Tech in Robotics and Automation
# First we use google Colab to train the CNN model using python3, YOLOV8 and Tesla T4 GPU
#install ultralytics , import YOLO and check for its version using orint command

!pip install ultralytics
from ultralytics import YOLO
print(YOLO)  

#prepare the dataset from Roboflow by following the steps : Sign in to Roboflow -> Search for drone, aeroplane, birds and other aerial objects' images for object detection in the universe section -> Choose desired dataset and click on fork dataset -> Crete new version -> download the version in zip  format 
# the downloaded zip file should be extracted and arranged in a folder named [dataset name] in the given order :
#dataset --->images, labels ->test,train,val that is dataset must contain two folders namely images and labels and each of these folders must contain test,train and val folders for corresponding images and labels .Test folder is optional but train and val is required.
#Zip your dataset folder

# Upload our datset folder to Colab as a zip file and unzip the same

from google.colab import files
files.upload()
!unzip A1dataset.zip -d /content/dataset

# check if the datsets are there by choosing the first few files

!ls /content/dataset/A1dataset/images/train |head
!ls /content/dataset/A1dataset/labels/train |head

# create the yaml file

with open('/content/data.yaml', 'w') as f:
    f.write("""
train: /content/dataset/A1dataset/images/train
val: /content/dataset/A1dataset/images/val
nc: 3
names: ['drone','aeroplane','bird',]
""")

#check if the yaml file has all the necessary classes

!cat data.yaml

# mount the data.yml file and the subsequent training to google drive so as to not lose weights and save progress of training in case of any disconnection to GPU

from google.colab import drive
drive.mount('/content/drive')

#train the dataset using YOLOv8

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="/content/data.yaml",
    epochs=50, #increase for more accuracy
    imgsz=640, #use your preferred size
    batch=16,
    device=0,
    project="/content/drive/MyDrive/percepto_runs"
)

#go to drive and download the best.pt under the training to your device 
# Open VS Code, open a folder [say ' percepto'] with our best.pt file source it to python environment and install the libraries as follows:

python -m venv .venv #sourcing python environment in terminal
.venv/Scripts/activate.ps1 # activating the environment
pip install ultalytics #for accessing YOLO models
pip install pyserial # providing backend for python
pip install lapx # for tracking objects in real time
pip install opencv -python # for computer vision applications like accessing webcam, frame processing,etc..

#create a python file [ say 'aerial_tracking.py'] inside percepto folder for tracking , detection and classification of aerial objects using the codes :
'''
import cv2
from ultralytics import YOLO
import serial
import time

#  Load trained YOLO model

model = YOLO("best.pt")  # replace with your trained model path

#  Connect to Arduino

arduino = serial.Serial('COM3', 9600, timeout=1)  # change COM port as per Arduino IDE
time.sleep(2)  # wait for Arduino to initialize

#  Pan-Tilt parameters

tilt_min = 80
tilt_max = 130
pan_min = 0
pan_max = 180
alpha = 0.3  # smoothing factor
pan_angle = 90
tilt_angle = 90
pan_angle = 180 - pan_angle
tilt_angle = 180 - tilt_angle


pan_offset = 0    # adjust after calibration
tilt_offset = 0   # adjust after calibration

prev_pan = 0
prev_tilt = (tilt_min + tilt_max) // 2

last_send = 0
send_interval = 0.05  # 50ms

#  Video source

# webcam
cap = cv2.VideoCapture(0)

# phone webcam (comment demo video above to use)# install IP webcam on phone, open the application, connect the phone to system via USB, click start server option and paste the "http :.........." link 

#cap = cv2.VideoCapture("http://IP address /video") 

# set full frame resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#  Helper functions
def map_angle(value, src_min, src_max, dst_min, dst_max):
    """Map value from one range to another."""
    return int(dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min))

def send_servo(pan, tilt):
    """Send pan-tilt angles to Arduino."""
    command = f"{pan},{tilt}\n"
    arduino.write(command.encode())
    print(f"Sent → Pan: {pan}, Tilt: {tilt}")
#  Main loop
while True:                                                      
    ret, frame = cap.read()
    if not ret:
        print("Frame not received, exiting...")
        break

    frame_h, frame_w = frame.shape[:2]

    # YOLO detection
    results = model(frame, conf=0.4)  # adjust confidence if needed
    annotated_frame = results[0].plot()

    # check if any object detected
    if results[0].boxes.xyxy.shape[0] > 0:
        # take first/highest confidence object
        box = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # map to pan-tilt angles
        pan_angle = map_angle(cx, 0, frame_w, pan_max, pan_min)  # invert left-right
        tilt_angle = map_angle(cy, 0, frame_h, tilt_max, tilt_min)  # invert Y if needed

        # apply offsets
        pan_angle += pan_offset
        tilt_angle += tilt_offset

        # smooth movement
        pan_angle = int(prev_pan * (1 - alpha) + pan_angle * alpha)
        tilt_angle = int(prev_tilt * (1 - alpha) + tilt_angle * alpha)
        prev_pan, prev_tilt = pan_angle, tilt_angle

        # clip to safe ranges
        pan_angle = max(pan_min, min(pan_max, pan_angle))
        tilt_angle = max(tilt_min, min(tilt_max, tilt_angle))

        # send to Arduino (throttled)
        current_time = time.time()
        if current_time - last_send > send_interval:
            send_servo(pan_angle, tilt_angle)
            last_send = current_time

        # draw crosshair on detected object
        cv2.drawMarker(annotated_frame, (cx, cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # draw center crosshair
    cv2.drawMarker(annotated_frame, (frame_w//2, frame_h//2), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # show frame
    cv2.imshow("YOLO Pan-Tilt Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#  Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
'''
#save the same
#assemble the pantilt mechanism for tracking and in Arduino IDE after choosing arduino uno board and corresponding port add the following code, upload to arduino (connected to pan-tilt servos) and close Arduino IDE after saving the sketch. Without closing Arduino IDE , VS code will show error during its running.

#include <Servo.h>

Servo panServo;
Servo tiltServo;

int panAngle = 90;
int tiltAngle = 90;

void setup() {
  Serial.begin(9600);
  panServo.attach(9);
  tiltServo.attach(10);

  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      int pan = data.substring(0, commaIndex).toInt();
      int tilt = data.substring(commaIndex + 1).toInt();

      pan = constrain(pan, 0, 180);
      tilt = constrain(tilt, 0, 180);

      panServo.write(pan);
      tiltServo.write(tilt);
    }
  }
}

# close Arduino IDE and navigate back to VS code and run aerial_tracking.py
# adjust offsets and inversion of servos using python codes which will be uploaded later once I solve it :)
# Hardware connections : use bread board 

Paan and tilt servo : 
*Brown wire to GND of UNO 
*Red wire to 5V of UNO 
*Orange wire of Pan servo to PWM pin 9 of UNO
*Orange wire of Tilt servo to PWM pin 10 of UNO
*connect a laser of 3 to 4 V rating to GND and 5V of arduino with a 110 or 100 ohm resistor

