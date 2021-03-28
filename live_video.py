from keras.preprocessing.image import img_to_array
import cv2
import imutils
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')

video_file_name = "sample/live_vid2.mp4"

video_emotion_model_path = 'models/model_num.hdf5'

use_live_video=False

emotion_classifier = load_model(video_emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","fear", "happy", "sad", "surprised",
 "neutral"]

cv2.namedWindow('Student Attention Detector')
emotions_map = {
    "angry" : 0,
    "disgust" : 0,
    "fear" : 0,
    "happy" : 0,
    "sad" : 0,
    "surprised" : 0,
    "neutral" : 0
}


count = 0

if use_live_video==True:
    cap = cv2.VideoCapture(0)
else:      
    cap = cv2.VideoCapture(video_file_name)
  
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
current_second = 0



while 1:
    try:
        if cap!=None:
            ret, frame = cap.read()
            frame = imutils.resize(frame,width=400)
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
            canvas = np.zeros((350, 400, 3), dtype="uint8")
            
            if (len(faces)==0):
                attentive=False
                cv2.putText(frame, "Not-Attentive (student unavailable)", (10, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 255), 2)
                    
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                eyes = eye_cascade.detectMultiScale(roi)
                for (ex,ey,ew,eh) in eyes[:2]:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                emotions_map[label]+=1
            
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%, {}s".format(emotion, prob * 100, current_second)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                                  (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
                    
                    attentive = False;
                    if (len (eyes)>=1):
                        attentive = True;
                    
                    if (attentive):
                        label_text = "Attentive ("+label+")"
                    else:
                        label_text = "Not-Attentive ("+label+")"
                        
                    cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
            cv2.imshow('Student Attention Detector',frame)
            cv2.imshow('Face Emotion Probabilities using AI', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        break
        print ("Exiting")

cap.release()
cv2.destroyAllWindows()