from tensorflow.keras.models import model_from_json

import cv2
import numpy as np

json_file = open("10_alpha_ASL_Model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("10_alpha_ASL_Model_weights.h5")

labels = list("ABC")

def image_predict(image):
    return labels[np.argmax(model.predict(image))]

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()

        cv2.rectangle(frame,(300,0),(500,200),(255,0,0),thickness=1)
    
        img = frame[0:200,300:500]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100,100))
        img = img/255.0
        img = np.reshape(img, (1,100,100,1))
        try:
            predictions = model.predict(img)
            cv2.putText(frame, labels[np.argmax(predictions)], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        except:
            pass

        resized_img = cv2.resize(frame, (1000, 600))

        _, jpeg = cv2.imencode('.jpg', resized_img)

        return jpeg.tobytes()