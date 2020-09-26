from tensorflow.keras.models import model_from_json

import cv2
import numpy as np

json_file = open("ASL_Model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("ASL_Model_weights.h5")

labels = list("ABC")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    cv2.rectangle(frame,(300,0),(500,200),(255,0,0),thickness=1)
    
    img = frame[0:200,300:500]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))
    cv2.imshow('ASL_Hand',img)
    img = np.reshape(img, (1,100,100,1))
    try:
        predictions = model.predict(img)
        print(predictions)
        cv2.putText(frame, labels[np.argmax(predictions)], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    except:
        pass

    resized_img = cv2.resize(frame, (1200, 900))  
    cv2.imshow('ASL',resized_img)

    if cv2.waitKey(10) == ord('s'): 
            break  

cap.release()  
cv2.destroyAllWindows()