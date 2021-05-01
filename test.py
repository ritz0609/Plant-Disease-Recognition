from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
import re
import cv2
import imutils

with open('model.json','r') as f:
    json = f.read()
model = model_from_json(json)

model.load_weights("model.h5")
k =[]
def sc2std(x):
    s = str(x)
    if 'e' in s:
        num,ex = s.split('e')
        if '-' in num:
            negprefix = '-'
        else:
            negprefix = ''
        num = num.replace('-','')
        if '.' in num:
            dotlocation = num.index('.')
        else:
            dotlocation = len(num)
        newdotlocation = dotlocation + int(ex)
        num = num.replace('.','')
        if (newdotlocation < 1):
            return negprefix+'0.'+'0'*(-newdotlocation)+num
        if (newdotlocation > len(num)):
            return negprefix+ num + '0'*(newdotlocation - len(num))+'.0'
        return negprefix + num[:newdotlocation] + '.' + num[newdotlocation:]
    else:
        return s

my_dict = {1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust', 4: 'Apple___healthy', 5: '___background', 6: 'Blueberry___healthy',7: 'Cherry_(including_sour)___healthy', 8: 'Cherry_(including_sour)___Powdery_mildew', 9: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 10: 'Corn_(maize)___Common_rust_', 11: 'Corn_(maize)___healthy', 12: 'Corn_(maize)___Northern_Leaf_Blight', 13: 'Grape___Black_rot', 14: 'Grape___Esca_(Black_Measles)', 15: 'Grape___healthy', 16: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 17: 'Orange___Haunglongbing_(Citrus_greening)', 18: 'Peach___Bacterial_spot', 19: 'Peach___healthy', 20: 'Pepper,_bell___Bacterial_spot', 21: 'Pepper,_bell___healthy', 22: 'Potato___Early_blight', 23: 'Potato___healthy', 24: 'Potato___Late_blight', 25: 'Raspberry___healthy', 26: 'Soybean___healthy', 27: 'Squash___Powdery_mildew', 28: 'Strawberry___healthy', 29: 'Strawberry___Leaf_scorch', 30: 'Tomato___Bacterial_spot', 31: 'Tomato___Early_blight', 32: 'Tomato___healthy', 33: 'Tomato___Late_blight', 34: 'Tomato___Leaf_Mold', 35: 'Tomato___Septoria_leaf_spot', 36: 'Tomato___Spider_mites Two-spotted_spider_mite', 37: 'Tomato___Target_Spot', 38: 'Tomato___Tomato_mosaic_virus', 39: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus' }
img='0c8fd2f4-9c26-4e6d-90b5-fae10602579c___Matt.S_CG 1072.JPG'
test_image = image.load_img(img, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
y=[]
result = model.predict(test_image)
k = np.array(result).tolist()
k = k[0]
print(result)
p = np.argmax(k)+1
s = my_dict.get(p)
y = s.split('___')
print('Plant--',y[0])
print('Disease caused--',y[1])
while True:
    frame = cv2.imread(img)
    frame = imutils.resize(frame, width=550)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, y[0], (10, 30),cv2.FONT_HERSHEY_COMPLEX, 1.0 , (0, 0, 255), 2)
    cv2.putText(frame, y[1], (10, 240),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()


