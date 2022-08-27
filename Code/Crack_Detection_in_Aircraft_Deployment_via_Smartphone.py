import cv2
import numpy as np
import urllib.request
import imutils


import tensorflow as tf
import tensorflow.keras.models

# Load Trained Model
new_model = tf.keras.models.load_model('trained_model.h5')

# Connect to smartphone
url='http://192.168.240.2:8080/shot.jpg'

# Take the input from the smartphone camera and perform prediction
while cv2.waitKey(1) == -1:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    img = imutils.resize(img, width=450)

    cv2.imwrite('snapshot.png', img)

    image = tf.keras.preprocessing.image.load_img("snapshot.png",target_size=(128, 128))
    input_arr = np.array([tf.keras.preprocessing.image.img_to_array(image)]).astype('float32') / 255
    predictions = new_model.predict(input_arr)

    if (predictions[0] >= 0.5):
        print("Crack Detected")
    else:
        print("No Crack Detected")

    gray = cv2.imread('snapshot.png', cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('snapshot.png', img)
    gray = cv2.imread('snapshot.png', cv2.IMREAD_GRAYSCALE)
    Reading_Img = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    _, Threshold_Img = cv2.threshold(Reading_Img, 150, 255, cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img, 90, 100)
    contours, _ = cv2.findContours(Canny_Img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Draw_Contours = cv2.drawContours(Reading_Img, contours, -1, (255, 0, 0), 1)


    canny_3d = cv2.cvtColor(Canny_Img, cv2.COLOR_GRAY2BGR)

    np_hor1 = np.hstack((img,canny_3d))
    np_hor2 = np.hstack((Threshold_Img, Draw_Contours))

    final = np.vstack((np_hor1, np_hor2))
    cv2.imshow('Final',final)


cv2.waitKey()
cv2.destroyAllWindows()


