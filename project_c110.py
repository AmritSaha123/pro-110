# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(0)
#vid = cv2.imread("images.jpg")
my_model = tf.keras.models.load_model("model1.h5") 
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    if ret:
        frame=cv2.flip(frame,1)
    #frame = vid
        img = cv2.resize(frame,(224,224))
        test_image = np.array(img,dtype=np.float32)
        test_image = np.expand_dims(test_image,axis=0)
        n_image = test_image/255.0
        predict = model.predict(n_image)
        print("prediction",predict)
        rock =int(predict[0][0]*100)
        paper =int(predict[0][1]*100)
        scissors =int(predict[0][3]*100)
        print(f'rock: {rock}% paper:{paper}% scissors:{scissors}%')
        cv2.imshow('frame', frame)

    # Display the resulting frame
      
    # Quit window with spacebar
        key = cv2.waitKey(1)
    
        if key == 32:
            break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()