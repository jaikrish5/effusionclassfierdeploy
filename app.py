from flask import Flask, render_template, request, jsonify
import os
import json 
# import requests as rq
from skimage import io
from skimage.transform import rescale


# from urllib.request import urlopen as uReq
import glob
import resnet
import pickle
import numpy as np
from PIL import Image 
from numpy import asarray
from skimage.transform import resize 


from keras.preprocessing.image import ImageDataGenerator



def preprocess_img(img, mode):
    img = (img - img.min())/(img.max() - img.min())
    # img = rescale(img, 0.25, multichannel=True, mode='constant')
    img = resize(img, (256, 256))
    
    return img



app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = pickle.load(open('models/best_model3.hdf5','rb'))


@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')

@app.route('/scan', methods=['GET','POST'])  # This will be called from UI
def math_operation():
    if request.method == 'POST':
        try:
            image =request.files['myImage']
            img = Image.open(image) 

            numpydata = asarray(img) 
            # image = io.imread(image)

            # image = np.array(map(int,image))
            print(type(numpydata))
            num_len = numpydata.shape
            print(num_len)
            
            if len(num_len) !=2:
                numpydata = numpydata[:,:,0]
            print(numpydata.shape)


            img_channels = 1
            img_rows = 256
            img_cols = 256

            nb_classes = 2
            val_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
            val_model.load_weights('models/best_model3.hdf5')

            img = preprocess_img(numpydata[:, :, np.newaxis], 'validation')
            print(img.shape)
            prediction=val_model.predict(img[np.newaxis,:])
            print(prediction)

            no_infection = prediction[0][0]
            infection = prediction[0][1]

            if  no_infection > 0.75:
                verdict = int(no_infection*100)
                predict = 'no_infection'
            else:
                verdict = int(infection*100)   
                predict = 'infection' 
            


            prediction = [verdict, predict] 

                   
            return render_template('results1.html',result=prediction) 

        except:    
            return render_template('index.html')   

        
    else:
        return render_template('index.html')    

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=8080)

if __name__ == '__main__':
    app.run(debug=True)

