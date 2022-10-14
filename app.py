# from __future__ import division, print_function
# # coding=utf-8
# import os
# import numpy as np

# import keras
# from PIL import Image, ImageOps
# import numpy as np

# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename


# # Define a flask app
# app = Flask(__name__)
# cf_port = os.getenv("PORT")


# def model_predict(img_path):
#     np.set_printoptions(suppress=True)
    
#     # Create the array of the right shape to feed into the keras model
#     # The 'length' or number of images you can put into the array is
#     # determined by the first position in the shape tuple, in this case 1.
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
#     # Replace this with the path to your image
#     image = Image.open(img_path)
#     #resizing the image to be at least 224x224 
    
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
#     #turn the image into a numpy array
#     image_array = np.asarray(image)
    
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
#     # Load the image into the array
#     data[0] = normalized_image_array
    
        
#     # Load the model
#     model = keras.models.load_model('ripeness.h5')
    
#     # run the inference
#     preds = ""
#     prediction = model.predict(data)
#     # max_val = np.amax(prediction)*100
#     # max_val = "%.2f" % max_val
#     if np.argmax(prediction)==0:
#         preds = f"Unripe😑"
#     elif np.argmax(prediction)==1:
#         preds = f"Overripe😫"
#     else :
#         preds = f"ripe😄"

#     return preds


# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path)
#         return preds
#     return None


# if __name__ == '__main__':
#     if cf_port is None:
#         app.run(host='0.0.0.0', port=8000, debug=True)
#     else:
#         app.run(host='0.0.0.0', port=int(cf_port), debug=True)


import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# Banana Ripeness Detection 🍌
''')
st.write("A Image Classification Web App That Detects the Ripeness Stage of Banana")

file = st.file_uploader("", type=['jpg','png','jpeg'])

def predict_stage(image_data,model):
    size = (224, 224)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = ""
    prediction = model.predict(data)
    if np.argmax(prediction)==0:
        preds = f"Overripe😫"
    elif np.argmax(prediction)==1:
        preds = f"ripe😄"
    else :
        preds = f"Unripe😑"
    return preds

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('ripeness.h5')
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        prediction = predict_stage(image, model)
    
        st.text(prediction)