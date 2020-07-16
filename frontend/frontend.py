import os
import numpy as np
import cv2
import io
import requests
from PIL import Image
from flask import Flask, render_template, url_for, request, send_file

# start application
app = Flask(__name__)

# routes
@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home():
    # handle POST req
    if request.method == 'POST':
        for filename in request.files.keys():
            # hit backend inference API and return boxed snakes
            img_file = {'file' : request.files[filename]}
            response = requests.post(url='http://localhost:5001/inference', files=img_file)

            # pull image from response and send back to user
            img = Image.open(io.BytesIO(response.content))
            img = np.array(img)
            file_obj = io.BytesIO()
            ret_img = Image.fromarray(img.astype('uint8'))
            ret_img.save(file_obj, 'jpeg')
            file_obj.seek(0)
            return send_file(file_obj, attachment_filename='ret.jpg', mimetype='image/jpeg')

    # otherwise return template
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# run in Docker container
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
