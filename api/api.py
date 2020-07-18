import requests
import zipfile
import io
from PIL import Image
from flask import Flask, render_template, url_for, request, send_file, jsonify, abort

# start application
app = Flask(__name__)

# routes
@app.route('/api/v1.0/inference/<save>', methods=['POST'])
def inference(save):
    # verify save value
    if save != 'true' or save != 'false':
        abort(400)  # bad req

    for filename in request.files.keys():
        # hit backend inference API and return boxed snakes
        img_file = {'file' : request.files[filename]}
        response = requests.post(url='http://localhost:5001/inference/' + save, files=img_file)

        # pull image from response and send back to user
        img = Image.open(io.BytesIO(response.content))
        img = np.array(img)
        file_obj = io.BytesIO()
        ret_img = Image.fromarray(img.astype('uint8'))
        ret_img.save(file_obj, 'jpeg')
        file_obj.seek(0)
        return send_file(file_obj, attachment_filename='ret.jpg', mimetype='image/jpeg')

# REST API endpoints
@app.route('/api/v1.0/gpu', methods=['GET'])
def gpu():
    response = requests.get(url='http://localhost:5001/gpu')
    return response.json()

@app.route('/api/v1.0/test', methods=['GET'])
def test():
    response = requests.get(url='http://localhost:5001/test')
    zf = io.BytesIO(response.content)
    return send_file(zf, attachment_filename='test.zip', mimetype='zip')

@app.route('/api/v1.0/train', methods=['GET'])
def train():
    response = requests.get(url='http://localhost:5001/train')
    zf = io.BytesIO(response.content)
    return send_file(zf, attachment_filename='train.zip', mimetype='zip')
    
# run in Docker container
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
