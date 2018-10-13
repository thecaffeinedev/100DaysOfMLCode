from flask import Flask, request, Response,send_file
from run import predict

 
app = Flask(__name__)

img1, img2, img3, img4 = predict()

@app.route('/pred1', methods=['GET'])
def pred1():
    response =  send_file(img1, mimetype='image/png')
    return response
@app.route('/pred2', methods=['GET'])
def pred2():
    response =  send_file(img2, mimetype='image/png')
    return response
@app.route('/pred3', methods=['GET'])
def pred3():
    response =  send_file(img3, mimetype='image/png')
    return response
@app.route('/pred4', methods=['GET'])
def pred4():
    response =  send_file(img4, mimetype='image/png')
    return response
 
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',threaded=False)
