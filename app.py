from flask import Flask, render_template, request

# # import tensorflow.keras.utils.load_img
# # import tensorflow.keras.preprocessing.image.img_to_array
# # from tf.keras.preprocessing.image import
# import tensorflow.keras.preprocessing.image.load_img
from PIL import Image
import tensorflow
import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

app = Flask( __name__)
model = VGG16()

@app.route('/', methods =['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" +imagefile.filename
    imagefile.save(image_path)
# load an image from file
    image = load_img(image_path, target_size=(224, 224))
    #convert the image  pixel to a numpy array
    image = img_to_array(image)
    #reshape data for the model
    image = image.reshape(1, image.shape[0],image.shape[1],image.shape[2])
    #prepare the image for the VGG Model
    image = preprocess_input(image)
    #predict the probability  across all output classes
    yhat = model.predict(image)
    #convert the probabilities to class labels / decode predictions
    label = decode_predictions(yhat)
    #retrieve the most likely result : eg: highest probability
    label = label[0][0]
    
    #print classification

    classfification = '%s (%.2f%%' % (label[1], label[2]*100)


    return render_template('index.html', prediction= classfification)


if __name__ == '__main__':
    app.run(port=5003, debug=True, use_reloader=True)
