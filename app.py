import numpy as np
from flask import Flask, request, render_template
import pickle
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import os
from skimage import transform
from skimage.io import imread, imshow

# creating flask app
app = Flask(__name__)


# model = pickle.load(open("model.pkl","rb"))

# this will take me to the home page
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    # getting the image from the web
    # D:\flaskoMLModle\imageScreenshot 2022-12-10 032627.png
    meme = request.files['memefile']
    # saving the image in folder so that we can read for prediction
    image_path = "\\flaskoMLModle\\image\\" + meme.filename
    print(image_path)
    meme.save(image_path)

    # extrating the features of the image to give the model
    x = transform.resize(imread(image_path, as_gray=True), (100, 100))
    flattened = x.flatten()

    print(flattened.shape)
    # loading the saved model
    pickled_model = pickle.load(open('model_ADABoost.pkl', 'rb'))
    pred = pickled_model.predict([flattened])

    if(pred[0] == 2):
        predicted_Sentiment = "Positive"
    if (pred[0] == 1):
        predicted_Sentiment = "Neutral"
    if (pred[0] == 0):
        predicted_Sentiment = "Negative"


    return render_template('index.html',predictedSentiment = predicted_Sentiment)


if __name__ == "__main__":
    app.run(port=3000, debug=True)


