
# Mashroom Clasification Web App
<img align="center" height="300" src="Screenshot 2024-03-15 at 10.26.02 PM.png"  />


Welcome to my Mushroom Classification Web App!
This app is designed to help you detect whether a mushroom is poisonous or not by simply loading a photo, achieved 90% accuracy. 

As a data scientist, I have developed a machine learning model that is capable of accurately classifying mushrooms into their respective categories based on their visual features. 

With this web app, you can easily upload a photo of a mushroom and have it classified instantly. Whether you are a mushroom forager or simply curious about the mushrooms in your backyard, this app is a reliable tool to help you make safe decisions. 

So, go ahead and give it a try!

※CAUTION: The rejection criteria are only a guide. They may be misrecognized.

## Demo
Try this from the link!
https://mashclasification-9ckegyf6yxrlecipzw9tzh.streamlit.app/

<img align="center" height="300" src="Screen Recording 2024-03-15 at 10.15.10 PM.gif"  />

By uploading your mashroom photo (JPG or JPEG format), it can return result.

## 💻 Used Software
Python

streamlit

Keras

numpy

## How I built
I have created a mushroom classification model using deep learning techniques. Initially, I collected mushroom images by scraping and then processed them to develop the model. 

For this project, I used Keras to develop the classification model. The model has 11 layers for specific operations such as Conv2D, Activation, MaxPooling2D, Flatten, and Dense, and 2 layers for activation functions, making it a total of 13 layers. 

After training the model, I achieved a 90% accuracy rate. I saved the model as "model.h5" and then implemented this classification process using Streamlit. 

With this project, I have successfully developed a deep learning model for mushroom classification and made it available for others to use. Anyone interested in this project can access it on my Github repository.
