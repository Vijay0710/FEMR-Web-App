from django.http import HttpResponse
from django.shortcuts import render,redirect
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import utils
import tensorflow as tf
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
# from tensorflow.python.keras.utils import to_categorical
from keras_preprocessing import image
from .forms import *
from sentiment.forms import AnalysisForm
# import scipy
import os
import emoji
from PIL import Image
from django.core.files.storage import FileSystemStorage
# import cv2
# from PIL import Image
# Create your views here.


def predict(fileUrl):
    interpreter = tf.lite.Interpreter(model_path="F:/faceemotion/sentiment/model.tflite")
    interpreter.allocate_tensors()

        # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()

    img_path = "F:/faceemotion/"+fileUrl
    img = image.load_img(img_path, grayscale=True, target_size=(48,48))
    print("Image of pixel 48 x 48")
    print(img)

    label_dict = {0 : 'Angry', 1 : 'Disgust '+'\U0001F922', 2 : 'Fear', 3 : 'Happiness '+'\U0001F60A', 4 : 'Sad ' + '\U0000E413', 5 : 'Surprise', 6 : 'Neutral '+'\U0001F610'}

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    input_shape = input_details[0]['shape']
    input_data = np.array(x)
    interpreter.set_tensor(input_details[0]['index'], input_data)
        # prediction = (interpreter.set_tensor(input_details[0]['index'],x_test[1:2]))
    interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    prediction = np.argmax(output_data)
    print('The predicted emotion is : ' + label_dict[prediction])
    my_image = image.load_img(img_path)
    plt.imshow(my_image)
    return label_dict[prediction]


# def say_hello(request):
#     if request.method == 'POST':
#         my_file = request.FILES['testing_file']
#         print(my_file)
        
#         interpreter = tf.lite.Interpreter(model_path="F:/faceemotion/sentiment/model.tflite")
#         interpreter.allocate_tensors()

#         # Get input and output tensors.
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()

#         interpreter.allocate_tensors()



#         img_path = "F:/faceemotion/sentiment/static/man_neutral_test.jpg"
#         # img_path = 'sad_women_test.jpg'
#         # img_path = 'fear_man_test.jpg'
#         # img_path = 'happy_test_women.jpg'
#         # img_path = 'output-onlinejpgtools.jpg'

        
#         img = image.load_img(img_path, grayscale=True, target_size=(48,48))
#         label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}


#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)

#         input_shape = input_details[0]['shape']
#         input_data = np.array(x)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         # prediction = (interpreter.set_tensor(input_details[0]['index'],x_test[1:2]))
#         interpreter.invoke()

#         # The function `get_tensor()` returns a copy of the tensor data.
#         # Use `tensor()` in order to get a pointer to the tensor.
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         print(output_data)
#         prediction = np.argmax(output_data)
#         print('The predicted emotion is : ' + label_dict[prediction])
#         my_image = image.load_img(img_path)
#         plt.imshow(my_image)
        

#         return render(request,'hello.html')
#     else:
#         return render(request,'hello.html')


def say_hello(request):
    if(request.method == 'GET'):
        sentimentAnalysis = Analysis.objects.all()
    return render(request,'hello.html',{'images':sentimentAnalysis})



    # if request.method == 'POST':
    #     form = AnalysisForm(request.POST,request.FILES)
    #     print(form)
    #     if form.is_valid():
    #         form.save()
    #         return redirect('success')
    # else:
    #     form = AnalysisForm()    

def demo(request):
    result = ' '
    imageURL = ''
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        print(uploaded_file.name)
        print("The size is : " + str(uploaded_file.size))


        fs = FileSystemStorage()
        if(os.path.exists("F:/faceemotion/"+fs.url(uploaded_file))):
            print('File already exists')
        else:
            fs.save(uploaded_file.name,uploaded_file)
            image_path = str("F:/faceemotion/"+fs.url(uploaded_file))
            image_file = Image.open(image_path)
            image_file.save("demoImage.jpg",quality=95)

        
        print("The file url is : " + str(fs.url(uploaded_file)))
        uploaded_file_url =  str(fs.url(uploaded_file))
        result = predict(uploaded_file_url)
        imageURL = fs.url(uploaded_file)
    return render(request,'demo.html', {'result': result,'Image':imageURL})

def delete_image(request,pk):
    if request.method == 'POST':
        objects = Analysis.objects.get(pk=pk)
        objects.delete()
    return redirect('success')



def success(request):
    return HttpResponse('successfully uploaded')