from django.shortcuts import render
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from django.conf import settings
from django.http import JsonResponse,Http404
import numpy
from random import randint

# Create your views here.
def readmission_view(request):
	loaded_model_json = None
	json_file = open('static/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	print loaded_model_json
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("static/model.h5")
	print("Loaded model from disk")
	data = []
	res1 = numpy.array([[-0.49998495, -0.29902692, -0.09806889,  0.10288913,  3.92109167,
        -0.29902692,  0.10288913,  0.70576322, -0.09806889, -0.09806889,
         1.10767928, -0.49998495, -0.49998495, -0.49998495,  0.50480519,
        -0.49998495,  0.10288913, -0.49998495, -0.49998495, -0.49998495,
        -0.49998495]])
	res2 = numpy.array([[-0.49998495, -0.49998495, -0.02130989, -0.38031619, -0.38031619,
         0.3376964 , -0.26064742, -0.49998495,  4.0474281 , -0.49998495,
         1.05570899, -0.49998495, -0.49998495, -0.38031619,  0.45736516,
        -0.38031619,  0.57703393, -0.49998495, -0.38031619, -0.49998495,
        -0.38031619]])
	res3 = numpy.array([[-0.3500394 , -0.42501217, -0.3500394 , -0.05014829, -0.42501217,
         0.02482449,  0.09979727, -0.49998495,  4.37324558, -0.42501217,
         0.47466115, -0.49998495, -0.49998495, -0.42501217, -0.3500394 ,
         0.09979727, -0.05014829, -0.49998495, -0.27506662, -0.42501217,
        -0.42501217]])
	res1 = res1.reshape(1,1,21) 
	res2 = res2.reshape(1,1,21)
	res3 = res3.reshape(1,1,21)
	data.append(res1)
	data.append(res2)
	data.append(res3)
	data = numpy.array(data)
	print data.shape
	i = randint(0,2)
	loaded_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
	print(loaded_model.summary())
	scores = loaded_model.evaluate(data[i], numpy.array([1]), verbose=0)
	return JsonResponse(scores[0],status=200,safe=False)