from flask import Flask, Response ,send_file , send_from_directory, request
import os

import cv2

import base64
from PIL import Image
from StringIO import StringIO

import json

import sys

from skimage.segmentation import mark_boundaries

import numpy as np
import math
import copy


### Setup Sys path for easy imports
# base_dir = "/media/harborned/ShutUpN/repos/dais/p5_afm_2018_demo"
# base_dir = "/media/upsi/fs1/harborned/repos/p5_afm_2018_demo"

def GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo"):
	cwd = os.getcwd()
	split_cwd = cwd.split("/")

	base_path_list = []
	for i in range(1, len(split_cwd)):
		if(split_cwd[-i] == base_dir_name):
			base_path_list = split_cwd[:-i+1]

	if(base_path_list == []):
		raise IOError('base project path could not be constructed. Are you running within: '+base_dir_name)

	base_dir_path = "/".join(base_path_list)

	return base_dir_path

base_dir = GetProjectExplicitBase(base_dir_name="p5_afm_2018_demo")



#TODO remove dependancy on reloading the model
#add all model folders to sys path to allow for easy import
models_path = os.path.join(base_dir,"models")

model_folders = os.listdir(models_path)

for model_folder in model_folders:
	model_path = os.path.join(models_path,model_folder)
	sys.path.append(model_path)



#add all explanation folders to sys path to allow for easy import
explanations_path = os.path.join(base_dir,"explanations")

explanation_folders = os.listdir(explanations_path)

for explanation_folder in explanation_folders:
	explanation_path = os.path.join(explanations_path,explanation_folder)
	sys.path.append(explanation_path)



app = Flask(__name__)


def readb64(base64_string,convert_colour=True):
	sbuf = StringIO()
	sbuf.write(base64.b64decode(base64_string))
	pimg = Image.open(sbuf)
	if(convert_colour):
		return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
	else:
		return np.array(pimg)

def encIMG64(image,convert_colour = False):
	if(convert_colour):
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	retval, img_buf = cv2.imencode('.jpg', image)

	return base64.b64encode(img_buf)




def DecodeTestImages(images):
	test_images = []

	for string_image in images:
		test_images.append(readb64(string_image))

	return test_images
	


def FilterExplanations(explanations_json,dataset_name,model_name):
	available_explanations = [explanation for explanation in explanations_json["explanations"] if (dataset_name in [dataset["dataset_name"] for dataset in explanation["compatible_datasets"]]) and (model_name in [model["model_name"] for model in explanation["compatible_models"]])]
	print("filtering by :" +dataset_name + " and " + model_name)
	return {"explanations":available_explanations}


@app.route("/explanations/get_available_for_filters/<string:filters>", methods=['GET'])
def GetAvailableExplanationsJSONforFilters(filters):
	print("filters",filters)
	filters_split = filters.split(",")
	dataset_name = filters_split[0]
	model_name = filters_split[1]

	available_explanations = FilterExplanations(explanations_json,dataset_name,model_name)
	
	return json.dumps(available_explanations)


@app.route("/explanations/get_available", methods=['GET'])
def GetAvailableExplanationsJson():
	return json.dumps(explanations_json)


def LoadExplainerFromJson(explanation_json,model_instance):
	ExplanationModule = __import__(explanation_json["script_name"]) 
	ExplanationClass = getattr(ExplanationModule, explanation_json["class_name"])
	
	return ExplanationClass(model_instance)


def LoadModelFromJson(model_json,dataset_json):
	model_name = model_json["model_name"]
	
	ModelModule = __import__(model_json["script_name"]) 
	ModelClass = getattr(ModelModule, model_json["class_name"])
	
	dataset_name = dataset_json["dataset_name"]
	input_image_height = dataset_json["image_y"]
	input_image_width = dataset_json["image_x"]
	input_image_channels = dataset_json["image_channels"]


	n_classes = len(dataset_json["labels"]) 
	
	#TODO need a clever way of handling additonal args
	learning_rate = 0.001
	dropout = 0.25

	additional_args = {"learning_rate":learning_rate,"dropout":dropout}

	trained_on_json = [dataset for dataset in model_json["trained_on"] if dataset["dataset_name"] == dataset_name][0]

	model_path = os.path.join(models_path,model_name,trained_on_json["model_path"])
	model_instance = ModelClass(input_image_height, input_image_width, input_image_channels, n_classes, model_dir=model_path, additional_args=additional_args)
	model_instance.LoadModel(model_path) ## for this model, this call is redundant. For other models this may be necessary. 

	return model_instance

###to fix: this function is pretty inefficient
def listOrganizer(organized_list, to_be_organized_list):
	improved_list = list()
	for entry in organized_list:
		for item in to_be_organized_list:
			if (entry[0] == item[0]):
				improved_list.append(item)
	return improved_list


def ImagePreProcess(image):
	#TODO check if this division by 255 is needed
	if(True):
		image = image/255.0
	
	return image.astype(np.float32)


def CreateAttributionMap(attribution_slice,slice_weights):
	output_map = np.array(attribution_slice).astype(np.float32)

	for region_weight in slice_weights:
		# print(region_weight[0],region_weight[1])
		output_map[output_map == region_weight[0]] = region_weight[1]

	return output_map


@app.route("/explanations/attribution_map", methods=['POST'])
def GetAttributionMap():
	raw_json = json.loads(request.data)

	attribution_slices = json.loads(raw_json["attribution_slices"])
	attribution_slice_weights = json.loads(raw_json["attribution_slice_weights"])

	attribution_map = CreateAttributionMap(attribution_slices,attribution_slice_weights)

	json_data = json.dumps({'attribution_map': attribution_map.tolist()})

	return json_data

def getThreeGreatestRegion(average_list):
	region1 = (0, 0)
	region2 = (0, 0)
	region3 = (0, 0)
	for i in range(len(average_list)):
		if abs(average_list[i][1]) > abs(region1[1]):
			region3 = region2
			region2 = region1
			region1 = average_list[i]
		elif abs(average_list[i][1]) > abs(region2[1]):
			region3 = region2
			region2 = average_list[i]
		elif abs(average_list[i][1]) > abs(region3[1]):
			region3 = average_list[i]
	return region1, region2, region3

def getAverageSDand3RegionPicture(average_list, sd_list, region_list, picture_to_be_edited, weight_threshold, sd_threshold):
	three_region_picture = copy.deepcopy(picture_to_be_edited)
	average_picture = copy.deepcopy(picture_to_be_edited)
	sd_picture = copy.deepcopy(picture_to_be_edited)
	first_reg, second_reg, third_reg = getThreeGreatestRegion(average_list)
	print(first_reg)
	print(second_reg)
	print(third_reg)
	for i in range(128):
		for j in range(128):
			for k in range(len(average_list)):
				#print(weight_pair[0])
				#print(i)
				#print(j)
				#print(region_list[i][j])
				if average_list[k][0] == region_list[i][j]:
					if abs(average_list[k][1]) >= weight_threshold:
						if average_list[k][1] < 0:
							new_red_value = average_picture[i][j][0] + 1
							average_picture[i][j][0] = new_red_value
						else:
							new_green_value = average_picture[i][j][1] + 1
							average_picture[i][j][1] = new_green_value
					if sd_list[k][1] > sd_threshold:
						new_blue_value = sd_picture[i][j][2] + 1
						sd_picture[i][j][2] = new_blue_value
					region_boolean = average_list[k][0] == first_reg[0] or average_list[k][0] == second_reg[0] or average_list[k][0] == third_reg[0]
					#print(average_list[k][0])
					#print(region_boolean)
					if abs(average_list[k][1]) >= weight_threshold  and region_boolean:
						if average_list[k][1] < 0:
							new_red_value = three_region_picture[i][j][0] + 1
							three_region_picture[i][j][0] = new_red_value
						else:
							new_green_value = three_region_picture[i][j][1] + 1
							three_region_picture[i][j][1] = new_green_value
	return average_picture, sd_picture, three_region_picture

@app.route("/explanations/explain", methods=['POST'])
def Explain():
	raw_json = json.loads(request.data)

	dataset_json = json.loads(raw_json["selected_dataset_json"])
	model_json = json.loads(raw_json["selected_model_json"])
	explanation_json = json.loads(raw_json["selected_explanation_json"])

	input_image = ImagePreProcess(readb64(raw_json["input"],convert_colour=False))

	dataset_name = dataset_json["dataset_name"]
	model_name = model_json["model_name"]
	explanation_name = explanation_json["explanation_name"]

	if(not model_name in loaded_models):
		loaded_models[model_name] = {}

	if(not dataset_name in loaded_models[model_name] ):
		loaded_models[model_name][dataset_name] = LoadModelFromJson(model_json,dataset_json)



	if(not explanation_name in loaded_explanations):
		loaded_explanations[explanation_name] = {}

	if(not model_name in loaded_explanations[explanation_name]):
		loaded_explanations[explanation_name][model_name] = {}

	if(not dataset_name in loaded_explanations[explanation_name][model_name]):
		loaded_explanations[explanation_name][model_name][dataset_name] = LoadExplainerFromJson(explanation_json,loaded_models[model_name][dataset_name])


	explanation_instance = loaded_explanations[explanation_name][model_name][dataset_name]
	
	#TODO allow for better handling of additonal arguments, currently additional arguments for ALL explanations must be placed here
	additional_args = {
	"num_samples":100,
	"num_features":300,
	"min_weight":0.01, 
	"model_name":model_name, 
	"dataset_name":dataset_name, 
	"num_background_samples":200
	}

	display_explanation_input = False
	if(display_explanation_input):
		cv2_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
		cv2.imshow("image: input_image",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	#explanation_image, explanation_text, prediction, additional_outputs = explanation_instance.Explain(input_image,additional_args=additional_args)
	
	explanation_image, explanation_text, prediction, additional_outputs, regions = explanation_instance.Explain(input_image,additional_args=additional_args)
	#print("explanation image:", explanation_image)
	boundary_image = mark_boundaries(explanation_image, regions, (0, 0, 0))
	#print("boundary_image:", boundary_image)
	for region in regions:
		print(region)
	#print("mask:", additional_outputs["mask"])
	#print("mask length:", len(additional_outputs["mask"]))
	#print("mask segment:", additional_outputs["mask"][0])
	#print("mask segment length:", len(additional_outputs["mask"][0]))
	previous_explanations_file = "explanation_statistics.json"
	old_explanation_json = open(previous_explanations_file, 'r')
	old_explanation_stats = json.load(old_explanation_json)
	iteration = old_explanation_stats['iteration']
	sum_of_weights = old_explanation_stats['sum_of_weights']
	average_weights = old_explanation_stats['average_weights']
	sum_of_squares = old_explanation_stats['sum_of_squares']
	standard_deviations = old_explanation_stats['standard_deviations']
	img_name = old_explanation_stats['image_title']
	if iteration == 0:
		iteration += 1
		average_weights = additional_outputs["attribution_slice_weights"]
		sum_of_weights = additional_outputs["attribution_slice_weights"]
		new_sum_of_squares_list = list()
		new_standard_deviation_list = list()
		for weight_pair in additional_outputs["attribution_slice_weights"]:
			new_sum_of_squares = weight_pair[1]**2
			new_sum_of_squares_list.append((weight_pair[0], new_sum_of_squares))
			new_standard_deviation_list.append((weight_pair[0], 0))
		standard_deviations = new_standard_deviation_list
		sum_of_squares = new_sum_of_squares_list
	else:
		new_sum_list = list()
		new_sum_of_squares_list = list()
		iteration += 1
		new_average_list = list()
		new_standard_deviation_list = list()
		attribution_slice_weights_index = 0
		slice_weights_for_instance = listOrganizer(sum_of_weights, additional_outputs["attribution_slice_weights"])
		for weight_pair in sum_of_weights:
			new_sum = weight_pair[1] + slice_weights_for_instance[attribution_slice_weights_index][1]
			new_sum_list.append((weight_pair[0], new_sum))
			new_sum_of_squares = sum_of_squares[attribution_slice_weights_index][1] + slice_weights_for_instance[attribution_slice_weights_index][1]**2
			new_sum_of_squares_list.append((weight_pair[0], new_sum_of_squares))
			new_average = new_sum/iteration
			new_average_list.append((weight_pair[0], new_average))
			#print(new_sum_of_squares)
			#print(((new_sum**2)/iteration))
			#print((new_sum_of_squares-((new_sum**2)/iteration)))
			#print((1/(iteration-1))*(new_sum_of_squares-((new_sum**2)/iteration)))
			#print(((new_sum_of_squares-((new_sum**2)/iteration)))/(iteration-1))
			new_standard_deviation = math.sqrt(abs((((new_sum_of_squares-((new_sum**2)/iteration)))/(iteration-1))))
			new_standard_deviation_list.append((weight_pair[0], new_standard_deviation))
			attribution_slice_weights_index += 1
		sum_of_weights = new_sum_list
		average_weights = new_average_list
		standard_deviations = new_standard_deviation_list
		sum_of_squares = new_sum_of_squares_list
	old_explanation_json.close()

	##### testing attribution maps
	# if("attribution_slices" in additional_outputs.keys() and "attribution_slice_weights" in additional_outputs.keys() ):
	# 	attribution_map = CreateAttributionMap(additional_outputs["attribution_slices"],additional_outputs["attribution_slice_weights"])
	# # print("")
	# print(attribution_map)
	# print("")

	avg_pic, sd_pic, three_reg_pic = getAverageSDand3RegionPicture(average_weights, standard_deviations, regions, input_image, additional_args["min_weight"], 0.1)
	#print("average pic:", avg_pic)

	#TODO check if this is needed
	if(explanation_image.max() <=1):
		explanation_image_255 = explanation_image*255
		avg_pic_255 = avg_pic*255
		sd_pic_255 = sd_pic*255
		three_reg_pic_255 = three_reg_pic*255
		boundary_image_255 = boundary_image*255
		boundary_image_255 = boundary_image_255.astype(np.float32)
	else:
		explanation_image_255 = explanation_image

	encoded_explanation_image = encIMG64(explanation_image_255,False)
	encoded_avg_pic = encIMG64(avg_pic_255,True)
	encoded_boundary_image = encIMG64(boundary_image_255,True)
	encoded_sd_pic = encIMG64(sd_pic_255, True)
	encoded_three_reg_pic = encIMG64(three_reg_pic_255, True)
	#print("explanation_image_255:", explanation_image_255)
	#print("avg_pic_255:", avg_pic_255)
	#print("boundary_image_255:", boundary_image_255)
	#print("encoded_explanation_image:", encoded_explanation_image)
	#print("encoded_avg_pic:", encoded_avg_pic)

	### test images by displaying pre and post encoding
	display_encoded_image = False

	if(display_encoded_image):
		decoded_image = readb64(encoded_explanation_image)
		cv2_image = decoded_image
		cv2.imshow("image 0",cv2_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
	labels = [label["label"] for label in dataset_json["labels"]]
	labels.sort()

	print("prediction:"+str(labels[int(prediction)]))#+" - "+labels[prediction)
	# json_data = json.dumps({'prediction': labels[prediction[0]],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image})
	# json_data = json.dumps({'prediction': labels[int(prediction)],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image, "additional_outputs":additional_outputs})

	json_data = json.dumps({'prediction': labels[int(prediction)],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image, "average_picture":encoded_avg_pic, "boundary_image":encoded_boundary_image, "standard_deviation_picture":encoded_sd_pic, "three_region_picture":encoded_three_reg_pic, "additional_outputs":additional_outputs, "iteration": iteration, "sum_of_weights": sum_of_weights, "sum_of_squares": sum_of_squares, "average_weights": average_weights, "standard_deviations":standard_deviations})
	new_explanation_file = open("explanation_statistics.json", "w")
	json_explanations_stats = json.dumps({"iteration":iteration, "sum_of_weights":sum_of_weights, "sum_of_squares":sum_of_squares, "average_weights":average_weights, "standard_deviations":standard_deviations, "image_title":img_name})
	new_explanation_file.write(json_explanations_stats)
	new_explanation_file.close()

	return json_data



if __name__ == "__main__":
	print("load explanations jsons")

	#### load explanations json
	explanations_json_path = os.path.join(explanations_path,"explanations.json")

	explanations_json = None
	with open(explanations_json_path,"r") as f:
		explanations_json = json.load(f)
	

	loaded_models = {}
	loaded_explanations = {}

	print('Starting the API')
	app.run(host='0.0.0.0',port=6201)

