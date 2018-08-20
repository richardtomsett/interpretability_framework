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


# dhm Drawing a "tinted" image with weights and mask overlaid

# scaling weights in range 0 -100
def scaleWeights(weights):
#  print(weights.shape)
    scaledweights = weights.copy()
    weights_vals = weights[:,1]
    print(weights_vals)
    minw = np.amin(weights_vals)
    maxw = np.amax(weights_vals)
  
    diffw = maxw - minw
    print("Scaling:", minw, maxw, diffw)
    if diffw==0:
      print("ERROR: 0 difference in weights")
      return weights
    
    for i in range(len(weights)):
      scw = int(100*(weights[i,1]-minw)/diffw)
      scaledweights[i] = np.array([weights[i,0],scw])

    return scaledweights

def getWeightForSeg(weights,seg):
    for i in range(len(weights)):
      if seg == weights[i,0]:
        return weights[i,1]
      
    return -100

# generates three numbers for BGR
# B covers the neutral (mask=0), G covers the positive (mask=2), R covers the negative(mask = 1)
# note due to the different levels of weight, we need to bump up the colours differently
# assume that the image is encoded as RGB

tintMultR = 3 # 3
tintMultB = 4 #4
tintMultG = 3 # 3
tintDefault=0
tintFloorR = 50
tintFloorB = 20
tintFloorG = 0

def tintForWtAndMask3(wt,msk):
    if msk == 0:
        return tintDefault,tintDefault,min(255,(wt+tintFloorR)*tintMultR) #B
    elif msk == 1:
        return min(255,(wt+tintFloorB)*tintMultB),tintDefault,tintDefault #R
    elif msk == 2:
        return tintDefault,min(255,(wt+tintFloorG)*tintMultG),tintDefault #G
    else:
        return 0,0,0

def tintAt3(i,j,segments,scaled_weights,mask):
    segno = segments[i,j]
    segwt = getWeightForSeg(scaled_weights,segno)
    msk = mask[i,j]
    return tintForWtAndMask3(segwt,msk)

# end dhm for tinted image

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
	
	# dhm
	print("ModelClass=" + str(ModelClass))

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

# dhm documentation: given a list in the form ((regid,average)...) then find the three regions with the three highest average values
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

# dhm documentation: given the region averagelist, the region stdlist, the list of pixels in each region, a base picture to be overlaid, 
#      an average weight threshold (AWT) and a sd threshold (SDT)
# 
# create an image with the three highest average regions overlaid, an image with the average weights for the region overlaid,
#   and an image with the std overlaid
#
# For the average image, if the region average is extreme (ie the abs value is > AWT) the region pixels colour will be increased by colour_1 (red if neg and green if pos)
# For the st image, if the region std is extreme (ie the value is > SDT) the region pixels colour will be increased by colour_inc (blue)
# For the three highest region image, if the region average is extreme (ie the abs value is > AWT) and is one of the three highest,
#   then the region pixels colour will be increased by colour_inc (red if neg and green if pos)
#
# Note that these are done on a per region basis not a per pixel basis, ie all pixels in a given region will have the same value
# Note this presupposes the images colours are [R,G,B]

# dhm make a greyscale image
def makeGreyScale(picture):
	greyscale = copy.deepcopy(picture)

	for i in range(picture.shape[0]):
		for j in range(picture.shape[1]):
			g = (picture[i][j][0] + picture[i][j][1] + picture[i][j][2])/3
			greyscale[i][j][0]= g
			greyscale[i][j][1]= g
			greyscale[i][j][2]= g
			
	return greyscale


# dhm added the colour_inc arg - but doesnt seem to make a difference
# dhm we colour as follows:
#   average: first grayscale the image and then add light red to high negative average values, and light green to high positive average values
#   std: first grayscale the image and then add light yellow to values above 1st threshold and light orange to values above 2nd threshold
#  NB increasing the colour_inc (between 0 and 1) will intensify the colours
def getAverageSDand3RegionPicture(average_list, sd_list, region_list, picture_to_be_edited, weight_threshold, sd_threshold, sd_threshold1, colour_inc):
	three_region_picture = copy.deepcopy(picture_to_be_edited)
	average_picture = copy.deepcopy(picture_to_be_edited)
	sd_picture = copy.deepcopy(picture_to_be_edited)
	
	average_picture=makeGreyScale(average_picture) # dhm
	sd_picture=makeGreyScale(sd_picture) # dhm
	three_region_picture=makeGreyScale(three_region_picture) # dhm

	first_reg, second_reg, third_reg = getThreeGreatestRegion(average_list)
	print(first_reg)
	print(second_reg)
	print(third_reg)
	
	# dhm any  sized image is ok
	irange=picture_to_be_edited.shape[0]
	jrange=picture_to_be_edited.shape[1]
	print("Image size=" + str(irange) + "*" + str(jrange) + " colour_inc " + str(colour_inc))
	
	for i in range(irange):
		for j in range(jrange):
			for k in range(len(average_list)):
				#print(weight_pair[0])
				#print(i)
				#print(j)
				#print(region_list[i][j])
				if average_list[k][0] == region_list[i][j]:
					if abs(average_list[k][1]) >= weight_threshold:
						if average_list[k][1] < 0:
							new_red_value = average_picture[i][j][0] + colour_inc # dhm was 1
							average_picture[i][j][0] = new_red_value
						else:
							new_green_value = average_picture[i][j][1] + colour_inc # dhm was 1
							average_picture[i][j][1] = new_green_value
					if sd_list[k][1] > sd_threshold1:
						new_red_value = sd_picture[i][j][0] + colour_inc*3 # dhm was blue 
						sd_picture[i][j][0] = new_red_value # dhm ditto
						new_green_value = sd_picture[i][j][1] + colour_inc*2 # dhm added to create orange (with relative multipliers of the colour_inc)
						sd_picture[i][j][1] = new_green_value # dhm ditto
					elif sd_list[k][1] > sd_threshold:
						new_red_value = sd_picture[i][j][0] + colour_inc # dhm was blue
						sd_picture[i][j][0] = new_red_value # dhm ditto
						new_green_value = sd_picture[i][j][1] + colour_inc # dhm added to create yellow (with equal colour_inc)
						sd_picture[i][j][1] = new_green_value # dhm ditto
					region_boolean = average_list[k][0] == first_reg[0] or average_list[k][0] == second_reg[0] or average_list[k][0] == third_reg[0]
					#print(average_list[k][0])
					#print(region_boolean)
					if abs(average_list[k][1]) >= weight_threshold  and region_boolean:
						if average_list[k][1] < 0:
							new_red_value = three_region_picture[i][j][0] + colour_inc # dhm was 1
							three_region_picture[i][j][0] = new_red_value
						else:
							new_green_value = three_region_picture[i][j][1] + colour_inc # dhm was 1
							three_region_picture[i][j][1] = new_green_value

	return average_picture, sd_picture, three_region_picture

@app.route("/explanations/explain", methods=['POST'])
def Explain():

	print("EXPLAIN")
	
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

	# dhm documentation: Cadets addition to generate average pictures etc
	# It is assumed that this code is called many times for the same image, and calculates the values of average and standard deviation
	# for each region. 
	# The running totals of sum of weights and sum of squared weights for each region
	# are included in a file explanations_statistics.json in the flask_services directory in the format [(region id, relevant value)...]
	# This code iterates over each region and adds the new weight for that region to the sum of weights and sum of squared weights
	# then computes the current average and standard deviation for each region
	#
	# Then new images are created as overlays to the original input image:
	# the average image which shows where the (abs) region average is above a threshold
	# the standard deviation image which shows where the region std is above a threshold
	# the regions with the three best average values
	# See details of getAverageSDand3RegionPicture for details
	
	# There are some parameters that can be altered:
	#   the min average threshold for display is in "min_weights" in additional_args above
	#   the min std threshold for display std_thresh (below)
	#   the col_inc for incrementing the overlay colours (below)
	#
	# Questions/bugs - does this actually check the image name, if not then there might be some confusion?
	#    if there is a blank explanation_statistics.json file then an error will be produced
	#    the explanations running totals is not automatically reset to zero unless a new image is chosen
	
	#print("explanation image:", explanation_image)
	boundary_image = mark_boundaries(explanation_image, regions, (0, 0, 0))
	#print("boundary_image:", boundary_image)
	
	#for region in regions:
	#	print(region)
		
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
	
	old_explanation_json.close() # dhm better to close here rather than later

	
	# dhm
	print("ITERATION " + str(iteration))
	# mhd
	
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


	##### testing attribution maps
	# if("attribution_slices" in additional_outputs.keys() and "attribution_slice_weights" in additional_outputs.keys() ):
	# 	attribution_map = CreateAttributionMap(additional_outputs["attribution_slices"],additional_outputs["attribution_slice_weights"])
	# # print("")
	# print(attribution_map)
	# print("")

	# dhm added colour_inc and std threshold
	col_inc=0.4 # needs to be 0 to 1, I think
	std_thresh = 0.05 # was 0.1
	std_thresh1 = 0.07 # new
	av_th=0.1 # was additional_args["min_weight"], ie 0.01

	# dhm avg_pic, sd_pic, three_reg_pic = getAverageSDand3RegionPicture(average_weights, standard_deviations, regions, input_image, additional_args["min_weight"], 0.1)
	avg_pic, sd_pic, three_reg_pic = getAverageSDand3RegionPicture(average_weights, standard_deviations, regions, input_image, av_th, std_thresh, std_thresh1, col_inc)
	#print("average pic:", avg_pic)

	#TODO check if this is needed
	if(explanation_image.max() <=1):
		print("SCALING") # dhm
		explanation_image_255 = explanation_image*255
		avg_pic_255 = avg_pic*255
		sd_pic_255 = sd_pic*255
		three_reg_pic_255 = three_reg_pic*255
		boundary_image_255 = boundary_image*255
		boundary_image_255 = boundary_image_255.astype(np.float32)
	else:
		print("NOT SCALING")
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

	# dhm to show tinted image
	
	# We will show each pixel as the segment weight, by scaled by the mask, so that:
	#   value for 2 is in range 600-900, value for 1 is in range 300-599, and value for 0 is in range 0-299
	# We assume the weight is scaled 0-100
	# We need the mask and the weights from additional_outputs, and the regions returned by the explainer

	weights = np.array(additional_outputs["attribution_slice_weights"])
	mask = np.array(additional_outputs["mask"])
	scaled_weights = scaleWeights(weights)

	# convert to CV_8UC3 ( 3 channel unsigned 8 integer)
	tinted_image = np.zeros((regions.shape[0],regions.shape[1],3), np.uint8)

	for i in range(tinted_image.shape[0]):
		for j in range(tinted_image.shape[1]):
			tintb,tintg,tintr = tintAt3(i,j,regions,scaled_weights,mask)
			tinted_image[i,j] = [tintb,tintg,tintr]

	encoded_tinted_image = encIMG64(tinted_image, True)
	
	# optional print of the tinted image
	display_tinted_image = False

	if (display_tinted_image):
		cv2.imshow("tinted",tinted_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	# end dhm tinted image code

	labels = [label["label"] for label in dataset_json["labels"]]
	labels.sort()

	print("prediction:"+str(labels[int(prediction)]))#+" - "+labels[prediction)
	# json_data = json.dumps({'prediction': labels[prediction[0]],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image})
	# json_data = json.dumps({'prediction': labels[int(prediction)],"explanation_text":explanation_text,"explanation_image":encoded_explanation_image, "additional_outputs":additional_outputs})

	# dhm adding the tinted image to the return json
	json_data = json.dumps({
		'prediction': labels[int(prediction)],
		"explanation_text":explanation_text,
		"explanation_image":encoded_explanation_image,
		"average_picture":encoded_avg_pic, 
		"boundary_image":encoded_boundary_image, 
		"standard_deviation_picture":encoded_sd_pic, 
		"three_region_picture":encoded_three_reg_pic, 
		"additional_outputs":additional_outputs, 
		"iteration": iteration, 
		"sum_of_weights": sum_of_weights,
		"sum_of_squares": sum_of_squares, 
		"average_weights": average_weights, 
		"standard_deviations":standard_deviations,
		"tinted_image":encoded_tinted_image # dhm
		})

	json_explanations_stats = json.dumps({
		"iteration":iteration, 
		"sum_of_weights":sum_of_weights, 
		"sum_of_squares":sum_of_squares, 
		"average_weights":average_weights, 
		"standard_deviations":standard_deviations, 
		"image_title":img_name
		})
		
	new_explanation_file = open("explanation_statistics.json", "w")
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
