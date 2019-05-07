### TRAINING YOUR OWN DATASET - [Link](http://pylessons.com/Tensorflow-object-detection-installation/).

* Collect at least 500 images that contain your object
* Annotate/label the images. Using LabelImg tool.
  * This process is basically drawing boxes around your object in an image.
	* The label program automatically will create an XML file that describes the object in the picture
	* From here, choose to open dir and pick the directory that you saved all of your images to. Now, you can begin to annotate images with the create rectbox button. Draw your box, add the name in, and hit ok. Save, hit next image, and repeat! You can press the w key to draw the box and do ctrl+s to save faster.
	* LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.

* Split data into train and test samples.
  * Training data should be around 80% and test around 20%
* Generate TF records from these splits

		* This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts, with some slight modifications to work with our directory structure.
		* First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the main folder, if you are using the same file structure issue the following command in command prompt: python xml_to_csv.py.
		* This creates a train_labels.csv and test_labels.csv file in the images folder.
		* This will be create too a test.record and a train.record file in the images folder.
		* Will be create too a labelmap.pbtxt in the training folder with labelled classification

* Setup a .config file for the model of choice (using transfer learning)
  * We can use a pre-treinned model such as faster_rcnn_inception_v2_coco.config to apply transfer learning (OR any other on the observation image about COCO)
	* need change some parameters like 'name file', 'filepath' etc.

* Train our model
* Export inference graph from new trained model
* Detect custom objects in real time


OBS:
LabelImg GitHub link
LabelImg download link

if you are training your own classifier, you will replace the following code in generate_tfrecord.py:


# TO-DO replace this with label map
```
def class_text_to_int(row_label):
if row_label == 'c':
return 1
elif row_label == 'ch':
return 2
elif row_label == 't':
return 3
elif row_label == 'th':
return 4
else:
return None
```
