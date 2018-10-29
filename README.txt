1. deleteall.py

Delete All scripts deletes not only images from your specified Custom Vision Project, but also
iteration and prediction images in the project. Please use it cautiously.


2. process_data.py

This scripts processes data from csv file and Azure blob storage, and upload all the images
to your specified Custom Vision project. Upon uploading images, it creates tags, iteration and 
uploads testing data to obtain prediction results.
***For detailed instruction, please read process_data instruction inside of the file.


3. data_analysis.py

This scripts reads two result csv files created by process_data (test_data and prediction_result_data)
and creates a csv file where it shows the accuracy of each classification/feature.
***Update
	The data_analysis.py scripts also writes precision and recall result to a separate csv file.
	If you have multiple iterations, the script will choose the first iteration in the list. 


4. custom_model_img_file_moving.py

This scripts read 'training_car.csv' and 'testing_car.csv', and entire image folder to split all images
into three folders (train, val, and test) based on a classifier.
For instance, if the classifier user select is 'GlassDamage,' the three folder gets splitted according to 
the label provided in training_car.csv and testing_car.csv.


5. tensorFlow_karas_custom_model.py

This script conduct transfer learning based on images splitted by 'custom_model_img_file_moving.py' script.
The CNN model is customizable and it is currently using VGG16 (Visual Geometry Group). There are many things
that user can modify as they use this script. For details, please read HowToCreateCustomModelWithTransferLearning.docx.


6. custom_model_evaluation.py

This script takes in a model (ex. VGG16.h5) and evaluates the model using the testing data.
The model classifies images in testing data and outputs an excel csv file that contains the classification.


7. data_analysis 

Another purpose of data_analysis.py script is to analyze custom_model. Based on the result that was created by
'custom_model_evaluation.py,' we can now use this script to compare the result with original labels to find out
the custom model's accuracy. 

