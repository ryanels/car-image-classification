# car-image-classification
This project was completed in Spring 2022 for ISYE 6740 Computational Data Analytics/Machine Learning, a graduate course through Georgia Tech.

Car image classification using Python, Keras. See the PDF report for full analysis and writeup. 

Using Stanford Cars image dataset, classify vehicles based on "Make, Model, Year" (196 classes), "Body Type" (8 classes), and "High/Low MPG" (2 classes, binary)

Not accepting push requests

# Problem Statement
Image classification is an important field that is broadly used across various industries. Whether looking at MRIs to determine presence of a medical issue, analyzing remote sensing data to determine what type of ground coverage a satellite is viewing, training a self-driving car, or even looking at products on an assembly line to locate defects, image classification is at the heart of these applications. 

With this in mind, I decided that it would be interesting to build a model(s) to classify images of cars. The model will be used to try and predict the car (make, model, year) from a test image dataset as well as predicting the body type of car (e.g., sedan, SUV, truck), and whether the car’s miles per gallon (MPG) or miles per gallon equivalent (MPGe) is high or low, a non-visual characteristic that may be influenced or correlated with the car’s physical form. 

# Data Source
For this analysis, I located a dataset with images of cars that was originally used in a paper from 2013 entitled, “3D Object Representations for Fine-Grained Categorization,” by Krause, et al (2). The entire dataset has 16,185 images, roughly split in half between training and testing subsets. There were 8144 training images and 8041 test images. 

There are additional MAT and CSV files containing label/class information for the type of car (make, model, and year) and “x” and “y” coordinates for cropping the images. 

The dataset was available through the Stanford Artificial Intelligence (AI) department’s website and through Kaggle.

# Methodology
The first step was working with the associated files. Two separate CSV files contained training and test subset details. These files contained X, Y coordinates for cropping the car images. Classes for the datasets were numeric and associated make, model, year details were not included. 

To obtain the testing subset’s classifications, I read the details from one of the MAT files and created my own test data CSV which mirrored the provided training data CSV. This was required so I could evaluate model accuracy on the test data. 

Using another MAT file, I read in the class details so I could associate the numeric classifications with the make, model, and year of the cars. This gave me a two-column CSV, which I manually updated by adding my own additional classifications, body type and a binary MPG classification. 

I decided to use eight body type categories for the analysis. When I did exploratory data analysis, I found the classes were not evenly balanced so I used a cost-sensitive approach (sklearn's class_weight method) when fitting my model. 

Finally, to prepare the images for analysis, I used the “x” and “y” coordinates from the MAT and CSV files to crop the images and then resized them so they all had the same dimensions. All processed images were saved to new folders so they could be used when building and testing my model. 

To build and test my model, I opted to use Keras, an API built on top of TensorFlow. Keras uses convolutional neural networks (CNN) and can be used to build different neural network models with different numbers of hidden layers and parameters. There are various references and tutorials online which I used to learn how to perform image classification with Keras.

To tune the models, I used the Keras Tuner package and tried different learning rates as well as different numbers of filters for some of the layers. This is an important step since tuning hyperparameters will generally improve a model’s performance. 

When building and tuning all three separate models, to avoid overfitting I used the Keras EarlyStopping method to stop when the validation loss was no longer decreasing (with a patience delay to ensure this was the case). At that point, even if the training accuracy improved, that would indicate overfitting of the model. 

I will select the model with the best tuning parameters that leads to the smallest validation loss for each different classification. For the models with fewer classes (body type and Median MPG), I will also look at the confusion matrix and classification reports (i.e., precision, recall, F1 score).  
The final accuracy rates will be those attained when the validation loss was the smallest. 

# Evaluation and Final Results

Body Type Classification

Some body types are very similar to each other, which made this classification challenging. Regardless, the model was able to predict body type across eight classes with 50.91% accuracy. 

Median MPG

Different categories of car often have higher or lower average fuel economy, so I plotted the average MPG across the different vehicle classes to understand the data. Hatchbacks have the highest average fuel economy and pickups have the worst. Since the different body types vary in appearance, intuitively it seems like using images to try and predict Low vs High MPG (below or above Median MPG) should yield a model with good prediction accuracy. 

However, when looking at the body type classes separated by Low or High MPG the picture is less clear. Although all hatchbacks had good fuel economy, and most vans and pickups had poor fuel economy, some car types (convertible, SUV) had many examples that fell into both Low and High MPG classes. This makes differentiating between vehicles that are visually similar but fall into different MPG classes challenging and will negatively affect prediction accuracy. 

With this in mind, the results of my analysis were positive. After my model stopped when the validation loss was minimized, the test dataset was predicted with 65.91% accuracy. This is a significant result since the test dataset was well balanced with a 49.10/50.90% Low/High MPG split and I used class weighting to further try and account for any imbalance. 

Make, Model, Year

Predicting the correct Make, Model, Year classification proved to be the most challenging task. This was expected and makes sense considering there are 196 classes and many of the vehicles and images are very similar to each other. 

The best model’s overall accuracy rate was 14.69% which initially didn’t seem very good. However, random chance would dictate an accuracy rate of approximately 0.51% so the model is an improvement of more than 28x. 

For the test dataset, it is also informative to look at a random selection of some predicted classes versus actual “Make, Model, Year” classes. There are many misclassifications that seem quite close to the true class and if we were to score accuracy not only on the prediction itself but on the actual class being one of the top five predictions then the accuracy would likely improve a good deal. A few similar predictions vs actual classifications are: 

1. Lamborghini Diablo Coupe 2001	vs Lamborghini Gallardo LP 570-4 Superleggera 2012
2. Mercedes-Benz Sprinter Van 2012 vs Dodge Sprinter Cargo Van 2009
3. Chevy Silverado 1500 Hybrid Crew Cab vs Chevy Silverado 1500 Regular Cab

When investigating “Make, Model, Year” classifications more closely, it’s easy to compare the predictions to the actual classes and find evidence of these intuitively close classifications. Even though the model was only trying to label the images correctly into 1 of 196 “Make, Model, Year” classes, there are an additional 2872 misclassifications (35.72%) that were still correct for either the car’s body type or the vehicle’s make even though incorrect for the exact “Make, Model, Year.” 

# Conclusion

In conclusion, I was able to use image classification to predict a non-visual MPG classification with 65.91% accuracy, body type classification with 50.91% accuracy, and “Make, Model, Year” classification with 14.69% accuracy. These are all significant results and show the power of machine learning for image classification. 

Future work may include additional image classification tasks such as training a model on certain views (e.g., front, rear, side, isometric) of vehicles rather than including all views in a single class. It should also be possible to identify the symbol of the vehicle’s make and use it to improve prediction accuracy. For example, it’s unlikely that a vehicle identified as a Buggati will be a station wagon. 

The main lessons learned were how to perform image classification using Keras and how important tuning hyperparameters can be. Initially, being new to image classification using Keras and Tensorflow, I simply tried to build a functional model and start understanding the package. For this reason, I started by hard-coding the number of filters and learning rates in my models and only achieved a “Make, Model, Year” accuracy of 4.07%. Tuning hyperparameters brought that accuracy rate up to 14.69%, so the improvement was substantial. The other models’ accuracy rates also improved. I also tried different Image Generator parameters like rotation range and whether to flip images in order to improve each model’s performance. 
