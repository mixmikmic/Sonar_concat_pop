# # Pixel-level land use classification
# 
# The notebooks in this folder contain a tutorial illustrating how to create a deep neural network model that accepts an aerial image as input and returns a land cover label (forested, water, etc.) for every pixel in the image and deploy it in ESRI's [ArcGIS Pro](https://pro.arcgis.com/) software. Microsoft's [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) is used to train and evaluate the model on an [Azure Geo AI Data Science Virtual Machine](http://aka.ms/dsvm/GeoAI). The method shown here was developed in collaboration between the [Chesapeake Conservancy](http://chesapeakeconservancy.org/), [ESRI](https://www.esri.com), and [Microsoft Research](https://www.microsoft.com/research/) as part of Microsoft's [AI for Earth](https://www.microsoft.com/en-us/aiforearth) initiative.
# 
# We recommend budgeting two hours for a full walkthrough of this tutorial. The code, shell commands, trained models, and sample images provided here may prove helpful even if you prefer not to complete the tutorial: we have provided explanations and direct links to these materials where possible.
# 

# ## Getting started
# 
# This notebook is intended for use on an NC-series [Azure Geo AI Data Science VM](http://aka.ms/dsvm/GeoAI). For step-by-step instructions on provisioning the VM, visit [our git repository](https://github.com/Azure/pixel_level_land_classification).
# 
# 1. [Train a land classification model from scratch](./02_Train_a_land_classification_model_from_scratch.ipynb)
# 
#     In this section, you'll produce a trained CNTK model that you can use anywhere for pixel-level land cover prediction.
#     
# 1. [Apply your trained model to new aerial images](./03_Apply_trained_model_to_new_data.ipynb)
# 
#     You'll predict land use on a 1 km x 1 km region not previously seen during training, and examine your results in a full-color output file.
#     
# 1. [Apply your trained model in ArcGIS Pro](./04_Apply_trained_model_in_ArcGIS_Pro.ipynb)
# 
#     You'll apply your trained model to aerial data in real-time using ESRI's ArcGIS Pro software.
# 

# ## Sample Output
# 
# This tutorial will train a pixel-level land use classifier for a single epoch: your model will produce results similar to bottom-left. By expanding the training dataset and increasing the number of training epochs, we achieved results like the example at bottom right. The trained model is accurate enough to detect some features, like the small pond at top-center, that were not correctly annotated in the ground-truth labels.
# 
# <img src="https://github.com/Azure/pixel_level_land_classification/raw/master/outputs/comparison_fullsize.PNG"/>
# 
# This notebook series will also illustrate how to apply your trained model in real-time as you scroll and zoom through regions in ArcGIS Pro:
# 
# <img src="https://github.com/Azure/pixel_level_land_classification/raw/master/outputs/arcgispro_finished_screenshot.png"/>
# 

# ## (Optional) Setting up an ArcGIS Pro trial membership
# 

# The Geo AI Data Science VM comes with ESRI's ArcGIS Pro pre-installed, but you will need to supply credentials for an ArcGIS Pro license in order to run the program. You can obtain a 21-day trial license as follows:
# 
# 1. Complete the form on the [ArcGIS Pro free trial](https://www.esri.com/en-us/arcgis/products/arcgis-pro/trial) page.
# 1. You will receive an email from ESRI. Follow the activation URL to continue with registration.
# 1. After selecting account credentials, you will be asked to provide details of your organization. Fill these out as directed and click "Save and continue."
# 1. When prompted to download ArcGIS Pro, click 'Continue with ArcGIS Pro online." (The program has already been downloaded and installed on the VM.)
# 1. Click on the "Manage Licenses" option on the menu ribbon.
# 1. In the new page that appears, you will find a "Members" section with an entry for your new username. Click the "Configure licenses" link next to your username.
# 1. Ensure that the ArcGIS Pro radio button is selected, and click the checkbox next to "Extensions" to select all extensions. Then, click "Assign."
# 
# You should now be able to launch ArcGIS Pro with your selected username and password.
# 

# ## Next steps
# 
# We recommend that you begin this tutorial series with the [training notebook](./02_Train_a_land_classification_model_from_scratch.ipynb).
# 
# In this notebook series, we train and deploy a model on a Geo AI Data Science VM. To improve model accuracy, we recommend training for more epochs on a larger dataset. Please see [our GitHub repository](https://github.com/Azure/pixel_level_land_classification) for more details on scaling up training using Azure Batch AI.
# 
# When you are done using your Geo AI Data Science VM, we recommend that you stop or delete it to prevent further charges.
# 
# For comments and suggestions regarding this notebook, please post a [Git issue](https://github.com/Azure/pixel_level_land_classification/issues/new) or submit a pull request in the [pixel-level land classification repository](https://github.com/Azure/pixel_level_land_classification).
# 

# # Apply a trained land classifier model in ArcGIS Pro
# 

# This tutorial will assume that you have already provisioned a [Geo AI Data Science Virtual Machine](http://aka.ms/dsvm/GeoAI) and are using this Jupyter notebook while connected via remote desktop on that VM. If not, please see our guide to [provisioning and connecting to a Geo AI DSVM](https://github.com/Azure/pixel_level_land_classification/blob/master/geoaidsvm/setup.md).
# 
# By default, this tutorial will make use of a model we have pre-trained for 250 epochs. If you have completed the associated notebook on [training a land classifier from scratch](./02_Train_a_land_classification_model_from_scratch.ipynb), you will have the option of using your own model file.
# 

# ## Setup instructions
# 
# ### Log into ArcGIS Pro
# 
# [ArcGIS Pro](https://pro.arcgis.com) 2.1.1 is pre-installed on the Geo AI DSVM. If you are running this tutorial on another machine, you may need to perform these additional steps: install ArcGIS Pro, [install CNTK](https://docs.microsoft.com/cognitive-toolkit/setup-windows-python) in the Python environment ArcGIS Pro creates, and ensure that [ArcGIS Pro's Python environment](http://pro.arcgis.com/en/pro-app/arcpy/get-started/installing-python-for-arcgis-pro.htm) is on your system path.
# 
# To log into ArcGIS Pro, follow these steps:
# 
# 1. Search for and launch the ArcGIS Pro program.
# 1. When prompted, enter your username and password.
#     - If you don't have an ArcGIS Pro license, see the instructions for getting a trial license in the [intro notebook](./01_Intro_to_pixel-level_land_classification.ipynb).
# 

# ### Install the supporting files
# 
# If you have not already completed the associated notebook on [training a land classifier from scratch](./02_Train_a_land_classification_model_from_scratch.ipynb), execute the following cell to download supporting files to your Geo AI DSVM's D: drive.
# 

get_ipython().system('AzCopy /Source:https://aiforearthcollateral.blob.core.windows.net/imagesegmentationtutorial /SourceSAS:"?st=2018-01-16T10%3A40%3A00Z&se=2028-01-17T10%3A40%3A00Z&sp=rl&sv=2017-04-17&sr=c&sig=KeEzmTaFvVo2ptu2GZQqv5mJ8saaPpeNRNPoasRS0RE%3D" /Dest:D:\\pixellevellandclassification /S')
print('Done.')


# ### Install the custom raster function
# 
# We will use Python scripts to apply a trained model to aerial imagery in real-time as the user scrolls through a region of interest in ArcGIS Pro. These Python scripts are surfaced in ArcGIS Pro as a [custom raster function](https://github.com/Esri/raster-functions). The three files needed for the raster function (the main Python script, helper functions for e.g. colorizing the model's results, and an XML description file) must be copied into the ArcGIS Pro subdirectory as follows:
# 
# 1. In Windows Explorer, navigate to `C:\Program Files\ArcGIS\Pro\Resources\Raster\Functions` and create a subdirectory named `Custom`.
# 1. Copy the `ClassifyCNTK` folder in `D:\pixellevellandclassification\arcgispro` into your new folder named `Custom`.
# 
# When this is complete, you should have a folder named `C:\Program Files\ArcGIS\Pro\Resources\Raster\Functions\Custom\ClassifyCNTK` that contains two Python scripts and an XML file.
# 

# ## Evaluate the model in real-time using ArcGIS Pro
# 
# ### Load the sample project in ArcGIS Pro
# 
# Begin by loading the sample ArcGIS Pro project we have provided:
# 
# 1. Search for and launch the ArcGIS Pro program.
#     - If ArcGIS Pro was open, restart it to ensure that all changes above are reflected when the proram loads.
# 1. On the ArcGIS Pro start screen, click on "Open an Existing Project".
# 1. Navigate to the folder where you extracted the sample project, and select the `D:\pixellevellandclassification\arcgispro\sampleproject.aprx` file. Click "OK."
# 
# Once the project has loaded (allow ~30 seconds), you should see a screen split into four quadrants. After a moment, NAIP aerial imagery and ground-truth land use labels should beome visible in the upper-left and upper-right corners, respectively.
# 
# <img src="https://github.com/Azure/pixel_level_land_classification/raw/master/outputs/arcgispro_finished_screenshot.png">
# 

# The bottom quadrants will show the model's best-guess labels (bottom right) and an average of label colors weighted by predicted probability (bottom left, providing an indication of uncertainty). If you wish to use your own trained model, or the bottom quadrants do not populate with results, you may need to add their layers manually using the following steps:
# 

# 1. Begin by selecting the "AI Mixed Probabilities" window at bottom-left.
# 1. Add and modify an aerial imagery layer:
#     1. In the Catalog Pane (accessible from the View menu), click on Portal, then the cloud icon (labeled "All Portal" on hover).
#     1. In the search field, type NAIP.
#     1. Drag and drop the "USA NAIP Imagery: Natural Color" option into the window at bottom-left. You should see a new layer with this name appear in the Contents Pane at left.
#     1. Right-click on "USA NAIP Imagery: Natural Color" in the Contents Pane and select "Properties".
#     1. In the "Processing Templates" tab of the layer properties, change the Processing Template from "Natural Color" to "None," then click OK.
# 1. Add a model predictions layer:
#     1. In the Raster Functions Pane (accessible from the Analysis menu), click on the "Custom" option along the top.
#     1. You should see a "[ClassifyCNTK]" heading in the Custom section. Collapse and re-expand it to reveal an option named "Classify". Click this button to bring up the raster function's options.
#     1. Set the input raster to "USA NAIP Imagery: Natural Color".
#     1. Set the trained model location to `D:\pixellevellandclassification\models\250epochs.model`.
#         - Note: if you trained your own model using our companion notebook, you can use it instead by choosing `D:\pixellevellandclassification\models\trained.model` as the location.
#     1. Set the output type to "Softmax", indicating that each pixel's color will be an average of the class label colors, weighted by their relative probabilities.
#         - Note: selecting "Hardmax" will assign each pixel its most likely label's color insead.
#     1. Click "Create new layer". After a few seconds, the model's predictions should appear in the bottom-left quadrant.
# 1. Repeat these steps with the bottom-right quadrant, selecting "Hardmax" as the output type.
# 

# Now that your project is complete, you can navigate and zoom in any quadrant window to compare ground truth vs. predicted labels throughout the Chesapeake Bay watershed region. If you venture outside the Chesapeake watershed, you may find that ground truth regions are no longer available, but NAIP data and model predictions should still be displayed. 
# 

# ## Next steps
# 
# In this notebook series, we trained and deployed a model on a Geo AI Data Science VM. To improve model accuracy, we recommend training for more epochs on a larger dataset. Please see [our GitHub repository](https://github.com/Azure/pixel_level_land_classification) for more details on scaling up training using Batch AI.
# 
# When you are done using your Geo AI Data Science VM, we recommend that you stop or delete it to prevent further charges.
# 
# For comments and suggestions regarding this notebook, please post a [Git issue](https://github.com/Azure/pixel_level_land_classification/issues/new) or submit a pull request in the [pixel-level land classification repository](https://github.com/Azure/pixel_level_land_classification).
# 

