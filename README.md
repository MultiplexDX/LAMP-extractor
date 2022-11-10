Description - LAMP extractor
***************************************************************************************************************************
The provided package contains a Windows-compatible executable with LAMP color determination logic implemented as described 
in "Vivid COVID-19 LAMP: an ultrasensitive, quadruplexed test using LNA-modified primers and a novel zinc ion and 5-Br-PAPS 
colorimetric detection system". The algorithm used in this software matches the one from our automated analysis pipeline 
used to obtain the data presented in the manuscript. For details regarding any restrictions pertaining to this software and 
accompanying code, please see the section "Code Availability" located at the end of the manuscript.

Required operating system: Windows 10 or greater
Install time: none
Tested on platform: Windows-10-10.0.19041-SP0

Running the application 
***************************************************************************************************************************
### Executing .exe file - Default settings
Simply run the application. The application first checks for potentially missing libraries and software dependencies and 
downloads them before analyzing input data.

1. assets\App1.jpg - expected application behavior when running with default settings.

Expected test data analysis time: <10 minutes.

### Executing .exe file - Custom arguments
Custom arguments can be used to use a different input folder than the one specified by default. 

First, open context menu by left-clicking the software folder and then select "Open in terminal".

1. assets\App2.jpg - image shows how to open PowerShell to be able to run the application with arguments.

A custom command in the following format must then be used:
  
```
.\\evaluate -i .\CUSTOM_FOLDER
```

2. assets\App3.jpg - image shows command line with example custom arguments. 

If successful, the application will run, analyze input data from the custom set input folder and generate the
corresponding output files.


Input
***************************************************************************************************************************
The provided input data corresponds to field-use data of Vivid COVID-19 LAMP during National Public Testing in Slovakia.
Every data entry consists of 2 file types for 2 input types (4 files in total):
yyy_type.jpg - JPEG image of analyzed plate with LAMP reaction results.
yyy_type.csv - table representing human operator-classified colors; 3 = positive, 2 = inconclusive, 1 = negative, 0 = empty.

"yyy" - data entry name.
"type" - type of output, sar (SARS-CoV-2) / rna (RNase P).

Unpaired data entries with missing data cannot be analyzed.

Output
***************************************************************************************************************************
All generated files are put in a new folder in the format: "out_" + input_folder_name + timestamp. The following data is 
generated:

cmatrix_xxx__type.jpg - confusion matrix for the dataset (nominal).
cmatrix_n_xxx__type.jpg - confusion matrix for the dataset (frequency).
report_spec_sens_xxx__type.csv - sensitivity/specificity data for the dataset.
report_xxx__type.txt - additional performance data for the dataset.
results/zzz__type.jpg - per plate comparison of machine-classified reaction colors to human operator-classified colors.
out.log - data generation logfile.

"xxx" - name of the dataset taken from input folder name.
"zzz" - name of the input plate from input file name.
"type" - type of output, sar (SARS-CoV-2) / rna (RNase P) / result (combined result).

1. assets\App4.jpg - example output folder contents. 

2. assets\App5.jpg - expected confusion matrix output (nominal) with the whole dataset, SARS-CoV-2. 

3. assets\App6.jpg - expected confusion matrix output (nominal) with only the first 10 plates, SARS-CoV-2. 

If the application is run with the supplied input data with no modifications, the results should precisely match the
results as reported in the manuscript. Running a different subset of results will return different performance
characteristics.

Licenses
***************************************************************************************************************************
```
+---------------------------------+------------------------------------------------+
|             Package             |                      License                   |
+---------------------------------+------------------------------------------------+
|       albumentations 1.1.0      |                        MIT                     |
|        atomicwrites 1.4.0       |                        MIT                     |
|           attrs 21.2.0          |                        MIT                     |
|           build 0.7.0           |                        MIT                     |
|          fastapi 0.63.0         |                        MIT                     |
|          imageio 2.13.1         |                    BSD-2-Clause                |
|         imantics 0.1.12         |                        MIT                     |
|          imutils 0.5.4          |                        MIT                     |
|       lamp-extractor 3.0.1      |     “Commons Clause” License Condition v1.0    |
|           loguru 0.5.3          |                        MIT                     |
|         matplotlib 3.3.3        |                        PSF                     |
|           numpy 1.21.4          |                        BSD                     |
| opencv-python-headless 4.5.1.48 |                        MIT                     |
|           pandas 1.2.3          |                        BSD                     |
|        prettytable 2.4.0        |                   BSD (3 clause)               |
|           pytest 6.2.5          |                        MIT                     |
|        pytest-mock 3.6.1        |                        MIT                     |
|      python-multipart 0.0.5     |                       Apache                   |
|           PyYAML 5.3.1          |                        MIT                     |
|         requests 2.25.1         |                     Apache 2.0                 |
|       scikit-image 0.18.1       |                    Modified BSD                |
|       scikit-learn 0.24.0       |                      new BSD                   |
|           scipy 1.6.0           |                        BSD                     |
|         starlette 0.13.6        |                        BSD                     |
|           toml 0.10.2           |                        MIT                     |
|           torch 1.10.0          |                       BSD-3                    |
|        torchvision 0.11.1       |                        BSD                     |
|           tqdm 4.56.0           |               MPLv2.0, MIT Licences            |
|           wheel 0.37.0          |                        MIT                     |
+---------------------------------+------------------------------------------------+
```
