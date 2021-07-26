 *Template for Your ReadMe File*
 
 A ReadMe file helps reviewers in accessing and navigating through your code. This file will serve as a template for your own ReadMe file. Do fill out the sections carefully, in a clear and concise manner
 
 # Smart Queue Monitoring System
*TODO:* Write a short introduction to your project.
In this course, I have learned about different hardwares and the methods to access these different hardwares in dev-cloud. Each of these hardwares have its own
advantages. By using these hardwares in the Dev cloud, the performance of these hardwares are analysed. A person-detection model is used to identify the perons
working in three different scenarios. In the all the three scenarios, the person is detected using a bounding box and the number of persons are counted. Time
taken to identify the persons in 3 different scenarios are analysed and documented.

## Proposal Submission
*TODO* Specify the type of hardware you have chosen for each sector 
Mention the type of hardware you have chosen for each of the scenarios:
- Manufacturing: CPU+ FPGA
- Retail: CPU+ IN GPU
- Transportation:CPU + VPU 

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
# Project structure
The working directory is ./starter_files
The model is downloaded and kept inside ./starter_files/intel
For running the model, person_detect.py files are created for 3 different scenario's and it is added in the jupyter notebook itself.
Install all the dependencies like openvino, Hadoop, spark, etc. To test the model successfully.
The out for each of the scenarios are available in the output folder
# How to run the project
All the required packages needs to be installed and then run jupyter notebook
Command line arguments are available in tje jupyter notebook



## Documentation
*TODO* In this section, you can document your project code. You can mention the scripts you ran for your project. This could also include scripts for running your project and the scripts you ran on Devcloud too. Make sure to be as informative as possible!

The project was ran on a jupyter notebook. For the 3 different scenarios, 3 person_detect.py files are there inside the folder. 
Also you can find 3 different script files i.e. bash-'.sh' files for 3 scenarios are present inside the folder.
The commands used to run the files in the dev-cloud, commands used to run different hardwares are given in the jupyter files.

## Results
*TODO* In this section you can write down all the results you got. You can go ahead and fill the table below.

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                   1024                       |         26.5               |        FP 32            |
| GPU              |                    876                       |         23.6               |        FP 16            |
| FPGA             |                    415                       |         3.9                |        FP 16            |
| VPU              |                    350                       |         2.3 secs           |        FP 16            |

## Conclusions
*TODO* In this section, you can give an explanation for your results. You can also attach the merits and demerits observed for each hardware in this section.
The VPU NCS2 plugin has the minimum time required for loading and inference. After that FPGA loads in 4 secs. The GPU and the CPU takes more time to load and infer the model as they are not really computing efficiently. Both CPU and CPU + GPU takes more time than others to load and infer the model.

## Stand Out Suggestions
*TODO* If you have implemented the standout suggestions, this is the section where you can document what you have done.
- Fallback Policy: Use OpenVINO's Hetero plugin to use a fallback device like CPU when the main prediction device fails. Write about the changes you had to make in the code to implement these changes. More information regarding Fallback Policy can be found [here](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HETERO.html)
- MultiDevice Plugin: Use the multi device plugin available as a part of the OpenVINO toolkit. Write about the changes that you had to make in your code to implement this.
