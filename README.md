 # Smart Queue Monitoring System
In this course, I have learned about different hardwares and the methods to access these different hardwares in dev-cloud. Each of these hardwares have its own
advantages. By using these hardwares in the Dev cloud, the performance of these hardwares are analysed. A person-detection model is used to identify the perons
working in three different scenarios. In the all the three scenarios, the person is detected using a bounding box and the number of persons are counted. Time
taken to identify the persons in 3 different scenarios are analysed and documented.

## Proposal Submission

Mention the type of hardware you have chosen for each of the scenarios:
- Manufacturing: CPU+ FPGA
- Retail: CPU+ IN GPU
- Transportation:CPU + VPU 

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

The project was ran on a jupyter notebook. For the 3 different scenarios, 3 person_detect.py files are there inside the folder. 
Also you can find 3 different script files i.e. bash-'.sh' files for 3 scenarios are present inside the folder.
The commands used to run the files in the dev-cloud, commands used to run different hardwares are given in the jupyter files.

## Results

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                   1024                       |         26.5               |        FP 32            |
| GPU              |                    876                       |         23.6               |        FP 16            |
| FPGA             |                    415                       |         3.9                |        FP 16            |
| VPU              |                    350                       |         2.3 secs           |        FP 16            |

## Conclusions
The VPU NCS2 plugin has the minimum time required for loading and inference. After that FPGA loads in 4 secs. The GPU and the CPU takes more time to load and infer the model as they are not really computing efficiently. Both CPU and CPU + GPU takes more time than others to load and infer the model.

## Stand Out Suggestions
- Fallback Policy: Use OpenVINO's Hetero plugin to use a fallback device like CPU when the main prediction device fails. Write about the changes you had to make in the code to implement these changes. More information regarding Fallback Policy can be found [here](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HETERO.html)
- MultiDevice Plugin: Use the multi device plugin available as a part of the OpenVINO toolkit. Write about the changes that you had to make in your code to implement this.
