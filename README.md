# DSA4262_genex

There are 2 python scripts created: [training.py](./script/training.py) and [predicting.py](./script/predicting.py). Below are the input arguments for the 2 scripts and their corresponding name, default value and function.

## **Training Script**
### **Input Argument**:
| Argument name | Required | Default value | Description |
| --- | --- | --- | --- |
| **--json_path** | True | ```../data/data.json ```| Specify the path to input json file |
| **--info_path** | True | ```../data/data.info``` | Specify the path to input info file with labels |
| **--data_out_path** | False | ```../data/parsed_train_data.csv``` | Specify the path to save processed train data |
| **--model_out_path** | False | ```../model/rf_model.sav``` | Specify the path to save trained model |
| **--selected_var_out_path** | False | ```../data/selected_variables.txt``` | Specify the path to save selected variables in text file |


### **Output**:
| File | Directory | Description |
| --- | --- | --- | 
| Parsed Training Data | Specified in input parameter (--data_out_path) | Parsed input json file with feature engineering |
| Selected Variables | Specified in input parameter (--selected_var_out_path) | Text file containing variables selected for the model |
| Fitted model | Specified in input parameter (--model_out_path) | Fitted model saved as pickle file under the directory |
| Fitted LSTM model | [model/](./model/) | Fitted LSTM model for converting sequence to sequence value (will only be trained if it does not exist) |


## **Testing Script**
### **Input Argument**:
| Argument name | Required | Default value | Description |
| --- | --- | --- | --- |
| **--json_path** | True | ```../data/data.json``` | Specify the path to input json file |
| **--model_path** | False | ```../model/rf_model.sav``` | Specify the path to obtain trained model |
| **--selected_variables_path** | False | ```../data/selected_variables.txt``` | Specify the path to obtain selected variables |
| **--data_out_path** | False | ```../data/parsed_prediction_data.csv``` | Specify the path to save processed test data |
| **--prediction_out_path** | True | ```../output/predicted_labels.csv``` | Specify the path to save predicted labels|


### **Output**:
| File | Directory | Description |
| --- | --- | --- | 
| Parsed Prediction Data | Specified in input parameter (--data_out_path) | Parsed input json file with feature engineering |
| Predicted Values | Specified in input parameter (--prediction_out_path) | Predictions on m6a modification by the model specified in input parameter (-–model_path) |
    
   

# Instruction to run the files: 
#### 1) Launch AWS and navigate to your home directory using the command: 
``` sh 
$ cd ~ 
```
#### 2) Create a virtual environment by running the following commands:   
- Install `pip` and `virtualenv`:
``` sh
$ sudo apt-get install python3-pip
$ sudo pip3 install virtualenv
```
- Replace 'name' with a name for your virtual environment:
```sh
$ virtualenv 'name'
```
- Activate your virtual environment 
```sh
$ source 'name'/bin/activate
```
#### 3) Git clone this repository using the command:
```sh
$ git clone https://github.com/joyee00/DSA4262_genex.git
```
#### 4) Navigate to the script directory:
```sh
$ cd script
```
#### 5) Download the python packages required to run the python scripts:
```sh 
$ pip install -r requirements.txt
```
 
&nbsp;
> #### If you would like to run the training script, proceed to **Step 6**. Else, to run the predicting script to make prediction, proceed to **Step 7**. 
&nbsp;

#### 6) To execute training script in the default setting and replace the required path if necessary:
```sh
$ python training.py --json_path '../data/data.json' --info_path '../data/data.info' 
```
- To customise the file path, run:
```sh
$ python training.py --json_path `path_to_json` --info_path `path_to_info` --data_out_path `path_to_save_data` --model_out_path `path_to_save_model` --selected_var_out_path `path_to_save_variables`
```
- The results of the model's training performance will also be printed.

#### 7) To execute the predicting script in the default setting and replace the required path if necessary:
```sh
$ python predicting.py --json_path '../data/data.json' --prediction_out_path '../output/predicted_labels.csv'
```
- To customise the file path, run:
```sh
$ python predicting.py --json_path `path_to_json` --prediction_out_path `path_to_save_prediction` --model_path `path_to_trained_model` --selected_var_out_path `path_to_saved_variables` --data_out_path `path_to_save_data`  
```

- Predicted label will be stored in the [output/](./output/) folder, as specified in the output table above.
