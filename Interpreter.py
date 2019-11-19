'''
 *
 * Author : - Imanpal Singh <imanpalsingh@gmail.com>
 * Description : -  Emotion Recognition from live camera feed.
 * Technology stack :- Numpy, Pandas, OpenCV, Scikit-Learn
 * Date created : - 11-10-2019
 * Last modified : - 19-11-2019
 * Version : - 0.0.1
 *
'''
'''
 * Chnage log :
    (0.0.1) : Different instruction line for Data extraction and training 

 * File Description : - This file contains interpreter functions.
                        
 *
 
'''

####################### Imports  #############################

# DataHandler 0.1.0
import DataHandler

# Model 0.1.0
import Model

# Deploy 0.0.1
import Deploy




######################## Functions #########################

# Defining Instructon line set
def instruction_line_set():

    instruction_set = ['load data','train model','run','help']
    return instruction_set


# Function to prompt user with different messages based on flags
def message_prompt(code):

    code = code.lower()
    
    if code =='greet':
        
        print("Welcome to Facial Expression Recognition. Type 'Help' for additional information")
        
    elif code =='exit':
        
        print("Program Terminated by the Exit instruction")
        
        return True
    
    return False


def execute(instruction_line):

    # Validating instruction line
    if instruction_line.lower() not in instruction_line_set():
        
        print("<In Function Execute> ERROR : '",instruction_line,"' is not a valid instruction ")


    instruction = instruction_line.lower()

    if instruction == instruction_line_set()[0]:
        
        DataHandler.Apply()
        
    elif instruction == instruction_line_set()[1]:
        
        Model.Apply()
        
    elif instruction == instruction_line_set()[2]:
        
        Deploy.Apply()

    elif instruction == instruction_line_set()[3]:
        
        print('The available commands are')
        
        for instruct in instruction_line_set():
            
            print(instruct + ' ',end=' ')
            
            print()
 


if __name__ == '__main__':

    print(" Warning! This file is not supposed to be running as the main file ")
        
