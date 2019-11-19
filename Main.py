'''
 *
 * Author : - Imanpal Singh <imanpalsingh@gmail.com>
 * Description : -  Emotion Recognition from live camera feed.
 * Technology stack :- Numpy, Pandas, OpenCV, Scikit-Learn
 * Date created : - 11-10-2019
 * Last modified : - 19-11-2019
 * Version : - 0.2.1
 *
'''

'''
 *
 * Change log :

   (0.0.1) : - UI : Driver program to extract dataset
   (0.0.2) : - UI : Files are distributed based on steps
   (0.2.1) : - UI : Now UI accepts commands instead of manually running files
 * File Description : - This is the driver File which accepts commands/instructions
                        and executes commands based on the instructions.
                        Instruction set can be viewed by using 'help' command
 *
 
'''

################################## IMPORTS ###################################

# Interpreter (0.0.1)
import Interpreter

#############################################################################


if __name__ == '__main__':

    # Welcome message
    Interpreter.message_prompt('Greet')
    
    while True:

        #Acception instruction line
        instruction_line = input('>> ')

        #Checking for Flag inputs
        check_flag = Interpreter.message_prompt(instruction_line)
        if check_flag == True:
            break;

        # Validating and executing instruction
        Interpreter.execute(instruction_line)
        

        
         


   
