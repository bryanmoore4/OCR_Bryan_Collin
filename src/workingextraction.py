from glob import glob
from PIL import Image as  Image
from os import listdir
import os as os
from scipy import misc
import numpy as np
import time

start_time =time.clock()

def extractfeatures():
    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((26, 1))
        e[j] = 1.0
        return e
        
    
    def rgb2gray(rgb):
    
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
        return gray
        
    def convpixels(images):
        images[:] = [np.abs(x - 255.0)/255.0 for x in images]

    
    def extract(filename):
        #Comments for Bryan
        #put it as '*.png' to do all the files, and just filename if you wanna work on only one file
        
    #    letters = glob('*.png')
        letters = glob('*.png')
        
        possible = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        training = []
        test = []
        validate = []
        
        for i,char in enumerate(letters):
        	name = char.split('__')[0].split('_')[-1]
        	character = char.split('__')[1].split('.')[0] 
        	image = misc.imread(filename,'flatten')
    #    	need to make it gray scale where 0 = white, 1 = black, and everything in between is a shade of grey
    #    	convpixels(image)
        	#a = (255 - a4[:,:,0])/255. <--- This is for converting to binary. need to change a4 to a4 = array(im.open(char)) and then convert
        	#Each list is appending (array, index), so a complete list will be [(array,index),(array,index)...etc] and when you take the shape of the array of that list, it gives (104,2), meaning that there are 104 items with 2 components, but each x component of those items is a (125,100) array. 
    
        	if name == 'torsey' or name =='siskar' or name =='salem' or name =='effland': training.append((image,possible.index(character)))
        	if name == 'martinez': test.append((image,possible.index(character)))
        	if name == 'ward': validate.append((image,possible.index(character)))
            
             
         
        #Also, in the future, i'm gonna customize this process. Instead of hardcoding the names in and what type of data we want it to be, i'll have it be input options when you run the script, so you can assign what people you want to go into which category. You know, cause i'm slick wit it.
        training_array = np.array(training)
        test_array = np.array(test)
        validate_array = np.array(validate)
        return(training_array,validate_array,test_array)
        #    file_name = gzip.open'(Training.pkl.gz','wb')
        #    file_name2 = gzip.open('Testing.pkl.gz','wb')
        #    file_name3 = gzip.open('Validating.pkl.gz','wb')
        #    pickle.dump(training_array,file_name)
        #    pickle.dump(test_array,file_name2)
        #    pickle.dump(validate_array,file_name3)
        #file_name.close()
        
    abspath2 = os.getcwd()
    abspath = os.path.abspath('../data/Byrie')
    os.chdir(abspath)
    
    for f in os.listdir(abspath):
        tr_d,va_d,te_d = extract(f)
    
    training_inputs,training_results = (zip(*tr_d))
    training_inputs = np.array(training_inputs)
    training_inputs = [np.reshape(x, (12500,1)) for x in training_inputs]
    training_results = np.array(training_results)
    training_results = [vectorized_result(y) for y in training_results]
    training_data = zip(training_inputs,training_results)

    validation_inputs,validation_results = (zip(*va_d))
    validation_inputs = np.array(validation_inputs)
    validation_inputs = [np.reshape(x, (12500,1)) for x in validation_inputs]
    validation_data = zip(validation_inputs,validation_results)
    
    test_inputs,test_results = (zip(*te_d))
    test_inputs = np.array(test_inputs)
    test_inputs = [np.reshape(x, (12500,1)) for x in test_inputs]
    test_data = zip(test_inputs,test_results)
    
    os.chdir(abspath2)
    return(training_data,validation_data,test_data)

training_data, validation_data, test_data = extractfeatures()
print time.clock() - start_time, "seconds"