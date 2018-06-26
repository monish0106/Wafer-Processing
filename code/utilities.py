import pandas as pd
from pandas import Series
from matplotlib import pyplot as plt 

class Utilities:
    
    def read_dataset_one(dir_name,filename,num_examples):
        
#         The filename is of the format 000_Et_H_CO_H 
#         where 7th position tells us whether Ethylene is present or not (instead of binary classification we can make 4 classes which tells us the proportion of Ethylene in the mixture - High, Medium, Low,Not-Present)         
#         Same goes with CO and Me        
#         class_1 -> Ethylene 
#         class_2 -> Carbon-Monoxide
#         class_3 -> Methane 
        if filename[7] == 'n':
            class_1 = 'N'
        else:
            class_1 = 'Y'
        
        if filename[9] == 'C':
            class_2 = filename[9:]
            class_3 = 'Me_n'
        else:
            class_2 = 'CO_n'
            class_3 = filename[9:]
        
        name = dir_name + "/" + filename
        data = pd.read_csv(name,header=None)
        
#         plt.scatter(data[0], data[1])
#         plt.show()
        
        del data[0]
        return [data.values[0:num_examples],class_1,class_2,class_3]
    
    def read_dataset_one_avg(dir_name,filename):

    #         The filename is of the format 000_Et_H_CO_H 
    #         where 7th position tells us whether Ethylene is present or not (instead of binary classification we can make 4 classes which tells us the proportion of Ethylene in the mixture - High, Medium, Low,Not-Present)         
    #         Same goes with CO and Me        
    #         class_1 -> Ethylene 
    #         class_2 -> Carbon-Monoxide
    #         class_3 -> Methane 
            if filename[7] == 'n':
                class_1 = 'N'
            else:
                class_1 = 'Y'

            if filename[9] == 'C':
                class_2 = filename[9:]
                class_3 = 'Me_n'
            else:
                class_2 = 'CO_n'
                class_3 = filename[9:]

            df = pd.DataFrame()
            print(df)
            
            
            name = dir_name + "/" + filename
            data = pd.read_csv(name,header=None)

            for row in data:
                print(data[row])
    #         plt.scatter(data[0], data[1])
    #         plt.show()

            del data[0]
            normalized_data=(data-data.mean())/data.std()
            return [normalized_data.values[0:num_examples],class_1,class_2,class_3]
