import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == "__main__":
    ### 20230211 -> progress: data cleaning process
    MainPath= os.getcwd()
    FileName= "Battery_TL_three_batches.csv"

    with open(os.path.join(MainPath, FileName), "r") as file:
        Data= pd.read_csv(open(os.path.join(MainPath, FileName)))
    
    ColumnsName= list(Data.columns)
    ColumnsName[0]= "Cycle"
    ColumnsName[9], ColumnsName[11], ColumnsName[13]= "#4_SOH", "#5_SOH", "#6_SOH"
    ColumnsName[16], ColumnsName[18], ColumnsName[20]= "#7_SOH", "#8_SOH", "#9_SOH"
    Data.columns= ColumnsName
    
    ChargingPolicy1= Data[["Cycle", "Battery#1", "#1_SOH", "Battery#2", 
    "#2_SOH", "Battery#3", "#3_SOH"]]

    ChargingPolicy2= Data[["Cycle", "Battery#4", "#4_SOH", "Battery#5", 
    "#5_SOH", "Battery#6", "#6_SOH"]]

    ChargingPolicy3= Data[["Cycle", "Battery#7", "#7_SOH", "Battery#8", 
    "#8_SOH", "Battery#9", "#9_SOH"]]

    ChargingPolicy1.to_csv("ChargingPolicy1.csv", index= 0)
    ChargingPolicy2.to_csv("ChargingPolicy2.csv", index= 0)
    ChargingPolicy3.to_csv("ChargingPolicy3.csv", index= 0)