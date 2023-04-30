import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
plt.rcParams["font.family"] = "Palatino Linotype"

if __name__== "__main__":
    file_path_ls= ["ChargingPolicy1.csv", "ChargingPolicy2.csv"]

    with open(os.path.join(os.getcwd(), file_path_ls[0]), "r") as file:
        charging_policy_1= pd.read_csv(file)

    with open(os.path.join(os.getcwd(), file_path_ls[1]), "r") as file:
        charging_policy_2= pd.read_csv(file)

    battery_ls= ["Battery#1", "Battery#2", "Battery#3", "Battery#4", "Battery#5", "Battery#6"]
    line_style= ["--", "solid"]
    plt.figure(figsize= (10, 4))
    for i in range(0, len(battery_ls)):
        if battery_ls[i].split("#")[-1] in ["1", "2", "3"]:
            if battery_ls[i].split("#")[-1]== "1":
                plt.plot(np.arange(0, len(charging_policy_1[battery_ls[i]])), charging_policy_1[battery_ls[i]], ls= line_style[0], color= "blue", label= "Battery#7")
            elif battery_ls[i].split("#")[-1]== "2":
                plt.plot(np.arange(0, len(charging_policy_1[battery_ls[i]])), charging_policy_1[battery_ls[i]], ls= line_style[0], color= "orange", label="Battery#8")
            else:
                plt.plot(np.arange(0, len(charging_policy_1[battery_ls[i]])), charging_policy_1[battery_ls[i]], ls= line_style[0], color= "green", label= "Battery#9")
        elif battery_ls[i].split("#")[-1] in ["4", "5", "6"]:
            if battery_ls[i].split("#")[-1]== "4":
                plt.plot(np.arange(0, len(charging_policy_2[battery_ls[i]])), charging_policy_2[battery_ls[i]], ls= line_style[1], color= "blue", label= battery_ls[i])
            elif battery_ls[i].split("#")[-1]== "5":
                plt.plot(np.arange(0, len(charging_policy_2[battery_ls[i]])), charging_policy_2[battery_ls[i]], ls= line_style[1], color= "orange", label= battery_ls[i])
            else:
                plt.plot(np.arange(0, len(charging_policy_2[battery_ls[i]])), charging_policy_2[battery_ls[i]], ls= line_style[1], color= "green", label= battery_ls[i])
    plt.legend(loc= "best", fontsize= 16)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.xlabel("Cycle", fontsize= 20)
    plt.ylabel("Discharge Capacity (mAh)", fontsize= 20)
    plt.grid(True)
    plt.show()
                
    
    
            
