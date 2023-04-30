import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams["font.family"] = "Palatino Linotype"

if __name__== "__main__":
    path= os.path.join(os.getcwd(), "ChargingPolicy1.csv")
    with open(path, "r") as file:
        data= pd.read_csv(file)

    battery_7= data[["Cycle", "#1_SOH"]].dropna()

    sp_index= np.where(np.array(battery_7["#1_SOH"])<= 0.888)[0][0]
    print("sp: ", sp_index)

    plt.figure(num= 1, figsize= (10, 4))
    plt.plot(battery_7["Cycle"].iloc[: sp_index], battery_7["#1_SOH"].iloc[: sp_index], color= "darkblue")
    plt.hlines(0.8, battery_7["Cycle"].iloc[0], battery_7["Cycle"].iloc[-1]+ 200, color= "red", ls= "--")
    plt.ylabel("SOH", fontsize= 20)
    plt.xlabel("Cycle", fontsize= 20)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.xlim(battery_7["Cycle"].iloc[0], battery_7["Cycle"].iloc[-1]+ 200)
    plt.ylim(0.75, 1.00)
    plt.grid(True, ls= "--")
    plt.show()
    plt.tight_layout()