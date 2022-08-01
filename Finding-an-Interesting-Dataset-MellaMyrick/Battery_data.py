import pandas as pd
import seaborn as sns

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

#raw data
battery_life_data = pd.read_csv("Battery_RUL.csv")
battery_life_data

#Trimmed data
updated_data = battery_life_data[['Cycle_Index', 'Discharge Time (s)', 'Charging time (s)', 'RUL']]
updated_data

# Answering question 1: What is the maximum battery cycle?
max_battery_cycle = battery_life_data['Cycle_Index'].max()
print('The average maximum battery cycle is: ', max_battery_cycle.round(3))

# Answering question 2: What is the average discharge time? 
avg_discharge_time = battery_life_data['Discharge Time (s)'].mean()
print('The average discharge time is: ', avg_discharge_time.round(3))

# Answering question 3: What is the average charge time?
avg_charge_time = battery_life_data['Charging time (s)'].mean()
print('The average charge time is: ', avg_charge_time.round(3))

# Answering question 4: Which variable has a positive correlation to remaining useful life?
sns.pairplot(updated_data)
print("Out of the 4 variables, no variable has a positive correlation to remaining useful life. However, it appears that there is a negative linear correlation between RUL and Cycle Index.")

# Answering question 5: What type of correlation does charging time and discharge time have?
updated_data.plot.scatter(x = 'Charging time (s)', y = 'Discharge Time (s)')
updated_data.plot.scatter(x = 'Discharge Time (s)', y = 'Charging time (s)')
print("No matter how the two are plotted, discharge and charging time have a positve linear correlation but not a strong relationship.")

# Answering question 6: At what cycle does the RUL decrease?
battery_life_data.plot.scatter(x = 'Cycle_Index', y = 'RUL')
print("Based on the scatter plot, the RUL decreases instantaneously at Cycle Index 0. This makes sense because there is no way to reverse the wear that cars experience the moment they turn on. It is just like how most passenger vehicles depreciate in value the moment the owner purchases them and takes them off the lot.")

# Answering question 7: Of the four factors, which has the greatest spread?
Cycle_std = updated_data['Cycle_Index'].std()
print('The standard deviation of the cycle index is: ', Cycle_std.round(3))

Disch_std = updated_data['Discharge Time (s)'].std()
print('The standard deviation of the discharge time is: ', Disch_std.round(3))

Charging_std = updated_data['Charging time (s)'].std()
print('The standard deviation of the charging time is: ', Charging_std.round(3))

RUL_std = updated_data['RUL'].std()
print('The standard deviation of the RUL is: ', RUL_std.round(3))

print("Discharge time appears to have the greatest spread out of the four factors.")

# Answering question 8: What is the average max voltage discharge?
avg_max_volt_discharge = battery_life_data['Max. Voltage Dischar. (V)'].mean()
print('The average max voltage discharge is: ', avg_max_volt_discharge.round(3))

# Answering question 9: What is the average minimun voltage charge?
avg_min_volt_charge = battery_life_data['Min. Voltage Charg. (V)'].mean()
print('The average minimum voltage charge is: ', avg_min_volt_charge.round(3))

# Answering question 10: What type of regression does the data have?
battery_life_data.plot.scatter(x = 'Charging time (s)', y = 'Discharge Time (s)')
battery_life_data.plot.scatter(x = 'Time constant current (s)', y = 'Discharge Time (s)')

print("The two plots appear to have a linear relationship that is not strongly correlated due to a few outliers between 0 and 200000 seconds.")