import pandas

pandas.options.display.max_columns = None
pandas.options.display.max_rows = None

battery_life_data = pandas.read_csv("Battery_RUL.csv")

print(battery_life_data)