from dtype_cleanup import *
import pandas as pd
import re

if __name__ == '__main__':
    print('########## Data Type Cleanup ##########')
    df = pd.read_csv("C:\\Users\\torna\\JupyterLap\\DS_Handbook\\file.tsv", sep="|")
    df["First Name"] = df["Name"].apply(lambda s: get_first_last_name(s)[0])
    df["Last Name"] = df["Name"].apply(lambda s: get_first_last_name(s)[1])
    df["Age"] = df["Age"].apply(format_age)
    df["Birthdate"] = df["Birthdate"].apply(format_date)
    print(df)
    print()

    print('########## Regular expression ##########')
    