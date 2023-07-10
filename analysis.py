import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('DIAGNOSES_ICD.csv')
    d = {}
    
    for index, row in data.iterrows():
        cur_val = d.get(row['ICD9_CODE'])
        if cur_val == None:
            d[row['ICD9_CODE']] = 1
        else:
            d[row['ICD9_CODE']] += 1
    
    sorted_d = sorted(d.items(), key=lambda x:x[1])
    print(sorted_d)