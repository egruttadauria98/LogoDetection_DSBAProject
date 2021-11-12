import pandas as pd

"""
Merge each _clean_annotations.csv file into a single big dataframe
"""
batches = ["BatchAleC", "BatchAleL", "BatchEline", "BatchElio", "BatchReb"]
splits = ["train", "valid"]

# Initialize a dataframe as the merged csv files of train and valid for the first batch
df = pd.concat([pd.read_csv(f"./data/{batches[0]}/{splits[0]}"), pd.read_csv(f"./data/{batches[0]}/{splits[1]}/_annotations_clean.csv")]]
# Merge remaining batches to the df
for batch in batches[1:]:
    for split in splits:
        df = pd.concat([df, pd.read_csv(f"./data/{batch}/{split}/_annotations_clean.csv")])


"""
Identify groups
"""
logo_groups = {1:["Nike"],
               2:["Adidas", "Starbucks"], 
               3:["MercedesBenz", "NFL"],
               4:["AppleInc", "UnderArmour", "Puma", "TheNorthFace"]}


#print(df.groupby("filename")["class"].count().sort_values(ascending=False))

group5 = df[df["class"].isin(logo_groups[3])]
print(group5["filename"].nunique())
