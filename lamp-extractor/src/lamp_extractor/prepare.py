# %%
import pandas as pd
from pathlib import Path
from lamp_extractor.utils import load_yaml
import numpy as np
import shutil
from tqdm import tqdm

rna_errors = []
sars_errors = []
config = load_yaml(r"/home/developer/ml/lamp-extractor/src/lamp_extractor/apis/rest/config.yaml")
ROWS = config['rows']
COLUMNS = config['columns']
CLASSNAMES = list(config['classes'].keys())

coldex_by_value = {col:index for index, col in enumerate(COLUMNS)} 
rowdex_by_value = {row:index for index, row in enumerate(ROWS)} 
cladex_by_value = {cla:index for index, cla in enumerate(CLASSNAMES)} 

print(cladex_by_value)
print(rowdex_by_value)
print(coldex_by_value)

DATA_FOLDER_PATH = r"/home/developer/ml/lamp-extractor/data/national_testing_dataset/data"
OUTPUT_FOLDER_PATH = Path(r"/home/developer/ml/lamp-extractor/data/national_testing_dataset_prepared")
OUTPUT_FOLDER_PATH.mkdir(exist_ok=True)

dtype= {
    "server":"str",
    "eventcode":"str",
    "rackid":"str"
}

lamp_test_df = pd.read_csv(
    r"/home/developer/ml/lamp-extractor/data/national_testing_dataset/db/lamp_test.csv", 
    delimiter=";",
    dtype=dtype,
    index_col=["server", "eventcode", "rackid"]
)
lamp_test_df.dropna(subset = ["sarscov2humanresult", "rnaintegrityhumanresult", "humanresult"], inplace=True)
#lamp_test_df = lamp_test_df.replace('',np.nan)
#lamp_test_df = lamp_test_df.dropna(axis="rows", how="any")
lamp_test_df.info()
# %%
lamp_test_df['sarscov2humanresult'].value_counts()
# %%
lamp_test_df['rnaintegrityhumanresult'].value_counts()
# %%
rack_df = pd.read_csv(
    r"/home/developer/ml/lamp-extractor/data/national_testing_dataset/db/rack.csv", 
    delimiter=";",
    dtype=dtype,
    index_col=["server", "eventcode", "rackid"],
)
rack_df.info()
# %%
lamp_test_df[lamp_test_df['rnaintegrityhumanresult']]
# %%
# %%
lamp_test_df['coldex'] = lamp_test_df['tubeposition'].str.replace(r"[A-H]", "").apply(lambda x: coldex_by_value[x]).astype(int)
lamp_test_df['rowdex'] = lamp_test_df['tubeposition'].str.replace(r"[0-9]", "").apply(lambda x: rowdex_by_value[x]).astype(int)
lamp_test_df['indices'] = len(ROWS) * lamp_test_df['rowdex'] + lamp_test_df['coldex'] 
lamp_test_df['rnalabel'] = lamp_test_df['rnaintegrityhumanresult'].apply(lambda x: cladex_by_value[x])
lamp_test_df['rnalabel'].fillna(value=-1, inplace=True)
lamp_test_df['rnalabel'] = lamp_test_df['rnaintegrityhumanresult'].apply(lambda x: cladex_by_value[x])
lamp_test_df['sarslabel'] = lamp_test_df['sarscov2humanresult'].apply(lambda x: cladex_by_value[x])
lamp_test_df['sarslabel'].fillna(value=-1, inplace=True)
# %%
for rindex, rrow in rack_df.iterrows():
    # Create matrices
    rna_label = np.zeros((len(ROWS), len(COLUMNS)))
    sars_label = np.zeros((len(ROWS), len(COLUMNS)))
    # Fulfill matrices
    if rindex not in lamp_test_df.index:
        print("Missing: ", rindex)
        continue
    
    for index, row in lamp_test_df.loc[rindex].iterrows():
        rna_label[row['rowdex'], row['coldex']] = row['rnalabel']
        sars_label[row['rowdex'], row['coldex']] = row['sarslabel']
    
    rnaintegrityimage = rack_df.at[rindex, 'rnaintegrityimage']
    print(rnaintegrityimage)
    if rnaintegrityimage is not np.nan:
        rna_img_name = Path(rnaintegrityimage).with_suffix(".jpg")
        rna_path = Path(DATA_FOLDER_PATH) / "images" / rna_img_name
        assert rna_path.exists(), rna_path
        shutil.copy(src=rna_path, dst=OUTPUT_FOLDER_PATH)
        np.savetxt((OUTPUT_FOLDER_PATH / rna_path.name).with_suffix(".csv"), rna_label, fmt="%d", delimiter=",")
        print((OUTPUT_FOLDER_PATH / rna_path.name).with_suffix(".csv"))

    sarscov2image = rack_df.at[rindex, 'sarscov2image']
    print(sarscov2image)
    if sarscov2image is not np.nan:
        sars_img_name = Path(sarscov2image).with_suffix(".jpg")
        sars_path = Path(DATA_FOLDER_PATH) / "images" / sars_img_name
        assert sars_path.exists(), sars_path 
        shutil.copy(src=sars_path, dst=OUTPUT_FOLDER_PATH)
        np.savetxt((OUTPUT_FOLDER_PATH / sars_path.name).with_suffix(".csv"), sars_label, fmt="%d", delimiter=",")
# %%
