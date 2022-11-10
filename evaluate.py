import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
import argparse

# Saving outputs based on executable
exec_path = Path(sys.executable).resolve()
if exec_path.name == "python.exe":
    root = Path(__file__).parent.resolve()
else:
    root = exec_path.parent

parser = argparse.ArgumentParser(prog = 'Lamp Extractor evaluation', description = 'Running application for results evaluation')
parser.add_argument('-i', '--input_dir', default=root / "b02_lnt_02102022-03112022", help='Make sure that your dataset is in following format: <file1>_rna.jpg, <file1>_rna.csv')

args = parser.parse_args()
data_dir = Path(args.input_dir)

assert data_dir.exists(), f"{data_dir} Not found, please check if you provided correct path to directory"
assert data_dir.is_dir(), f"{data_dir} is not a directory"

outtime = f"out_{data_dir.name}{'{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())}"
output = root / outtime
output.mkdir(exist_ok=False)

logger.remove(0)
logger.add(output / "file.log", level="TRACE", rotation="100 MB")


print(f"======================================================================================================")
print(f"*****     Running executable: {exec_path}         ")
print(f"*****     Input dir: {data_dir}                   ")
print(f"*****     Output dir: {output}   ")
print(f"======================================================================================================")

# %%
import numpy as np
import cv2
from lamp_extractor import main, utils
from tqdm import tqdm
from pkg_resources import resource_filename
#import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import pandas as pd
from matplotlib import pyplot as plt

wrongs = []
y_trues, y_preds = [], []
results = []
#type = "" # all
#type = "_sars"
#type = "_rna"
config = utils.load_yaml(resource_filename("lamp_extractor", f"apis/rest/config.yaml"))
classes = np.array(config['classes'])

def get_keytype(filepath):
    type = filepath.stem.split("_")[-1]
    key = "_".join(filepath.stem.split("_")[:3])
    return key, type

pairs = {}
for filepath in list(data_dir.glob(f"*.csv")):
    key, type = get_keytype(filepath)
    item = (type, filepath)
    if key not in pairs.keys():
         pairs[key] = []
    pairs[key].append(item)
    
# %%
print(f"\nNumber of plates pairs in input folder: {len(pairs)} (SarsCov2, RNasP)")
print("======================================================================================================")
# %%
print("\nAnalysing plates....")
dfs = []
t_matrices, p_matrices, paths = [], [], []
for key, value in tqdm(list(pairs.items())):
    # Predict rna
    df = pd.DataFrame()
    for type, filepath in value:
        t_matrix = np.loadtxt(filepath, delimiter=",", dtype=int)
        t_matrices.append(t_matrix)
        img = cv2.imread(filepath.with_suffix(".jpg").as_posix())
        p_matrix, classes, ROWS, COLUMNS, warped_img = main.predict(img, config)
        paths.append(filepath.with_suffix(".jpg"))
        p_matrices.append(p_matrix)
        df[f"machine_{type}"] = p_matrix.reshape(-1)
        df[f"human_{type}"] = t_matrix.reshape(-1)
        df[f"file_{type}"] = filepath
    df["id"] = key
    
    dfs.append(df)
    
df = pd.concat(dfs)
# %%
def print_stats(df, type, text):
    print(f"\n=========================={text}==========================\n")
    col1 = f"human{type}"
    hdf = df.rename({col1: "Human identified"}, axis=1)
    hdf = hdf["Human identified"].value_counts()
    hdf.index.name = "classes"
    print(hdf.to_markdown())
    print("")
    col1 = f"machine{type}"
    hdf = df.rename({col1: "System indetified"}, axis=1)
    hdf = hdf["System indetified"].value_counts()
    hdf.index.name = "classes"
    print(hdf.to_markdown())
# %% "Calculating aggregated value based on logic..."
hrna = "human_rna"
hsar = "human_sars"
mrna = "machine_rna"
msar = "machine_sars"
pos = "POSITIVE"
neg = "NEGATIVE"
inc = "INCONCLUSIVE"
emp = "EMPTY"
# %%
df.loc[:, [hrna, mrna, hsar, msar]] = df.loc[:, [hrna, mrna, hsar, msar]].replace({i:c for i, c in enumerate(classes)})
print_stats(df, "_rna", "RNasP samples stats")
print_stats(df, "_sars", "SarsCov2 samples stats")

# %%
POSITIVE = (
    (df[hrna] == pos) & (df[hsar] == pos)
    | (df[hrna] == neg) & (df[hsar] == pos)
    | (df[hrna] == inc) & (df[hsar] == pos)
    | (df[hrna].isna()) & (df[hsar] == pos)
)
INCONCLUSIVE = (
    (df[hrna] == pos) & (df[hsar] == inc)
    | (df[hrna] == neg) & (df[hsar] == neg)
    | (df[hrna] == neg) & (df[hsar] == inc)
    | (df[hrna] == inc) & (df[hsar] == neg)
    | (df[hrna] == inc) & (df[hsar] == inc)
)
NEGATIVE = ((df[hrna] == pos) & (df[hsar] == neg))
EMPTY = ((df[hrna] == emp) | (df[hsar] == emp))

df['human'] = "EMPTY"
df.loc[NEGATIVE, ['human']] = neg
df.loc[POSITIVE, ['human']] = pos
df.loc[INCONCLUSIVE, ['human']] = inc
# %%
POSITIVE = (
    (df[mrna] == pos) & (df[msar] == pos)
    | (df[mrna] == neg) & (df[msar] == pos)
    | (df[mrna] == inc) & (df[msar] == pos)
    | (df[mrna].isna()) & (df[msar] == pos)
)

INCONCLUSIVE = (
    (df[mrna] == pos) & (df[msar] == inc)
    | (df[mrna] == neg) & (df[msar] == neg)
    | (df[mrna] == neg) & (df[msar] == inc)
    | (df[mrna] == inc) & (df[msar] == neg)
    | (df[mrna] == inc) & (df[msar] == inc)
)
NEGATIVE = ((df[mrna] == pos) & (df[msar] == neg))

df['machine'] = "EMPTY"
df.loc[NEGATIVE, ['machine']] = neg
df.loc[POSITIVE, ['machine']] = pos
df.loc[INCONCLUSIVE, ['machine']] = inc

# %%
print_stats(df, "", "Final result samples stats")
# %%
def spec_sens_report(y_true, y_pred, class_names, filepath=None):
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names, normalize=None)
    df = get_sensitiviy_specificity_report(class_names, conf_matrix)
    df.set_index("class").round(3).to_csv(filepath)

def get_sensitiviy_specificity_report(class_names, conf_matrix):
    specificities = []
    sensitivities = []
    totals = []
    for pi, p in enumerate(class_names):
        others = []
        for oi, o in enumerate(class_names):
            if o == p:
                continue
            others.append((oi, o))
        
        tp = conf_matrix[pi, pi]
        fp = sum([conf_matrix[oi, pi] for (oi, o) in others])
        tn = sum([conf_matrix[oi, oi] for (oi, o) in others])
        fn = sum([conf_matrix[pi, oi] for (oi, o) in others])
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        total = sum([conf_matrix[pi, i] for i, _ in enumerate(class_names)])
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        totals.append(total)
        
    df = pd.DataFrame({
        "class":class_names, 
        "sensitivity": sensitivities, 
        "specificity": specificities,
        "samples": totals
    })
    
    return df


def show_confusion_matrix(y_true, y_pred, class_names, filepath=None, norm=False):
    plt.clf()
    #figure = plt.figure(figsize=(20, 20))
    print(f"\n=========================={filepath}==========================\n")
    if norm:
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names, normalize='true')
        con_mat_df = pd.DataFrame(conf_matrix,index=class_names,columns=class_names)
        con_mat_df.index.name = "Human \ System"
        #con_mat_df = pd.crosstab(df['human'], df['machine'], rownames=['Human'], colnames=['System'])
        print(con_mat_df.to_markdown())
    else:
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_names, normalize=None)
        con_mat_df = pd.DataFrame(conf_matrix,index=class_names,columns=class_names)
        con_mat_df.index.name = "Human \ System"
        #con_mat_df = pd.crosstab(y_true, y_pred, rownames=['Human'], colnames=['System'])
        print(con_mat_df.to_markdown())
        # sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt="d", linewidths=.5)
    #plt.frame_on(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.ylabel('Human operator')
    plt.xlabel('System')
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, pad_inches=5)
        
# y_preds_f = [classes[ci] for ci in np.array(y_preds).reshape(-1)]
# y_trues_f = [classes[ci] for ci in np.array(y_trues).reshape(-1)]

def report(df, truecol, predcol, event, label, classes):
    tmp_df = df.loc[df[truecol].isin(classes)]
    y_trues = tmp_df[truecol].values.astype(str)
    y_preds = tmp_df[predcol].values.astype(str)
    #print(f"{y_trues[:3]=}, {y_preds[:3]=}")
    show_confusion_matrix(y_trues, y_preds, classes, filepath=output / f"cmatrix_{event}_{label}.jpg", norm=False)
    show_confusion_matrix(y_trues, y_preds, classes, filepath=output / f"cmatrix_n_{event}_{label}.jpg", norm=True)
    spec_sens_report(y_trues, y_preds, classes, filepath=output / f"report_spec_sens_{event}_{label}.csv")
    with open(output / f"report_{event}_{label}.txt", "w") as f:
            f.write(classification_report(y_trues, y_preds, labels=classes))

thr = ""
iclasses = [c for c in classes if c != 'EMPTY']
report(df, 'human', 'machine', "national_testing", f'{thr}_result', iclasses)
report(df, hrna, mrna, "national_testing", f'{thr}_rna', iclasses)
report(df, hsar, msar, "national_testing", f'{thr}_sar', iclasses)

vis_dir = output / "results"
vis_dir.mkdir()

print(f"Generating visualizations to {vis_dir}...")
for tm, pm, ipath in tqdm(zip(t_matrices, p_matrices, paths)):
    #print(i, acc, ipath, cpath)
    ti = utils.colorize_matrix(tm)
    pi = utils.colorize_matrix(pm)
    
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig = plt.figure(figsize=(24,6))
    
    ax[0] = fig.add_subplot(131)
    ax[0].set_frame_on(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].imshow(cv2.cvtColor(cv2.imread(ipath.as_posix(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    
    ax[1] = fig.add_subplot(132)
    ax[1].set_title("Human")
    ax[1].set_xticks(range(tm.shape[1]), config['columns'])
    ax[1].set_yticks(range(tm.shape[0]), config['rows'])
    ax[1].set_xticklabels(config['columns'])
    ax[1].set_yticklabels(config['rows'])
    ax[1].imshow(ti)
    
    
    ax[2] = fig.add_subplot(133)
    ax[2].imshow(pi)
    ax[2].set_title("Machine")
    ax[2].set_xticks(range(pm.shape[1]))
    ax[2].set_yticks(range(pm.shape[0]), config['rows'])
    ax[2].set_xticklabels(config['columns'])
    ax[2].set_yticklabels(config['rows'])
    # ax[2].grid(visible=True, which='major')
    plt.tight_layout()

    plt.savefig(vis_dir / ipath.name)
    plt.close('all')
# %%
print("Completed...") 
print(f"Outputs can be found: {output}")
