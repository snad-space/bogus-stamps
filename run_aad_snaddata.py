import numpy as np
import pandas as pd
from coniferest.pineforest import PineForest
import glob,os


np.random.seed(42)

'''
files=[]
for file in glob.glob("data/oid_*"):
   files.append(file)

field_names = []
for j in range(0,len(files)):
    name1  = files[j].rsplit("_",1)[1]
    name2  = name1.rsplit(".",1)[0]
    field_names.append(int(name2))
'''

field = 837
#683 has a bunch of Miras
oid = np.memmap('data/oid_'+str(field)+'.dat', mode='r', dtype=np.uint64)
with open('data/feature_'+str(field)+'.name') as f:
    names = f.read().split()
dtype = [(name, np.float32) for name in names]
feature = np.memmap('data/feature_'+str(field)+'.dat', mode='r', dtype=dtype, shape=oid.shape)
metadata = oid

data = pd.DataFrame(feature)

model = PineForest(
    # Number of trees to use for predictions
    n_trees=256,
    # Number of new tree to grow for each decision
    n_spare_trees=768,
    # Fix random seed for reproducibility
    random_seed=0)

from coniferest.session import Session
from coniferest.label import Label
from coniferest.session.callback import (
    TerminateAfter, viewer_decision_callback,
)

session = Session(
    data=data,
    metadata=metadata,
    model=model,
    # Prompt for a decision and open object's page on the SNAD Viewer
    decision_callback=viewer_decision_callback,
    on_decision_callbacks=[
        # Terminate session after 10 decisions
        TerminateAfter(50),
    ],
)
session.run()

from pprint import pprint

print('Decisions:')
pprint({metadata[idx]: label.name for idx, label in session.known_labels.items()})
print('Final scores:')
pprint({metadata[idx]: session.scores[idx] for idx in session.known_labels})



labels = {metadata[idx]: int(label == Label.ANOMALY) for idx, label in session.known_labels.items()}
scored = {metadata[idx]: session.scores[idx] for idx in session.known_labels}

#m = m.reset_index() -- to make index a column

inds = pd.Series(metadata)
a_scores = session.scores
sc = pd.Series(a_scores)
all_df = pd.DataFrame({'oid':inds,'scores':sc})
all_df.to_csv('results_pf/allscores_'+str(field)+'.csv',index=False)

df_lab = pd.DataFrame.from_dict(labels,orient='index')
df_lab = df_lab.reset_index()
df_lab.columns=['oid','label']

df_sc = pd.DataFrame.from_dict(scored,orient='index')
df_sc = df_sc.reset_index()
df_sc.columns=['oid','score']

df_lab.to_csv('results_pf/labelled_ano_'+str(field)+'.csv',index=False)
df_sc.to_csv('results_pf/score_ano_'+str(field)+'.csv',index=False)


#a = list(map(list,f.index.value_counts().items()))
