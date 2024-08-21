# Note that the output for these algos are Markov equivalent classes
# They are not Directed acyclic graphs
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased. lingam. utils import make_dot
import pandas as pd


# Visualization using pydot
from causallearn. utils. GraphUtils import GraphUtils
import matplotlib. image as mpimg
import matplotlib. pyplot as plt
import io


data_file= pd.read_csv('ESSOREI_LV_24.csv')
data_df = pd.DataFrame(data=data_file, columns=['ATB',	'ESS',	'OR',	'PBC',	'SSN'])
vbles = [f'{col}' for i, col in enumerate(data_df.columns)]

data = data_df.to_numpy()
print(data_df)


""" PC Algorithm"""

# cg = pc(data)

# pyd = GraphUtils.to_pydot(cg.G, labels=vbles)
# tmp_png=pyd.create_png(f="png")
# fp=io.BytesIO(tmp_png)
# img =mpimg.imread(fp,format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()


""" FCI Algorithm """

# G, edges= fci(data)

# pyd = GraphUtils.to_pydot(G, labels=vbles)
# tmp_png=pyd.create_png(f="png")
# fp=io.BytesIO(tmp_png)
# img =mpimg.imread(fp,format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()


""" GES Algorithm"""

# Record = ges(data)

# pyd = GraphUtils.to_pydot(Record['G'], labels=vbles)
# tmp_png=pyd.create_png(f="png")
# fp=io.BytesIO(tmp_png)
# img = mpimg.imread(fp,format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()

""" cdnod Algorithm"""

# cg = cdnod(data)
# or customized parameters
# cg = cdnod(data, c_indx, alpha, indep_test, stable, uc_rule, uc_priority, mvcdnod,
# correction_name, background_knowledge, verbose, show_progress)



"""Lingam Based Method"""

model = lingam.ICALiNGAM()
model.fit(data)


print(make_dot(model._adjacency_matrix, labels=vbles))

#  OR -> ATB [label=0.60]
#         ESS -> OR [label=0.29]
#         ATB -> PBC [label=0.23]
#         OR -> PBC [label=0.52]
#         ATB -> SSN [label=0.30]
#         ESS -> SSN [label=0.15]
#         PBC -> SSN [label=0.33]

# pyd = GraphUtils.to_pydot(cg.G, labels=vbles)
# tmp_png=pyd.create_png(f="png")
# fp=io.BytesIO(tmp_png)
# img = mpimg.imread(fp,format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()

