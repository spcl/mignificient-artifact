
import sys
sys.path.append('/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/ml-inference/BERT-SQuAD/')

print("import", flush=True)
from bert import QA
print("finish import",flush=True)
from timeit import default_timer as timer

import torch
print(torch.__file__)

model = None

def function(obj):

    global model

    if model is None:

        print("model", flush=True)
        before = timer()
        model = QA('/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/ml-inference/BERT-SQuAD/model')
        after = timer()
        print("model end", flush=True)

        print('model eval time:', after - before, flush=True)

    start = timer()

    doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
        "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
        "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
        "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
        "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
        "people in a referendum, depending on the provision. "

    q = 'When did Victoria enact its constitution?'

    print('model predict', flush=True)
    answer = model.predict(doc, q)
    print(answer['answer'], flush=True)
# print(answer.keys())

    end = timer()
    print(end - start)

function({})
