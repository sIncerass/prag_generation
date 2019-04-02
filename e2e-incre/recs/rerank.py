import json
import numpy as np
input_name = "devset.recs.full.txt"
output_name = "devset.recs.final"

def minmax(a): # normalize the weight
    return (a-min(a))/(1.0*(max(a) - min(a)))

# adjust the fw/bw weights for reranker
fw_weight = 0.49
bw_weight = 1.0 - fw_weight
def reranker(fw_weight, bw_weight):
    global output_name, input_name
    with open(input_name, "r") as stream, open("rerank/%s_%.2f" % (output_name, fw_weight), "w") as out:
        for line in stream:
            data = line.strip().split("\t")
            texts = json.loads(data[0])
            fw_score = np.array(json.loads(data[1]))
            bw_score = -np.array(json.loads(data[2]))
            fw_score_norm, bw_score_norm = minmax(fw_score), minmax(bw_score)
            total_score = fw_score_norm * fw_weight + bw_score_norm * bw_weight
            select_id = np.argmax(total_score)
            out.write("%s\n" % texts[select_id])

for _ in range(11):
    fw_weight = 0.4 + _ / 100.0
    bw_weight = 1.0 - fw_weight
    reranker(fw_weight, bw_weight)
