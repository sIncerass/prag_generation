import csv
import json

beam_size = 10
def tidy_total(input_name):
    global beam_size
    import pickle as pk
    dev_recs_loss = pk.load(open("dev_recs_loss.pkl", "rb"))
    with open(input_name, "r") as stream, open("devset.recs.full.txt", "w") as stream_1:
        for idx, line in enumerate(stream):
            data = line.strip().split("\t")
            bw_loss = dev_recs_loss[ idx*beam_size:(idx+1)*beam_size ]
            stream_1.write("%s\t%s\t%s\n" % ( data[0], data[1], json.dumps(bw_loss) ))

src_name = "/data/sheng/e2e/data/devset.csv.multi-ref.src"
input_name = "weights.epoch8.devset.recs.txt"
output_name = "devset.recs.csv"
tidy_first = True
# after tidy the dataset into recs.csv for reconstructor and train the entire model
# we could tidy every thing together into xxx.recs.full.txt

if tidy_first:
    src = []
    with open(src_name, "r") as stream:
        for line in stream:
            src.append(line.strip())
    with open(output_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mr', 'ref'])
        with open(input_name, "r") as stream:
            for idx, line in enumerate(stream):
                data = line.strip().split("\t")
                refs = json.loads(data[0])
                for ref in refs:
                    writer.writerow([str(src[idx]), ref])
else:
    tidy_total(input_name)


