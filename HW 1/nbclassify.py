import os
import sys
import glob
import json
import re


test_path = sys.argv[1]
#test_path = r"C:\Users\siddh\PycharmProjects\naivebayes\op_spam_training_data"
test_files = glob.glob(os.path.join(test_path, "*/*/*/*.txt"))

model_file = open('nbmodel.txt', 'r')
data = model_file.read()
json_object = json.loads(data)


output_file = open("nboutput.txt", 'w')

for file in test_files:
    pos_true_prob = json_object["prior_prob"]["pos_true_prior"]
    pos_decep_prob = json_object["prior_prob"]["pos_decep_prior"]
    neg_true_prob = json_object["prior_prob"]["neg_true_prior"]
    neg_decep_prob = json_object["prior_prob"]["neg_decep_prior"]

    f = open(file, 'r')
    content = f.read()
    content = content.lower()
    content = re.sub(r'[^\w\s]', ' ', content)
    for word in content.split():
        if word in json_object["posterior_prob"]:
            pos_true_prob += json_object["posterior_prob"][word][0]
            pos_decep_prob += json_object["posterior_prob"][word][1]
            neg_true_prob += json_object["posterior_prob"][word][2]
            neg_decep_prob += json_object["posterior_prob"][word][3]

    dict = {
        pos_true_prob: "truthful positive {}\n".format(file),
        pos_decep_prob: "deceptive positive {}\n".format(file),
        neg_true_prob: "truthful negative {}\n".format(file),
        neg_decep_prob: "deceptive negative {}\n".format(file),
    }
    prob_max = max(pos_true_prob, pos_decep_prob, neg_true_prob, neg_decep_prob)
    output_file.write(dict[prob_max])
output_file.close()