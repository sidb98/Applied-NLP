import sys
import json
import math

path = sys.argv[1]
f = open(path, encoding='UTF-8')
content_lines = f.readlines()
tag_count = {"<START>": len(content_lines), "<END>": len(content_lines)}

transition_dict = {}
emission_dict = {}
prev_tag = ''

for line in content_lines:
    prev_tag = "<START>"
    wordNTagList = line.split()
    for wordsNTag in wordNTagList:
        word, tag = wordsNTag.rsplit("/", 1)
        if (tag not in tag_count):
            tag_count[tag] = 1
        else:
            tag_count[tag] += 1

        # Creating Emission Matrix
        if not (word in emission_dict):
            emission_dict[word] = {}
            emission_dict[word][tag] = 1
        else:
            if tag in emission_dict[word]:
                emission_dict[word][tag] += 1
            else:
                emission_dict[word][tag] = 1
        # Creating Transition Matrix
        if not (prev_tag in transition_dict):
            transition_dict[prev_tag] = {}
            transition_dict[prev_tag][tag] = 1
        else:
            if tag in transition_dict[prev_tag]:
                transition_dict[prev_tag][tag] += 1
            else:
                transition_dict[prev_tag][tag] = 1

        prev_tag = tag

    if not (prev_tag in transition_dict):
        transition_dict[prev_tag] = dict()
        transition_dict[prev_tag]['<END>'] = 1
    else:
        if ('<END>' in transition_dict[prev_tag]):
            transition_dict[prev_tag]['<END>'] = transition_dict[prev_tag]['<END>'] +1
        else:
            transition_dict[prev_tag]['<END>'] = 1



# Smoothing Transition Matrix
for tag_transition_dict in transition_dict:
    tag_transition_dict_val = 0
    #Adding +1
    for tag in tag_count:
        if tag in transition_dict[tag_transition_dict]:
            transition_dict[tag_transition_dict][tag] +=1
        else:
            transition_dict[tag_transition_dict][tag] = 1

    tag_transition_dict_val = transition_dict[tag_transition_dict].values()
    total = sum(tag_transition_dict_val)

    for tag in transition_dict[tag_transition_dict]:
        transition_dict[tag_transition_dict][tag] = transition_dict[tag_transition_dict][tag] / total
        transition_dict[tag_transition_dict][tag] = math.log(transition_dict[tag_transition_dict][tag])



#Log Space Emission Matrix
for word in emission_dict:
    for tag in emission_dict[word]:
        emission_dict[word][tag] = emission_dict[word][tag]/tag_count[tag]
        emission_dict[word][tag] = math.log(emission_dict[word][tag])


hmm_model = [tag_count, emission_dict, transition_dict]
with open('hmmmodel.txt', 'w') as file:
    file.write(json.dumps(hmm_model))
