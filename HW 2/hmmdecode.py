import sys
import json

model_path = 'hmmmodel.txt'
f = open(model_path, encoding='UTF-8')
matrix = json.loads(f.read())
tag_count, emission_prob, transition_prob  = matrix[0], matrix[1], matrix[2]

path = sys.argv[1]

f1 = open(path, encoding='UTF-8')
content_lines = f1.readlines()

#3d Dict

def ViterbiAlgorithm(line):
    dict_3d = [{}]
    words = line.split()
    words_length = len(words)
    # First Word
    em_prob = 0
    firstWord = words[0]
    if not (firstWord in emission_prob.keys()):
        tags_for_word = tag_count
    else:
        tags_for_word = emission_prob[firstWord]

    for tag in tags_for_word.keys():
        if tag !="<START>" and firstWord in emission_prob.keys() :
            em_prob = emission_prob[firstWord][tag]

        dict_3d[0][tag] = {}
        dict_3d[0][tag]['max_prob'],  dict_3d[0][tag]['back_pointer'] = transition_prob['<START>'][tag] + em_prob, '<START>'

    #For 2nd word to Nth word
    for i in range(1, words_length):
        word = words[i]
        dict_3d.append({})
        if not (word in emission_prob.keys()):
            tags_for_word = tag_count
        else:
            tags_for_word = emission_prob[word]
        for tag in tags_for_word.keys():
            if (tag != '<START>' and tag != '<END>') and word in emission_prob.keys():
                em_prob = emission_prob[word][tag]
            if word not in emission_prob.keys():
                em_prob = 0
            maxProb = {'p': -sys.maxsize, 'bp': ''}
            for prev_tag in dict_3d[i - 1].keys():
                if prev_tag == '<START>' or prev_tag == '<END>':
                    continue
                tempProb = dict_3d[i - 1][prev_tag]['max_prob'] + transition_prob[prev_tag][tag] + em_prob
                if (maxProb['p']<tempProb):
                    maxProb['p'], maxProb['bp'] = tempProb, prev_tag

            dict_3d[i][tag] = {}
            dict_3d[i][tag]['max_prob'], dict_3d[i][tag]['back_pointer'] = maxProb['p'], maxProb['bp']
    #handling end state
    tags_for_word = dict_3d[-1].keys()
    dict_3d.append({})
    maxProb = {'p': -sys.maxsize, 'bp': ''}
    for tag in tags_for_word:
        if tag != '_END_':
            tempProb = dict_3d[words_length - 1][tag]['max_prob'] + transition_prob[tag]['<END>']
        if (tempProb > maxProb['p']):
            maxProb['p'], maxProb['bp'] = tempProb, tag

    dict_3d[-1]['<END>'] = {}
    dict_3d[-1]['<END>']['max_prob'], dict_3d[-1]['<END>']['back_pointer']  = maxProb['p'], maxProb['bp']

    # POS tagging
    POS_of_line = ""
    tag = "<END>"
    prev_tag_idx = words_length
    j = words_length - 1
    for j in range(j, -1, -1):
        a = words[j] + "/"
        b = dict_3d[prev_tag_idx][tag]['back_pointer'] + " "
        POS_of_line = a + b + POS_of_line
        tag = dict_3d[prev_tag_idx][tag]['back_pointer']
        prev_tag_idx = prev_tag_idx - 1
    return POS_of_line

predicted = []
for line in content_lines:
    predicted.append(ViterbiAlgorithm(line))


output_file_path = 'hmmoutput.txt'
writeFile = open(output_file_path, mode='w', encoding='UTF-8')
for sentence in predicted:
    writeFile.write(sentence + "\n")

