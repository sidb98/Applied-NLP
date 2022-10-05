import os
import sys
import re
import json
import math
import glob


path = sys.argv[1]
#path = r"C:\Users\siddh\PycharmProjects\naivebayes\op_spam_training_data"
pos_true_path = glob.glob(os.path.join(path, r"positive_polarity/truthful_from_TripAdvisor/*/*.txt"))
pos_decep_path = glob.glob(os.path.join(path, r"positive_polarity/deceptive_from_MTurk/*/*.txt"))
neg_true_path = glob.glob(os.path.join(path, r"negative_polarity/truthful_from_Web/*/*.txt"))
neg_decep_path = glob.glob(os.path.join(path, r"negative_polarity/deceptive_from_MTurk/*/*.txt"))

total = len(pos_true_path) + len(pos_decep_path) + len(neg_true_path) + len(neg_decep_path)
pos_true_prior = math.log(len(pos_true_path)/total)
pos_decep_prior = math.log(len(pos_decep_path)/total)
neg_true_prior = math.log(len(neg_true_path)/total)
neg_decep_prior = math.log(len(neg_decep_path)/total)

prior_class_prob = {"pos_true_prior": pos_true_prior,
                    "pos_decep_prior": pos_decep_prior,
                    "neg_true_prior": neg_true_prior,
                    "neg_decep_prior": neg_decep_prior}

stopwordList = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]


def build_class_vocab(all_txt_path):
    vocab = {}
    file_count = 0
    for txt_path in all_txt_path:
        file_count += 1
        file = open(txt_path, 'r')
        content = file.read()
        content = content.lower()
        content = re.sub(r'[^\w\s]', ' ', content)
        for word in content.split():
            if word not in vocab:
                if word not in stopwordList:
                    vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


pos_true_vocab = build_class_vocab(pos_true_path)
pos_decep_vocab = build_class_vocab(pos_decep_path)
neg_true_vocab = build_class_vocab(neg_true_path)
neg_decep_vocab = build_class_vocab(neg_decep_path)

posterior_dict = {}

def createPosteriorDict(dict, idx):
    for key, value in dict.items():
        if key in posterior_dict:
            posterior_dict[key][idx] = dict[key]
        else:
            posterior_dict[key] = [0, 0, 0, 0]
            posterior_dict[key][idx] = dict[key]

createPosteriorDict(pos_true_vocab, 0)
createPosteriorDict(pos_decep_vocab, 1)
createPosteriorDict(neg_true_vocab, 2)
createPosteriorDict(neg_decep_vocab,3)

pre_smooth_dict = posterior_dict

def laplaceSmoothing():
    for key, value in posterior_dict.items():
        value[0] = (value[0]+1)/(sum(pos_true_vocab.values())+len(posterior_dict))
        value[1] = (value[1]+1)/(sum(pos_decep_vocab.values())+len(posterior_dict))
        value[2] = (value[2]+1)/(sum(neg_true_vocab.values())+len(posterior_dict))
        value[3] = (value[3]+1)/(sum(neg_decep_vocab.values())+len(posterior_dict))
        posterior_dict[key] = [math.log(value) for value in posterior_dict[key]]
laplaceSmoothing()

#print(posterior_dict)

nb_model = {'posterior_prob':posterior_dict,
            'prior_prob': prior_class_prob}
with open('nbmodel.txt', 'w') as file:
    file.write(json.dumps(nb_model))

