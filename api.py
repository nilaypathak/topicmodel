import flask
from flask import request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
import scipy
app = flask.Flask(__name__)
app.config["DEBUG"] = True
x = pd.read_csv('january2019_compiled.csv')
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'sit','next','make', 'see', 'want', 'come', 'take', 'use', 'would', 'can','could','find','many','feel','give','still','look','think']
stopwords_other = ['one', 'mr','image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something','of','s','THE','should','do','ms','week','another','thing','month','day','come',
    'york','away','left','wrote','came','tell','asked','new',"'",'to','it','email.___There','in',',','.','only',
    'left','right','hand','point','often','talk','head','point','ago','whether','ll','find',':',"didn'",
    'hour','group','became','become','becomes','often','sometimes','usually',"000","said","much","dr"]

my_stopwords = list(set(stopwords.words('English') + stopwords_verbs + stopwords_other))

def runmodel(date,topics):
    
    npr=x[x['date']==date]

    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words=my_stopwords)
    dtm = cv.fit_transform(npr['compiled'])
    LDA = LatentDirichletAllocation(n_components=topics,random_state=42)
    LDA.fit(dtm)
    results=[]
    for index,topic in enumerate(LDA.components_):
        results.append([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    return results


@app.route('/', methods=['GET'])
def home():
    results = []
    if 'date' in request.args and 'topics' in request.args:
        date = request.args['date']
        topics = int(request.args['topics'])
        # results.append({'date':date,
        # 'topics':topics})
        results = runmodel(date,topics)
    else:
        return "Error: No id field provided. Please specify an id."
    
    return jsonify(results)

app.run()