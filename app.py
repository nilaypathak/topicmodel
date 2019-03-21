import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
import scipy
app = flask.Flask(__name__)
CORS(app, support_credentials=True)
app.config["DEBUG"] = True
x = pd.read_csv('january2019_compiled.csv')
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'sit','next','make', 'see', 'want', 'come', 'take', 'use', 'would', 'can','could','find','many','feel','give','still','look','think']
stopwords_other = ['one', 'mr','image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something','of','s','THE','should','do','ms','week','another','thing','month','day','come',
    'york','away','left','wrote','came','tell','asked','new',"'",'to','it','email.___There','in',',','.','only',
    'left','right','hand','point','often','talk','head','point','ago','whether','ll','find',':',"didn'",
    'hour','group','became','become','becomes','often','sometimes','usually',"000","said","much","dr"]

my_stopwords = list(set(stopwords.words('english') + stopwords_verbs + stopwords_other))

def runmodel(date,topics):
    
    npr=x[x['date']==date]

    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words=my_stopwords)
    dtm = cv.fit_transform(npr['compiled'])
    LDA = LatentDirichletAllocation(n_components=topics,random_state=42)
    LDA.fit(dtm)
    topicwords=[]
    for index,topic in enumerate(LDA.components_):
        topicwords.append([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    topic_results = LDA.transform(dtm)
    npr['Topic'] = topic_results.argmax(axis=1)
    table=[]
    count=0
    for index,row in npr.iterrows():
        subtable={
            'title':row['main_headline'],
            'article':row['article'],
            'date':row['date'],
            'topic':row['Topic']
        }
        table.append(subtable)
        count=count+1
        if(count>=50):
            break
    results={
        'topicwords':topicwords,
        'table':table
    }
    return results


@app.route('/', methods=['GET'])
@cross_origin(supports_credentials=True)
def home():
    results = []
    if 'date' in request.args and 'topics' in request.args:
        date = request.args['date']
        topics = int(request.args['topics'])
        # results.append({'date':date,
        # 'topics':topics})
        results = runmodel(date,topics)
    else:
        return "Invalid input, please try again."
    
    return jsonify(results)

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)