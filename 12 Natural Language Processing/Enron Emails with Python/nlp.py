import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform


#-------------------------------------------------------------------------------------------------#
#--------------------------------------Preparation------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

file_dir = "dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/"

# Load SpaCy's small and large English models
# These must be downloaded first with pip before they will load
sp = spacy.load('en_core_web_sm')
splg = spacy.load('en_core_web_lg')

# Extend Spacy's Doc and Span classes to include NLTK's VADER sentiment analyzer as a method
vader_sentiment_analyzer = SentimentIntensityAnalyzer()
def polarity_scores(doc):
    return vader_sentiment_analyzer.polarity_scores(doc.text)
spacy.tokens.Doc.set_extension('polarity_scores', getter=polarity_scores)
spacy.tokens.Span.set_extension('polarity_scores', getter=polarity_scores)

# Load data
d = pd.read_json(file_dir+'emails.json', orient='records')[:100]


#-------------------------------------------------------------------------------------------------#
#-------------------POS Tagging, Entity Adjectives, and Sentiment Analysis------------------------#
#-------------------------------------------------------------------------------------------------#

def unique_entities(df_col):
    return list(set([entity for sublist in df_col for entity in sublist]))


print(len(unique_entities(d['from'])))
print(len(unique_entities(d['to'])))
print(len(unique_entities(d['cc_email'])))
print(len(unique_entities(d['bcc_email'])))


def mentioned_entities(spacy_obj):
    entities = [e.text.replace(" ", "_") for e in spacy_obj.ents]
    return entities


def adjectives_describing_word(spacy_obj, spacy_ent, window_size_one_side=2):
    words_around_entity = spacy_obj[spacy_ent.start-window_size_one_side:spacy_ent.end+window_size_one_side]
    adjs_describing_entity = [w.text for w in spacy_obj[spacy_ent.start-2:spacy_ent.end+2] if w.pos_ in ['ADJ']]
    return adjs_describing_entity


def describe_all_entities(spacy_obj):
    entity_descriptions = dict.fromkeys([e.text.replace(" ", "_") for e in spacy_obj.ents])
    for ent in spacy_obj.ents:
        entity_descriptions[ent.text.replace(" ", "_")] = adjectives_describing_word(spacy_obj, ent)
    return entity_descriptions


def entity_sentiment(spacy_obj, spacy_ent):
    all_sents_involving_entity = [sent for sent in spacy_obj.sents if spacy_ent.string in sent.string]
    entity_sent_scores = [s._.polarity_scores for s in all_sents_involving_entity]
    # Average the sentiment scores for sentences involving entity
    temp_df = pd.DataFrame.from_dict(entity_sent_scores)
    entity_sent_scores = dict(temp_df.mean())
    entity_sent_scores = {k:v for k,v in entity_sent_scores.items() if k in ['neg', 'pos', 'neu']}
    entity_sent_score_catg = list(entity_sent_scores.keys())
    return [(sent_score_catg, entity_sent_scores[sent_score_catg]) for sent_score_catg in entity_sent_score_catg]


def sentiment_analyze_all_entities(spacy_obj):
    entity_sentiments = dict.fromkeys([e.text.replace(" ", "_") for e in spacy_obj.ents])
    for ent in spacy_obj.ents:
        entity_sentiments[ent.text.replace(" ", "_")] = entity_sentiment(spacy_obj, ent)
    return entity_sentiments


# Explore one document
sen = sp(d['text'][2])
print(sen.ents)
print([e.label_ for e in sen.ents])
print([w.pos_ for w in sen if w.pos_ in ['ADJ']])
print([w.text for w in sen if w.pos_ in ['ADJ']])
# Test the 4 functions defined for entity adjectives and sentiment
print(adjectives_describing_word(sen, sen.ents[0]))
print(describe_all_entities(sen))
print(entity_sentiment(sen, sen.ents[0]))
print(sentiment_analyze_all_entities(sen))

# Apply functions to all documents
d['entities'] = d['text'].apply(lambda x: mentioned_entities(sp(x)))
# Filter dataframe to rows with entities listed, don't care about other rows for entity analysis
d_ents = d[d['entities'].map(len) > 0].copy().reset_index(drop=True)
d_ents['entity_adjectives'] = d_ents['text'].apply(lambda x: describe_all_entities(sp(x)))
d_ents['entity_sentiment'] = d_ents['text'].apply(lambda x: sentiment_analyze_all_entities(sp(x)))


def plot_wordcloud_of_entity_adjectives(df, entity):
    try:
        d_ent_sub = df[df['entities'].map(set([entity]).issubset)].copy().reset_index(drop=True)
        d_ent_sub['entity_adjectives'] = d_ent_sub['text'].apply(lambda x: describe_all_entities(sp(x)))
        adjectives_list = d_ent_sub['text'].apply(lambda x: adjectives_describing_word(sp(x), [ent for ent in sp(x).ents if ent.text == entity][0])).tolist()
        wc = WordCloud().generate([''.join(a) for sublist in adjectives_list for a in sublist][0])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        return print("Entity not found within these emails.")

plot_wordcloud_of_entity_adjectives(d, 'Enron')


def get_sentiment_of_entity_from_person(df, entity, person, plot=False):
    if person != 'All Employees':
        df = df[df['from'] == person]
    df_sub = df[pd.DataFrame(df['entities'].tolist()).isin([entity]).any(1)].copy().reset_index(drop=True)
    df_sub = df_sub[['date', 'entity_sentiment']]
    df_sub['entity_sentiment'] = df_sub['entity_sentiment'].apply(lambda x: x[entity])
    if plot:
        sentiment_names = df_sub['entity_sentiment'].apply(lambda x: [tup[0] for tup in x])[0]
        for sentiment_index, sentiment_name in enumerate(sentiment_names):
            if sentiment_name == 'pos':
                line_color ='darkgreen'
            elif sentiment_name == 'neg':
                line_color = 'crimson'
            else:
                line_color = 'blue'
            sns.set()
            plt.plot(df_sub['date'], df_sub['entity_sentiment'].apply(lambda x: x[sentiment_index][1]), line_color, label=sentiment_name)
        plt.xlabel('Date')
        plt.ylabel('Percentage of Sentiments in Category (VADER Score)')
        plt.title(person + ' Sentiment of ' + entity + ' over Time')
        plt.legend()
        plt.show()
    return df_sub


# Note that VADER scores for pos, neg, neu sum to 1, so each one is the proportion of sentiments in each category
print("Phillip K Allen sentiment of Enron:", get_sentiment_of_entity_from_person(d_ents, 'Enron', 'Phillip K Allen', plot=True))
print("All employees' sentiment of Enron:", get_sentiment_of_entity_from_person(d_ents, 'Lucy', 'All Employees', plot=True))


#-------------------------------------------------------------------------------------------------#
#------------------------------------Topic Modeling-----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Limit number of words/features to 1000
nbr_features = 1000
#NMF uses tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=nbr_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(d['text'])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#LDA uses raw term frequencies
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=nbr_features, stop_words='english')
tf = tf_vectorizer.fit_transform(d['text'])
tf_feature_names = tf_vectorizer.get_feature_names()

nbr_topics = 20
nmf = NMF(n_components=nbr_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf_topics = nmf.transform(tfidf)
lda = LatentDirichletAllocation(n_components=nbr_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_topics = lda.transform(tf)


# Plot pretty LDA output
lda_vis_data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
lda_vis_data_html = pyLDAvis.prepared_data_to_html(lda_vis_data)
pyLDAvis.show(lda_vis_data)


# Assume top 10 words will be used per topic
top_n = 10
total_rows = nbr_topics*top_n
topic = []
topic_top_ten = []
topic_top_ten_scores = []
for tid, t in enumerate(lda.components_):
    topic.append([tid+1]*top_n)
    topic_top_ten.append([tf_feature_names[i] for i in t.argsort()[:-top_n -1:-1]])
    topic_top_ten_scores.append(t[t.argsort()[:-top_n -1:-1]])
top_words = np.concatenate([np.array(topic).reshape(total_rows,1), np.array(topic_top_ten).reshape(total_rows,1), np.array(topic_top_ten_scores).reshape(total_rows,1)], axis=1)
topwordsdf = pd.DataFrame(top_words, columns=['topic', 'word', 'score'])
topwordsdf['topic'] = topwordsdf['topic'].astype('int64')
topwordsdf['score'] = topwordsdf['score'].astype('float64')

#The higher this threshold, the fewer results will be returned when data is filtered
lda_score_threshold = 0.2


#Set up bar chart of top words by topic
def create_bar_top_words(topwords_df, topic):
    topwords_df[topwords_df['topic']==topic].sort_values(by=['score'], ascending=True).plot.barh(x='word', y='score', title='Top 10 Words for Topic '+str(topic), colormap='Paired', legend=False)
    plt.xlabel('score')
    plt.tight_layout()
    plt.show()
    return

create_bar_top_words(topwordsdf, topic=1)


def create_doc_topic_heatmap(topics_input):
    heatmap_x_labels = ["Topic %d" % i for i in range(1, topics_input.shape[1]+1)]
    heatmap_y_labels = ["%d" % i for i in range(1, topics_input.shape[0]+1)]
    sns.heatmap(topics_input, cmap='Greys', xticklabels=heatmap_x_labels, yticklabels=heatmap_y_labels)
    plt.title('Probability that Records are Related to Topics')
    plt.tight_layout()
    plt.show()
    return

create_doc_topic_heatmap(lda_topics[:24,:])


# Query email text for emails containing a term
d[d['text'].str.contains('resign')].reset_index(drop=True)['text']
d[d['text'].str.contains('resign')].reset_index(drop=True)['text'].to_csv(file_dir+'resign_emails.csv')
# Hand pick the emails with true resignations (there are only 231 to sort through)
resignation_indices_found_by_manual_search = [4, 31, 87]
empls_who_resigned = d.loc[resignation_indices_found_by_manual_search, 'from']
# Phillip K Allen is the only person who resigned and could easily be identified
# Refer back to his sentiment of Enron over time, plotted above


#-------------------------------------------------------------------------------------------------#
#---------------Document Similarity with Document Vectors and Word Embeddings---------------------#
#-------------------------------------------------------------------------------------------------#


# Average word vectors from Word2Vec to create a document vector
doc_vecs = d['text'].apply(lambda x: splg(x).vector).values
doc_vecs = np.concatenate(doc_vecs, axis=0).reshape(len(doc_vecs), 300)
print(doc_vecs.shape)
print(doc_vecs[0].shape)
sns.distplot(doc_vecs[0]) # check for normality in embedding space
plt.show()
doc_sims = squareform(pdist(doc_vecs, metric='cosine')) # cosine similarity matrix

# Find similar emails using a similarity threshold
len(doc_sims[doc_sims > 0.9])
very_similar_docs = np.where(doc_sims > 0.9)

def print_similar_docs(df, doc1_df_index, doc2_df_index, field_name):
    print("Document 1: \n")
    print(df.loc[doc1_df_index, field_name])
    print("\nDocument 2: \n")
    print(df.loc[doc2_df_index, field_name])
    return

for comparison in range(len(very_similar_docs[0])):
    print("\n\n -----------------------------",
          "\nDocument Text Comparison\n")
    print_similar_docs(d, very_similar_docs[0][comparison], very_similar_docs[1][comparison], 'text')
