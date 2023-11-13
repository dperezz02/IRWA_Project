
import time
import json
from collections import Counter, defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import string
from openai.embeddings_utils import cosine_similarity
import re
import pandas as pd
import matplotlib.pyplot as plt
#from torch import cosine_similarity
#from wordcloud import WordCloud
#from sentence_transformers import SentenceTransformer
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
nltk.download('stopwords')
from utils import build_terms, read_tweets
import csv
#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def create_index(lines):
    """
    Implement the inverted index

    Argument:
    lines -- collection of Wikipedia articles

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    """
    tf = defaultdict(list) 
    df = defaultdict(int)  
    idf = defaultdict(float)

    index = defaultdict(list)
    num_documents = len(lines)
    for line in lines:  # Remember, lines contain all documents: article-id | article-title | article-body
       
        line = json.loads(line)
        line_arr = line["full_text"]
        tweet_id = line["id"]  # Get the tweet ID
        terms = build_terms(line_arr)

        current_page_index = {}

        for position, term in enumerate(terms): # terms contains page_title + page_text. Loop over all terms
            try:
                current_page_index[term][tweet_id].append(position)

            except:
               
                current_page_index[term] = [tweet_id, array('I', [position])]

        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append([posting[0], np.round(len(posting[1]) / norm, 4)]) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

        # Compute IDF following the formula (3) above. HINT: use np.log
        for term in df:
            idf[term] = np.round(np.log(float(num_documents / df[term])), 4)
    return index, tf, df, idf

def scatter_plot(df):
    # Apply T-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tweet_tsne = tsne.fit_transform(df.vector_representation.values())

    # Plot the tweets in a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tweet_tsne[:, 0], tweet_tsne[:, 1])
    plt.title('T-SNE Visualization of Tweets')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.savefig("./scatter_plot")
    plt.close()  # Close the plot to release resources

def rank_documents(terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)


    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index][1] * idf[term]

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    
    result_docs = [x[1] for x in doc_scores]
    result_scores = [x[0] for x in doc_scores]
    
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)

    return result_docs, result_scores


def search_tf_idf(query, index, idf, tf):
    """
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs = [posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs |= set(term_docs)
        except:
            #term is not in index
            pass
    docs = list(docs)
    ranked_docs, scores = rank_documents(query, docs, index, idf, tf)
    return ranked_docs, scores

def select_docs(data, query_id):
    subset = []
    ground_truths = []

    for line in data:
        doc, q_id, label = line.split(',')
        if q_id == query_id:
            subset.append(doc)
            if label == '1':
                ground_truths.append(1)
            else:
                ground_truths.append(0)
        elif label == '1':
            subset.append(doc)
            ground_truths.append(0)

    return subset, ground_truths

def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta el encabezado
        data = [",".join(row) for row in reader]
    return data

def precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    precision @k : float

    """
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    relevant = sum(doc_score == 1)
    return float(relevant) / k

def recall_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    precision @k : float

    """
    r = np.sum(doc_score)
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    relevant = sum(doc_score == 1)
    return float(relevant) / r

def avg_precision_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    average precision @k : float
    """
    gtp = np.sum(doc_score)
    order = np.argsort(y_score)[::-1]
    doc_score = np.take(doc_score, order[:k])
    ## if all documents are not relevant
    if gtp == 0:
        return 0
    n_relevant_at_i = 0
    prec_at_i = 0
    for i in range(len(doc_score)):
        if doc_score[i] == 1:
            n_relevant_at_i += 1
            prec_at_i += n_relevant_at_i / (i + 1)
    return prec_at_i / gtp

def dcg_at_k(doc_score, y_score, k=10):
    order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    doc_score = np.take(doc_score, order[:k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    gain = 2 ** doc_score - 1  # Compute gain (use formula 7 above)
    discounts = np.log2(np.arange(len(doc_score)) + 2)  # Compute denominator
    return np.sum(gain / discounts)  #return dcg@k


def ndcg_at_k(doc_score, y_score, k=10):
    dcg_max = dcg_at_k(doc_score, doc_score, k)
    if not dcg_max:
        return 0
    return np.round(dcg_at_k(doc_score, y_score, k) / dcg_max, 4)

def rr_at_k(doc_score, y_score, k=10):
    """
    Parameters
    ----------
    doc_score: Ground truth (true relevance labels).
    y_score: Predicted scores.
    k : number of doc to consider.

    Returns
    -------
    Reciprocal Rank for qurrent query
    """

    order = np.argsort(y_score)[::-1]  # get the list of indexes of the predicted score sorted in descending order.
    doc_score = np.take(doc_score, order[
                             :k])  # sort the actual relevance label of the documents based on predicted score(hint: np.take) and take first k.
    if np.sum(doc_score) == 0:  # if there are not relevant doument return 0
        return 0
    return 1 / (np.argmax(doc_score == 1) + 1)  # hint: to get the position of the first relevant document use "np.argmax"


def main():
    file_path = ''
    start_time = time.time()
    docs_path = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/Rus_Ukr_war_data.json'
    ev1 = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/Evaluation_gt.csv'
    evaluation_data1 = read_csv(ev1)
    ev2 = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/evaluation_custom_queries.csv'
    evaluation_data2 = read_csv(ev2)
    with open(docs_path) as fp: lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]
    print("There are ", len(lines), " tweets")

    ids_path = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/Rus_Ukr_war_data_ids.csv'
    doc_ids = pd.read_csv(ids_path,sep='\t', header=None)
    doc_ids.columns = ["doc_id", "tweet_id"]
    tweet_document_ids_map = {}
    for index, row in doc_ids.iterrows():
        tweet_document_ids_map[row['tweet_id']] = row['doc_id']
    
    # Process lines to create a list of tweet IDs
    tweet_ids = [json.loads(line)["id"] for line in lines]
    tweets_texts = [json.loads(line)["full_text"] for line in lines]
    tweet_text = pd.DataFrame({'tweet_id': tweet_ids, 'text': tweets_texts})
    #index, tf, df, idf = create_index(lines)

    baseline_queries = [
        "Tank Kharkiv",
        "Nord Stream pipeline",
        "Annexation territories"
    ]
    
    docs_Q1, ground_truths_Q1 = select_docs(evaluation_data1,"Q1")
    docs_Q2, ground_truths_Q2 = select_docs(evaluation_data1,"Q2")
    docs_Q3, ground_truths_Q3 = select_docs(evaluation_data1,"Q3")
    
    subsetQ1 = [line for line in lines if tweet_document_ids_map[json.loads(line)["id"]] in(docs_Q1)]
    subsetQ1 = sorted(subsetQ1, key=lambda line: docs_Q1.index(tweet_document_ids_map[json.loads(line)["id"]]))
    subset_tweets_idsQ1 = [json.loads(line)["id"] for line in subsetQ1]
    subindexQ1, subtfQ1, subdfQ1, subidfQ1 = create_index(subsetQ1)
    resultsQ1, scoresQ1 = search_tf_idf(baseline_queries[0], subindexQ1, subidfQ1, subtfQ1)
    y_scoresQ1 = [scoresQ1[resultsQ1.index(tweet)] if(tweet in resultsQ1) else 0 for tweet in subset_tweets_idsQ1]
    relevant_tweetsQ1 = tweet_text[tweet_text["tweet_id"].isin(resultsQ1)]
    #print(relevant_tweetsQ1["text"])
    file_path = 'outputQ1.txt'
    # Open the file in write mode and save the text content
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(relevant_tweetsQ1.to_string(index=False))

    subsetQ2 = [line for line in lines if tweet_document_ids_map[json.loads(line)["id"]] in(docs_Q2)]
    subsetQ2 = sorted(subsetQ2, key=lambda line: docs_Q2.index(tweet_document_ids_map[json.loads(line)["id"]]))
    subset_tweets_idsQ2 = [json.loads(line)["id"] for line in subsetQ2]
    subindexQ2, subtfQ2, subdfQ2, subidfQ2 = create_index(subsetQ2)
    resultsQ2, scoresQ2 = search_tf_idf(baseline_queries[1], subindexQ2, subidfQ2, subtfQ2)
    y_scoresQ2 = [scoresQ2[resultsQ2.index(tweet)] if(tweet in resultsQ2) else 0 for tweet in subset_tweets_idsQ2]

    subsetQ3 = [line for line in lines if tweet_document_ids_map[json.loads(line)["id"]] in(docs_Q3)]
    subsetQ3 = sorted(subsetQ3, key=lambda line: docs_Q3.index(tweet_document_ids_map[json.loads(line)["id"]]))
    subset_tweets_idsQ3 = [json.loads(line)["id"] for line in subsetQ3]
    subindexQ3, subtfQ3, subdfQ3, subidfQ3 = create_index(subsetQ3)
    resultsQ3, scoresQ3 = search_tf_idf(baseline_queries[2], subindexQ3, subidfQ3, subtfQ3)
    y_scoresQ3 = [scoresQ3[resultsQ3.index(tweet)] if(tweet in resultsQ3) else 0 for tweet in subset_tweets_idsQ3]

    #EVALUATION
    print(f"Query: {baseline_queries[0]}")
    precision_Q1 = precision_at_k(ground_truths_Q1, y_scoresQ1)
    print("Precision at 10 of Query 1: ", precision_Q1)
    recall_Q1 = recall_at_k(ground_truths_Q1, y_scoresQ1)
    print("Recall at 10 of Query 1: ", recall_Q1)
    avg_precision_Q1 = avg_precision_at_k(ground_truths_Q1, y_scoresQ1)
    print("Average Precision at 10 of Query 1: ", avg_precision_Q1)
    fscore_Q1 = (2*recall_Q1*precision_Q1)/(recall_Q1+precision_Q1)
    print("F1-Score at 10 of Query 1: ", fscore_Q1)
    ndcg_Q1 = ndcg_at_k(ground_truths_Q1, y_scoresQ1)
    print("NDG at 10 of Query 1: ", ndcg_Q1)

    print("\n")
    print(f"Query: {baseline_queries[1]}")
    precision_Q2 = precision_at_k(ground_truths_Q2, y_scoresQ2)
    print("Precision at 10 of Query 2: ", precision_Q2)
    recall_Q2 = recall_at_k(ground_truths_Q2, y_scoresQ2)
    print("Recall at 10 of Query 2: ", recall_Q2)
    avg_precision_Q2 = avg_precision_at_k(ground_truths_Q2, y_scoresQ2)
    print("Average Precision at 10 of Query 2: ", avg_precision_Q2)
    fscore_Q2 = (2*recall_Q2*precision_Q2)/(recall_Q2+precision_Q2)
    print("F1-Score at 10 of Query 2: ", fscore_Q2)
    ndcg_Q2 = ndcg_at_k(ground_truths_Q2, y_scoresQ2)
    print("NDG at 10 of Query 2: ", ndcg_Q2)

    print("\n")
    print(f"Query: {baseline_queries[2]}")
    precision_Q3 = precision_at_k(ground_truths_Q3, y_scoresQ3)
    print("Precision at 10 of Query 3: ", precision_Q3)
    recall_Q3 = recall_at_k(ground_truths_Q3, y_scoresQ3)
    print("Recall at 10 of Query 3: ", recall_Q3)
    avg_precision_Q3 = avg_precision_at_k(ground_truths_Q3, y_scoresQ3)
    print("Average Precision at 10 of Query 3: ", avg_precision_Q3)
    fscore_Q3 = (2*recall_Q3*precision_Q3)/(recall_Q3+precision_Q3)
    print("F1-Score at 10 of Query 3: ", fscore_Q3)
    ndcg_Q3 = ndcg_at_k(ground_truths_Q3, y_scoresQ3)
    print("NDG at 10 of Query 3: ", ndcg_Q3)

    print("\n")
    map = (avg_precision_Q1+avg_precision_Q2+avg_precision_Q3)/3
    print("MAP of Queries 1,2 and 3: ", map)
    mrr = (rr_at_k(ground_truths_Q1, y_scoresQ1)+rr_at_k(ground_truths_Q2, y_scoresQ2)+rr_at_k(ground_truths_Q3, y_scoresQ3))/3
    print("MRR of Queries 1,2 and 3: ", mrr)


    # query = 'putin and the war'
    # results = search_tf_idf(query, index, idf, tf)

    # relevant_tweets = tweet_text[tweet_text["tweet_id"].isin(results)]
    # print(relevant_tweets["text"])

    #     # Define test queries
    # test_queries = [
    #     "Russian military intervention in Ukraine",
    #     "Impact of sanctions on Russia",
    #     "Ukraine conflict timeline",
    #     "International response to Russia-Ukraine war",
    #     "Humanitarian crisis in Ukraine"
    # ]
    # query_results = []
    # # Evaluate search engine using test queries
    # for query in test_queries:
    #     print(f"Query: {query}")
    #     results = search_tf_idf(query, index, idf, tf)
    #     relevant_tweets = tweet_text[tweet_text["tweet_id"].isin(results)]
    #     print(relevant_tweets["text"])
    #     print("=" * 50)
    #     query_results.append(f"Query: {query}\n")
    #     query_results.extend(relevant_tweets["text"].tolist())
    #     query_results.append("=" * 50 + "\n")

    # # Save results to a text file
    # with open('search_results.txt', 'w', encoding='utf-8') as file:
    #     file.writelines(query_results)

main()
