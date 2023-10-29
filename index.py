
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
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
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


def generate_ranking(query, index, tf, idf):
    # TODO: Generate rankings
    return 0

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

def rank_documents(terms, docs, index, idf, tf, tweet_ids):
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

    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would become 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
   
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

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    
    result_docs = [x[1] for x in doc_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tf_idf(query, index)
    #print ('\n'.join(result_docs), '\n')
    return result_docs[:10]


def search_tf_idf(query, index, idf, tf, tweet_ids):
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
    ranked_docs = rank_documents(query, docs, index, idf, tf, tweet_ids)
    #print( ranked_docs)
    return ranked_docs

def select_docs(data, query_id):
    subset = []

    for line in data:
        doc, q_id, label = line.split(',')
        if q_id == query_id:
            subset.append(doc)
        elif label == '1':
            subset.append(doc)

    return subset

def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Salta el encabezado
        data = [",".join(row) for row in reader]
    return data

def main():
    file_path = ''
    start_time = time.time()
    docs_path = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/Rus_Ukr_war_data.json'
    ev1 = 'C:/Users/2002d/OneDrive/Documentos/UPF/2023-2024/1st Term/Information Retrieval and Web Analysis/Project/IRWA_data_2023/Evaluation_gt.csv'
    evaluation_data1 = read_csv(ev1)
    with open(docs_path) as fp:
        lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]
    print("There are ", len(lines), " tweets")
    
    # Process lines to create a list of tweet IDs
    tweet_ids = [json.loads(line)["id"] for line in lines]
    tweets_texts = [json.loads(line)["full_text"] for line in lines]
    tweet_text = pd.DataFrame({'tweet_id': tweet_ids, 'text': tweets_texts})
    index, tf, df, idf = create_index(lines)

    baseline_queries = [
        "Tank Kharkiv",
        "Nord Stream pipeline",
        "Annexation territories Russia"
    ]
    docs_Q1 = select_docs(evaluation_data1,"Q1")
    docs_Q2 = select_docs(evaluation_data1,"Q2")
    docs_Q3 = select_docs(evaluation_data1,"Q3")

    print(f"Query: {baseline_queries[0]}")
    results = search_tf_idf(baseline_queries[0], index, idf, tf)
    relevant_tweets = tweet_text[tweet_text["tweet_id"].isin(results)]
    print(relevant_tweets["text"])
    print("=" * 50)

 
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




# def vector_index(lines):
#     """
#     input: list of paragraphs
#     output: dataframe mapping each paragraph to its embedding
#     """
#     # from sklearn.cluster import AgglomerativeClustering
#     embeddings = model.encode(lines)
#     df = pd.DataFrame(
#         {"tweet": lines[i]["id"], "vector_representation": embeddings[i]}
#         for i in range(len(embeddings))
#     )
#     return df

# def obtain_similarity(query, df, k):
#     """
#     arguments:
#         - query: word or sentence to compare
#         - df: dataframe mapping paragraphs to embeddings
#         - k: number of selected similar paragraphs
#     output: list of paragraphs relevant for the query and the position in the datframe at which they are
#     """

#     query_embedding = model.encode(query)
#     df["similarity"] = df["vector_representation"].apply(
#         lambda x: cosine_similarity(x, query_embedding)
#     )
#     results = df.sort_values("similarity", ascending=False, ignore_index=True)
#     top_k = results["tweet"][1:k]
#     top_k = list(top_k)
#     ## Find positions of the top_k in df
#     positions = df.loc[df["tweet"].isin(top_k)].index
#     return top_k, positions

# def calculate_tf_idf(index):
#     tf_idf_scores = {}
#     total_tweets = len(index.keys())

#     tf = defaultdict(list)
#     df = defaultdict(int)  

#     # Calculate IDF for each term
#     idf = {term: np.log(total_tweets / len(postings)) for term, postings in index.items()}
#     norm = 0
#     for term, posting in index.items():
#         # posting will contain the list of positions for current term in current document.
#         # posting ==> [current_doc, [list of positions]]
#         # you can use it to infer the frequency of current term.
#         norm += len(posting[1]) ** 2
#     norm = math.sqrt(norm)
    
#     for term, posting in index.items():
#             # append the tf for current term (tf = term frequency in current doc/norm)
#             tf[term].append(np.round(len(posting[1]) / norm, 4)) ## SEE formula (1) above
#             #increment the document frequency of current term (number of documents containing the current term)
#             df[term] += 1 # increment DF for current term


#     # Calculate TF-IDF scores for each term in each tweet
#     for term, postings in index.items():
#         tf_idf_scores[term] = {}
#         for posting in postings:
#             tweet_id, positions = posting[0], posting[1]
#             tf = len(positions)
#             tf_idf = tf * idf[term]
#             tf_idf_scores[term][tweet_id] = tf_idf

#     return tf_idf_scores
