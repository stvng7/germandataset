import re
import math

#clean
def clean_text(text):
  text = re.sub(r'http[s]?://\S+', '', text)
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\s+', ' ', text)
  text = text.lower()
  return text.strip()

#remove stopwords
def remove_stopwords(text, stopwords):
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

#stemming
def stem_words(text):
    words = text.split()
    stemmed_words = []
    for word in words:
        if word.endswith('ing') and len(word) > 4:
            word = word[:-3] 
        elif word.endswith('ly') and len(word) > 3:
            word = word[:-2] 
        elif word.endswith('ment') and len(word) > 4:
            word = word[:-4]
        stemmed_words.append(word)
    return ' '.join(stemmed_words)

with open('stopwords.txt', 'r') as f:
    stopwords = set(f.read().split())

with open('tfidf_docs.txt', 'r') as f:
    doc_files = [line.strip() for line in f.readlines()]

for doc_file in doc_files:
    with open(doc_file, 'r') as f:
        text = f.read()
    text = clean_text(text)
    text = remove_stopwords(text, stopwords)
    text = stem_words(text)

    out_file = f'preproc_{doc_file}'
    with open(out_file, 'w') as f:
        f.write(text)
      
#----------------------------------------

def compute_tf(words):
    total = len(words)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return {w: counts[w] / total for w in counts}

def compute_idf(docs_words):
    num_docs = len(docs_words)
    idf = {}
    
    all_words = set()
    for words in docs_words:
        all_words.update(words)

    for word in all_words:
        doc_count = sum(1 for words in docs_words if word in words)
        idf[word] = math.log(num_docs / doc_count) + 1

    return idf

preproc_files = ["preproc_" + doc for doc in doc_files]

docs_words = []
doc_names = []

for file in preproc_files:
    with open(file, "r") as f:
        text = f.read().strip()

    words = text.split()
    docs_words.append(words)

    original_name = file.replace("preproc_", "")
    doc_names.append(original_name)

idf_scores = compute_idf([set(words) for words in docs_words])

for i, words in enumerate(docs_words):
    tf = compute_tf(words)
    tfidf = {word: round(tf[word] * idf_scores[word], 2) for word in tf}


    top5 = sorted(tfidf.items(), key=lambda x: (-x[1], x[0]))[:5]

    out_file = "tfidf_" + doc_names[i]
    with open(out_file, "w") as f:
        f.write(str(top5))


