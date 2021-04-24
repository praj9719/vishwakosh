from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

import os
import numpy as np
import math
import json

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import spacy
from gensim.summarization.bm25 import BM25
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import itertools
import operator

import constants as c
from flask import jsonify


class PreProcessor:

    def __init__(self, data):
        self.data = data;

    def execute(self):
        self.convert_lower_case()
        self.remove_punctuation()  # remove comma seperately
        self.remove_apostrophe()
        self.remove_stop_words()
        self.convert_numbers()
        self.stemming()
        self.remove_punctuation()
        self.convert_numbers()
        self.stemming()  # needed again as we need to stem the words
        self.remove_punctuation()  # needed again as num2word is giving few hypens and commas fourty-one
        self.remove_stop_words()  # needed again as num2word is giving stop words 101 - one hundred and one
        return self.data

    def convert_lower_case(self):
        self.data = np.char.lower(self.data)

    def remove_stop_words(self):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(self.data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        self.data = new_text

    def remove_punctuation(self):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(self.data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        self.data = data

    def remove_apostrophe(self):
        self.data = np.char.replace(self.data, "'", "")

    def stemming(self):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(self.data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        self.data = new_text

    def convert_numbers(self):
        tokens = word_tokenize(str(self.data))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        self.data = new_text


class DocumentRetrieval:

    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        json_path = f'{model_path}.json'
        npy_path = f'{model_path}.npy'
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        self.N = data['N']
        self.topics = data['topics']
        self.DF = data['DF']
        self.total_vocab = data['total_vocab']
        self.D = np.load(npy_path)
        print(f"[Info] model loaded")

    def doc_freq(self, word):
        c = 0
        try:
            c = self.DF[word]
        except:
            pass
        return c

    def gen_vector(self, tokens):
        Q = np.zeros((len(self.total_vocab)))
        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = self.doc_freq(token)
            idf = math.log((self.N + 1) / (df + 1))
            try:
                ind = self.total_vocab.index(token)
                Q[ind] = tf * idf
            except:
                pass
        return Q

    def cosine_sim(self, a, b):
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cos_sim

    def classify(self, k, query):
        preprocessed_query = PreProcessor(query).execute()
        tokens = word_tokenize(str(preprocessed_query))

        d_cosines = []
        query_vector = self.gen_vector(tokens)

        for d in self.D:
            sim = self.cosine_sim(query_vector, d)
            d_cosines.append(sim)

        out = np.array(d_cosines).argsort()[-k:][::-1]
        result = []
        for o in out:
            result.append({"topic": self.topics[o], "similarity": round(d_cosines[o] + 0.0001, 4)})
        return result

    def search(self, query):
        count = min(self.N, 7)
        relevent_docs = self.classify(count, query)
        threshold = 0.02
        docs = []
        for doc in relevent_docs:
            if doc['similarity'] > threshold:
                with open(f'{self.dataset_path}/{doc["topic"]}.txt', 'r', encoding="utf8") as file:
                    docs.append(file.read())
        return docs


class PassageRetrieval:

  def __init__(self, nlp):
    self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
    self.bm25 = None
    self.passages = None

  def preprocess(self, doc):
    passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
    return passages

  def fit(self, docs):
    passages = list(itertools.chain(*map(self.preprocess, docs)))
    corpus = [self.tokenize(p) for p in passages]
    self.bm25 = BM25(corpus)
    self.passages = passages

  def most_similar(self, question, topn=10):
    tokens = self.tokenize(question)
    average_idf = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys()))
    scores = self.bm25.get_scores(tokens, average_idf)

    pairs = [(s, i) for i, s in enumerate(scores)]
    pairs.sort(reverse=True)
    passages = [self.passages[i] for _, i in pairs[:topn]]
    return passages


class AnswerExtractor:

  def __init__(self, tokenizer, model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModelForQuestionAnswering.from_pretrained(model)
    self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

  def extract(self, question, passages):
    answers = []
    for passage in passages:
      try:
        answer = self.nlp(question=question, context=passage)
        answer['text'] = passage
        answers.append(answer)
      except KeyError:
        pass
    answers.sort(key=operator.itemgetter('score'), reverse=True)
    return answers


SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])


class Bot:
    def __init__(self, bot):
        self.passage_retriever = PassageRetrieval(nlp)
        self.answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

        self.bot_name = bot['name']
        self.bot = DocumentRetrieval(bot['database'], bot['model'])

    def answer(self, document_retriever, question):
        ans = []
        docs = document_retriever.search(question)
        if len(docs) < 1:
            ans.append({"short": "Out of context", "long": "Unable to find appropriate answer for this question!"})
            return ans
        self.passage_retriever.fit(docs)
        passages = self.passage_retriever.most_similar(question)
        answers = self.answer_extractor.extract(question, passages)
        for i in range(min(len(answers), 3)):
            ans.append({"short": answers[i]["answer"], "long": answers[i]["text"]})
        return ans

    def execute(self, request):
        if request.method == 'POST':
            input_data = request.get_json()
            if input_data:
                if c.input_text in input_data:
                    string = input_data[c.input_text]
                    output = self.answer(self.bot, string)
                    output_data = {c.status: c.status_success,
                                   c.title: self.bot_name,
                                   c.info: c.info_normal,
                                   c.data: {c.message: output}}
                    return jsonify(output_data)
                else:
                    output_data = {c.status: c.status_failed,
                                   c.title: "reverse string",
                                   c.info: f"Expecting '{c.input_text}' as input!"}
                    return jsonify(output_data)
            else:
                output_data = {c.status: c.status_failed,
                               c.title: "reverse string",
                               c.info: f"Expecting '{c.input_text}' as input in json format!"}
                return jsonify(output_data)
        else:
            output_data = {c.status: c.status_failed,
                           c.title: "reverse string",
                           c.info: f"Expecting POST method with '{c.input_text}' as input in json format!"}
            return jsonify(output_data)

