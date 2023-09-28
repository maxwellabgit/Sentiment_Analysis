#this is a quick draft of possible new projects that can become larger commits later on.

import transformers
from transformers import pipeline

#sentiment analysis
classifier = pipeline(
  "sentiment-analysis",
  model="distilbert-base-uncased-finetuned-sst-2-english")
input_text = ""
classifier(input_text)

#named entity recognition
input_text = ""
ner = pipeline("ner",
               model="dmdz/bert-large-cased-finetuned-conll03-english",
               grouped_entities=True)
ner("input_text")


#text generation
input_text = ""
generator = pipeline("text-generation",
                     model="gpt2")
generator(input_text, max_length=100)


#text summarization
import warnings
warnings.filterwarnings("ignore")

input_text = ""
summarizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-12-6")
summarizer(input_text, min_length=40)


#Question answering
question_answerer = pipeline("question-answering")
question_answerer(
  question="example question",
  context="""
      example context, super long"""
  )


#unmasking
#top_k is the amount of possible results to print
unmasker = pipeline("fill-mask",
                    model="distilroberta-base")
unmasker("this is <mask> you put the example text.", top_k=2)


