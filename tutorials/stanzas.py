# coding=utf-8
import json

import stanza


def json_load(file):
    with open(file, encoding='utf-8') as fin:
        return json.load(fin)


def json_dump(obj, file, ensure_ascii=False):
    with open(file, 'w') as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii)


nlp = stanza.Pipeline('en')
print(nlp)

file = '/Users/andersc/works/projects/labs/langs/llmbook_new_words_raw.json'
file_out = '/Users/andersc/works/projects/labs/langs/llmbook_new_words_normed.json'
words_raw = json_load(file)

words_normed = []
for word_line in words_raw:
    doc = nlp(word_line)
    if len(doc.sentences[0].words) != 1:
        # print('not simple word:', word_line)
        words_normed.append(word_line)
        continue

    for word in doc.sentences[0].words:
        if word.text != word.lemma:
            print(word.text, '->', word.lemma)
            words_normed.append(word.lemma)
        else:
            words_normed.append(word.text)

json_dump(words_normed, file_out)
