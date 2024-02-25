#encoding:utf8
import jieba
import re


class DataProcessor(object):
    re_alphabet = "[a-zA-Z]+"
    re_digit = "[0-9]+"

    def segment_words(self, data):
        seg_words = jieba.lcut(data)

        words = []
        for x in seg_words:
            if len(x) < 2:
                continue
            if re.match(self.re_digit, x):
                continue
            if re.match(self.re_alphabet, x):
                continue
            words.append(x)
        return words

    def split_sentence(self, data):
        return re.split("。|！|？", data)