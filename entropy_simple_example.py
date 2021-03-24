#! /usr/bin/env python3

import math
import string
import random

one_word_para = """dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude dude """
two_word_para = """two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes two dudes """
english_para = """This is a simple paragraph of English text We hope that there will
be many words that occur repeatedly.  The English language has many words and this paragraph might have many repetitions.  The hope is that there are enough repetitions.  So I ask: will the entropy be big or small?  The answer is that we will have to see what the program returns. """
# make them long
english_para *= 4
one_word_para *= 4
two_word_para *= 4

def main():
    n_words = len(english_para.split())
    n_chars = len(english_para)
    random_para = make_random_para(n_words, n_chars)
    para_list = [one_word_para, two_word_para, english_para, random_para]
    entropy_one_word = calc_word_entropy(one_word_para)
    entropy_two_words = calc_word_entropy(two_word_para)
    entropy_english = calc_word_entropy(english_para)
    entropy_random = calc_word_entropy(random_para)
    for para in para_list:
        H = calc_word_entropy(para)
        all_vars = globals().copy()
        all_vars.update(locals())
        # print(global_vars.items())
        # print(locals().items())
        my_var_name = [ k for k,v in all_vars.items() if v == para][0]
        print(f'{my_var_name} has entropy:   {H}')

def make_random_para(n_words, n_chars):
    para = ''
    done = False
    letters = string.ascii_lowercase
    while not done:
        wordlen = random.randint(1, 5)
        word = ''.join(random.choice(letters) for i in range(wordlen))
        if len(para) == 0:
            para = word
        else:
            para = para + ' ' + word
        if len(para) >= n_chars:
            done = True
    return para

def calc_word_entropy(para):
    para = para.lower()
    words = para.split()
    words = [word.strip('.') for word in words]
    words = [word.strip(',') for word in words]
    words = [word.strip(':') for word in words]
    words = [word.strip(';') for word in words]
    unique_words = list(set(words))    # make them unique
    word_dict = {}
    for word in unique_words:
        count = count_word_in_list(words, word)
        word_dict[word] = count
    entropy = 0
    for word in unique_words:
        count = word_dict[word]
        prob = count / len(words)
        word_surprise = - prob * math.log(prob)
        entropy += word_surprise
    return entropy

def count_word_in_list(l, w):
    count = 0
    for aword in l:
        if w == aword:
            count += 1
    return count


if __name__ == '__main__':
    main()
