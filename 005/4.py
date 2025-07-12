# 4. Напишите программу, которая будет находить самое длинное слово в заданной строке.
# string = "Каждый охотник желает знать, где сидит фазан."

import re

def get_max_length_string(string_list):
    max_length_word = ''
    max_length = 0
    for word in string_list:
        if len(word) > max_length:
            max_length_word = word
            max_length = len(word)
    return max_length_word




string = "Каждый охотник желает знать, где сидит фазан."

string_list = list(map(lambda i: i.strip('.,'), re.split(' ', string)))

print(get_max_length_string(string_list))



string = "Каждый охотник желает знать, где сидит фазан."

string_list = list(map(lambda i: i.strip('.,'), string.split()))

print(max(string_list,key=len))



