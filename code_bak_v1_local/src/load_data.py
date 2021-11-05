import random
import json
import copy
import re
import nltk
from src.data_utils import remove_brackets
from src.utils import is_equal, remove_bucket

# PAD_token = 0


def load_math23k_data(filename, full_mode=False): # load the json data to list(dict()) for MATH 23K
    # Math 23K format:
    # "id":"2",
    # "original_text":"一个工程队挖土，第一天挖了316方，从第二天开始每天都挖230方，连续挖了6天，这个工程队一周共挖土多少方？",
    # "segmented_text":"一 个 工程队 挖土 ， 第一天 挖 了 316 方 ， 从 第 二 天 开始 每天 都 挖 230 方 ， 连续 挖 了 6 天 ， 这个 工程队 一周 共 挖土 多少 方 ？",
    # "equation":"x=316+230*(6-1)",
    # "ans":"1466"
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            # 移除x和等号
            if not full_mode:
                data_d["equation"] = data_d["equation"][2:]
            data.append(data_d)
            js = ""
    f.close()
    return data


def load_math23k_data_from_json(filename, full_mode=False):
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "equation" not in d or "ans" not in d or d["ans"] == []:
            continue
        if "千米/小时" in d["equation"]:
            d["equation"] = d["equation"][:-5]
            # 移除x和等号
        if not full_mode:
            d["equation"] = d["equation"][2:]
        out_data.append(d)
    return out_data


def load_hmwp_data(filename):
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "equation" not in d or "ans" not in d or d["ans"] == []:
            continue
        x = d['equation']
        if len(set(x) - set("0123456789.+-*/^()=xXyY; ")) != 0:
            continue
        count1 = 0
        count2 = 0
        for elem in x:
            if elem == '(':
                count1 += 1
            if elem == ')':
                count2 += 1
        if count1 != count2:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['equation'] = new_eqs[0]
        else:
            d['equation'] = ' ; '.join(new_eqs)

        seg = d['original_text'].strip().split()
        new_seg = []
        for s in seg:
            if len(s) == 1 and s in ",.?!;":
                continue
            new_seg.append(s)
        d['original_text'] = ' '.join(new_seg)
        out_data.append(d)
    return out_data


def load_cm17k_data(filename):
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "equation" not in d or "ans" not in d or d["ans"] == []:
            continue
        x = d['equation']
        if len(set(x) - set("0123456789.+-*/^()=xXyY; ")) != 0:
            continue
        count1 = 0
        count2 = 0
        for elem in x:
            if elem == '(':
                count1 += 1
            if elem == ')':
                count2 += 1
        if count1 != count2:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['equation'] = new_eqs[0]
        else:
            d['equation'] = ' ; '.join(new_eqs)

        seg = d['original_text'].strip().split()
        new_seg = []
        for s in seg:
            if len(s) == 1 and s in ",.?!;":
                continue
            new_seg.append(s)
        d['original_text'] = ' '.join(new_seg)
        out_data.append(d)
    return out_data


def load_mawps_data(filename):
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "lEquations" not in d or len(d["lEquations"]) != 1:
            continue
        x = d["lEquations"][0].replace(" ", "")

        if "lQueryVars" in d and len(d["lQueryVars"]) == 1:
            v = d["lQueryVars"][0]
            if v + "=" == x[:len(v)+1]:
                xt = x[len(v)+1:]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

            if "=" + v == x[-len(v)-1:]:
                xt = x[:-len(v)-1]
                if len(set(xt) - set("0123456789.+-*/()")) == 0:
                    temp = d.copy()
                    temp["lEquations"] = xt
                    out_data.append(temp)
                    continue

        if len(set(x) - set("0123456789.+-*/()=xX")) != 0:
            continue

        if x[:2] == "x=" or x[:2] == "X=":
            if len(set(x[2:]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[2:]
                out_data.append(temp)
                continue
        if x[-2:] == "=x" or x[-2:] == "=X":
            if len(set(x[:-2]) - set("0123456789.+-*/()")) == 0:
                temp = d.copy()
                temp["lEquations"] = x[:-2]
                out_data.append(temp)
                continue
    return out_data


def load_alg514_data(filename):  # load the json data to list(dict()) for ALG514 data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    for d in data:
        if "lEquations" not in d:
            continue
        x = d['lEquations']
        if len(set(x) - set("0123456789.+-*/()=xXyY; ")) != 0:
            continue

        eqs = x.split(';')
        new_eqs = []
        for eq in eqs:
            sub_eqs = eq.split('=')
            new_sub_eqs = []
            for s_eq in sub_eqs:
                new_sub_eqs.append(remove_brackets(s_eq.strip()))
            new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
        if len(new_eqs) == 1:
            d['lEquations'] = new_eqs[0]
        else:
            d['lEquations'] = ' ; '.join(new_eqs)

        seg = d['sQuestion'].strip().split()
        new_seg = []
        for s in seg:
            if len(s) == 1 and s in ",.?!;":
                continue
            new_seg.append(s)
        d['sQuestion'] = ' '.join(new_seg)
        out_data.append(d)
    return out_data


def load_ape210k_data(filename, use_char=False):  # load the json data to list(dict()) for APE210k data
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    data = json.load(f)
    f.close()
    out_data = []
    max_len = 0
    for d in data:
        if use_char:
            question, equation, answer = d['segmented_text'], d['equation'], d['ans']
        else:
            question, equation, answer = d['jieba_segmented_text'], d['equation'], d['ans']

        # 处理带分数
        question = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', question)
        equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
        equation = re.sub('(\d+) \( (\d+) / (\d+) \)', '\\1(\\2/\\3)', equation)
        #equation = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', equation)
        #answer = re.sub('(\d+) \( (\d+ / \d+) \)', '\\1(\\2/\\3)', answer)
        equation = re.sub('(\d+) \(', '\\1(', equation)
        answer = re.sub('(\d+) \(', '\\1(', answer)
        # 分数去括号
        #question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 分数合并
        question = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', question)
        equation = re.sub('\( (\d+) / (\d+) \)', '(\\1/\\2)', equation)

        # 分数加括号
        #question = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', question)
        #equation = re.sub(' (\d+) / (\d+) ', ' (\\1/\\2) ', equation)
        # 处理百分数
        question = re.sub('([\.\d]+)%', '(\\1/100)', question)
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        question = question.replace('%', ' / 100')
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        equation = equation.replace('"千米/小时"', '')


        # 处理带分数
        # question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        # equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        # answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        # equation = re.sub('(\d+)\(', '\\1+(', equation)
        # answer = re.sub('(\d+)\(', '\\1+(', answer)
        # # 分数去括号
        # question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # # 处理百分数
        # equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        # answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # # 冒号转除号、剩余百分号处理
        # equation = equation.replace(':', '/').replace('%', '/100')
        # answer = answer.replace(':', '/').replace('%', '/100')

        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            if is_equal(eval(equation), eval(answer)):
                d['segmented_text'] = question
                d['equation'] = remove_bucket(equation)
                d['ans'] = answer
                # out_data.append((question, remove_bucket(equation), answer))
                out_data.append(d)

                if max_len < len(question):
                    max_len = len(question)
        except:
            continue

    print(max_len)
    return out_data