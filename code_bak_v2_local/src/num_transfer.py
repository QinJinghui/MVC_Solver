from sympy import *
import re


def transfer_math23k_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split()
        #  seg =  d["segmented_text"].strip().split(" ")
        equations = d["equation"].replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')

        for s in seg:
            pos = re.search(pattern, s) # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        def seg_and_tag(st) : # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq: # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # 将答案转换为浮点数
        if '%' in d['ans']:
            ans = [float(d['ans'][:-1]) / 100]
        else:
            if '(' in d['ans']:
                new_ans = []
                for idx in range(len(d['ans'])):
                    if d['ans'][idx] == '(' and idx > 0 and d['ans'][idx-1].isdigit():
                        new_ans.append('+')
                        new_ans.append(d['ans'][idx])
                    elif d['ans'][idx] == ')' and idx < len(d['ans']) - 1 and d['ans'][idx+1].isdigit():
                        new_ans.append(d['ans'][idx])
                        new_ans.append('+')
                    else:
                        new_ans.append(d['ans'][idx])
                d['ans'] = ''.join(new_ans)
            ans = [float(eval(d['ans']))]
        # if len(input_seq) > 256:
        #     input_seq = input_seq[:256]
        # pairs.append((input_seq, out_seq, nums, num_pos, ans))
        id = d['id']
        type = 0
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans
        }
        pairs.append(data_dict)

    temp_g = []

    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len


def transfer_cm17k_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    max_id = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        if len(seg) > 200:
            continue
        equations = d["equation"].replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')

        for s in seg:
            pos = re.search(pattern, s)  # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            # if len(nums) > 20:
            #     continue
            copy_nums = len(nums)
            # max_id = d['id']

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                # float_nums.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
                float_nums_fraction.append(str(float(round(eval(num[:-1].strip()) / 100, 3))))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    # elif nums.count(n) > 1:
                    #     # 多个的时候默认使用第一个index代替
                    #     res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('[SEP]')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq

        idx = 0
        new_out_seq = []
        while idx < len(out_seq):
            if out_seq[idx] == '[SEP]':
                new_out_seq.append(out_seq[idx])
                idx += 1
                continue

            if idx + 1 < len(out_seq):
                if out_seq[idx][0] == 'N' and (out_seq[idx+1] in 'xyz(' or  out_seq[idx+1][0].isdigit()):
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                elif out_seq[idx][0] == ')' and out_seq[idx+1] not in '+-*/^=)SEP':
                    new_out_seq.append(out_seq[idx])
                    new_out_seq.append('*')
                else:
                    new_out_seq.append(out_seq[idx])
            else:
                new_out_seq.append(out_seq[idx])
            idx += 1
        out_seq = new_out_seq

        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        ans = d['ans']
        id = d['id']
        type = d['type']
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans,
        }
        pairs.append(data_dict)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums
    return pairs, temp_g, max_num_list_len


def transfer_mawps_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+")
    pairs = []
    generate_nums = {}  # 用于记录题目中蕴含的数字，并不是显式出现的数字
    copy_nums = 0  # 记录最长的数字列表长度
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["sQuestion"].strip().split(" ")
        equations = d["lEquations"].replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')

        for s in seg:
            pos = re.search(pattern, s)
            if pos:
                if pos.start() > 0:
                    input_seq.append(s[:pos.start()])
                num = s[pos.start(): pos.end()]
                nums.append(num.replace(",", ""))
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            copy_nums = len(nums)

        eq_segs = []
        temp_eq = ""
        for e in equations:
            if e == ' ':
                continue
            elif e not in "()+-*/^=xXyYzZ;":
                temp_eq += e
            elif temp_eq != "":
                # 检查方程中出现了问题数字的列表
                count_eq = []
                for n_idx, n in enumerate(nums):
                    if abs(float(n) - float(temp_eq)) < 1e-4:
                        count_eq.append(n_idx)
                        if n != temp_eq:
                            nums[n_idx] = temp_eq # 以等式的为准

                if len(count_eq) == 0: # 如果等式中的数字不在问题中
                    flag = True
                    for gn in generate_nums:
                        if abs(float(gn) - float(temp_eq)) < 1e-4:
                            generate_nums[gn] += 1
                            if temp_eq != gn:
                                temp_eq = gn
                            flag = False
                    if flag:
                        generate_nums[temp_eq] = 0
                    eq_segs.append(temp_eq)
                elif len(count_eq) == 1:
                    eq_segs.append("N" + str(count_eq[0]))
                elif len(count_eq) > 1:
                    eq_segs.append("N" + str(count_eq[0]))
                else:
                    eq_segs.append(temp_eq)
                eq_segs.append(e)
                temp_eq = ""
            else:
                eq_segs.append(e)
        # 如果方程等式项不为空，则进行再次检查
        if temp_eq != "":
            count_eq = []
            for n_idx, n in enumerate(nums):
                if abs(float(n) - float(temp_eq)) < 1e-4:
                    count_eq.append(n_idx)
                    if n != temp_eq:
                        nums[n_idx] = temp_eq
            # 不在问题的数字列表中
            if len(count_eq) == 0:
                flag = True
                for gn in generate_nums:
                    if abs(float(gn) - float(temp_eq)) < 1e-4:
                        generate_nums[gn] += 1
                        if temp_eq != gn:
                            temp_eq = gn
                        flag = False
                if flag:
                    generate_nums[temp_eq] = 0
                eq_segs.append(temp_eq)
            elif len(count_eq) == 1:
                eq_segs.append("N" + str(count_eq[0]))
            else:
                eq_segs.append(temp_eq)

        # 找出数字在问题出现的位置
        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        if len(nums) != 0:
            answers = d["lSolutions"]
            ans = []
            for a in answers:
                ans.append(float(a))

            id = d['iIndex']
            type = 0
            data_dict = {
                "id": id,
                "type": type,
                "input_seq": input_seq,
                "out_seq": eq_segs,
                "nums": nums,
                "num_pos": num_pos,
                "ans": ans,
            }
            pairs.append(data_dict)

    # 构建需要频繁常识数字列表
    temp_g = []
    for g in generate_nums:
        if generate_nums[g] >= 5:  # 数字g在数据集中出现的次数超过5次
            temp_g.append(g)

    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len


def transfer_hmwp_num(data_list):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split()
        equations = d["equation"].replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')

        for s in seg:
            pos = re.search(pattern, s) # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        #if len(input_seq) > 384:
        #    continue

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            elif len(num) > 1 and num[0] == '0':
                float_nums.append(str(float(eval(num[1:].strip()))))
            else:
                float_nums.append(str(float(eval(num.strip()))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num.strip()))))
            elif '%' in num:
                float_nums.append(str(float(eval(num[:-1].strip()) / 100)))
            else:
                float_nums_fraction.append(str(float(eval(num.strip()))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('[SEP]')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq
        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        ans = d['ans']
        # if len(input_seq) > 384:
        #    input_seq = input_seq[:384]
        #    continue
        id = d['id']
        type = 0
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans,
        }
        pairs.append(data_dict)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len


def transfer_alg514_num(data_list): # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data_list:
        nums = []
        input_seq = []
        seg = d['sQuestion'].strip().split()
        equations = d['lEquations']

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append('[NUM]')
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        # print(nums)
        # print(nums_fraction)
        float_nums = []
        for num in nums:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums.append(str(float(eval(num))))
            else:
                float_nums.append(str(float(eval(num))))

        float_nums_fraction = []
        for num in nums_fraction:
            if ',' in num:
                new_num = []
                for c in num:
                    if c == ',':
                        continue
                    new_num.append(c)
                num = ''.join(new_num)
                float_nums_fraction.append(str(float(eval(num))))
            else:
                float_nums_fraction.append(str(float(eval(num))))
        # print(float_nums)
        # print(float_nums_fraction)
        nums = float_nums
        nums_fraction = float_nums_fraction

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    elif nums.count(n) > 1:
                        # 多个的时候默认使用第一个index代替
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                elif nums.count(st_num) > 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        new_out_seq = []
        for seq in out_seq:
            if seq == ' ' or seq == '':
                continue
            if seq == ';':
                new_out_seq.append('[SEP]')
                continue
            new_out_seq.append(seq)
        out_seq = new_out_seq
        # print(equations)
        # print(' '.join(out_seq))
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # print(nums, num_pos)
        # if len(nums) == 0:
        #     print(d['iIndex'])
        if len(input_seq) > 256:
            input_seq = input_seq[:256]

        ans = d['lSolutions']
        id = d['iIndex']
        type = 0
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans,
        }
        pairs.append(data_dict)

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_ape210k_num(data_list):  # transfer num into "[NUM]"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    ops_list = ["+", "-", "*", "/", "^"]
    for d in data_list:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split()
        #  seg =  d["segmented_text"].strip().split(" ")
        equations = d["equation"].replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')

        # check equation
        is_continue = False
        for i in range(len(equations) - 1):
            if equations[i] in ops_list and equations[i+1] in ops_list:  # 连续两个符号，跳过
                is_continue = True
                break
        if is_continue:
            continue

        print(d['id'])
        print(seg)
        print(equations)

        for s in seg:
            pos = re.search(pattern, s)  # 搜索每个词的数字位置
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("[NUM]")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        # if len(input_seq) > 384:
        #     continue

        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True) # 从大到小排序

        def seg_and_tag(st): # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    # if nums.count(n) == 1:
                    if nums.count(n) >= 1:  # relax
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)

                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
                elif n[0] == '(' and n[-1] == ')' and n[1:-1] in st:
                    nn = n[1:-1]
                    p_start = st.find(nn)
                    p_end = p_start + len(nn)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    # if nums.count(n) == 1: # relax
                    if nums.count(n) >= 1:
                        res.append("N" + str(nums.index(n)))
                    else:
                        res.append(n)

                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res

            pos_st = re.search("\d+\.\d+%?|\d+%?", st) # 带百分号的数字数
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                # if nums.count(st_num) == 1:
                if nums.count(st_num) >= 1: # relax
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq: # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "[NUM]":
                num_pos.append(i)
        assert len(nums) == len(num_pos)

        print(input_seq)
        print(out_seq)
        # 将答案转换为浮点数
        if '%' in d['ans']:
            ans = [float(d['ans'][:-1]) / 100]
        else:
            if '(' in d['ans']:
                new_ans = []
                for idx in range(len(d['ans'])):
                    if d['ans'][idx] == '(' and idx > 0 and d['ans'][idx-1].isdigit():
                        new_ans.append('+')
                        new_ans.append(d['ans'][idx])
                    elif d['ans'][idx] == ')' and idx < len(d['ans']) - 1 and d['ans'][idx+1].isdigit():
                        new_ans.append(d['ans'][idx])
                        new_ans.append('+')
                    else:
                        new_ans.append(d['ans'][idx])
                d['ans'] = ''.join(new_ans)
            ans = [float(eval(d['ans']))]
        # if len(input_seq) > 256:
        #     input_seq = input_seq[:256]
        # pairs.append((input_seq, out_seq, nums, num_pos, ans))
        id = d['id']
        type = 0
        data_dict = {
            "id": id,
            "type": type,
            "input_seq": input_seq,
            "out_seq": out_seq,
            "nums": nums,
            "num_pos": num_pos,
            "ans": ans
        }
        pairs.append(data_dict)

    temp_g = []

    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    max_num_list_len = copy_nums

    return pairs, temp_g, max_num_list_len








