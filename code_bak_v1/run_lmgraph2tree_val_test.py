# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.load_data import *
from src.num_transfer import *
from src.expression_tree import *
import json
from src.log_utils import *
from src.data_utils import *
from src.calculation import *
from transformers import BertTokenizer, AdamW

USE_CUDA = torch.cuda.is_available()
batch_size = 32
grad_acc_steps = 4  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
embedding_size = 128
hidden_size = 768
n_epochs = 80
bert_learning_rate = 5e-5
bert_path = "./pretrained_lm/chinese-bert-wwm"
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
fold_num = 5
drop_out = 0.5
random_seed = 1
beam_search = True
dataset_name = "Math23K"
ckpt_dir = "Math23K"
data_dir = "./dataset/math23k/"
data_path = data_dir + "Math_23K.json"
prefix = '23k_processed.json'
group_data_path = data_dir + "Math_23K_processed.json"
var_nums = []

if dataset_name == "Math23K":
    var_nums = []
    ckpt_dir = "Math23K_bg2t_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K.json"
    prefix = '23k_processed.json'
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    group_data_path = data_dir + "Math_23K_processed.json"
elif dataset_name == "Math23K_char":
    ckpt_dir = "Math23K_char_bg2t_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K_char.json"
    prefix = '23k_processed.json'
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    group_data_path = data_dir + "Math_23K_char_processed.json"

ckpt_dir = ckpt_dir + '_' + str(n_epochs) + '_' + str(batch_size) + '_' + str(embedding_size) + '_' + str(hidden_size)
if beam_search:
    ckpt_dir = ckpt_dir + '_' + 'beam_search'

save_dir = os.path.join("./models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

pairs = None
generate_nums = None
copy_nums = None
group_data = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
    group_data = read_json(group_data_path)
elif dataset_name == "Math23K_char":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
    group_data = read_json(group_data_path)

temp_pairs = []
for p, g in zip(pairs, group_data):
    ept = ExpressionTree()
    ept.build_tree_from_infix_expression(p["out_seq"])
    p['out_seq'] = ept.get_prefix_expression()
    p['group_num'] = g['group_num']
    temp_pairs.append(p)
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(data_dir, prefix, data, pairs)

last_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

pairs_tested = test_fold
# pairs_valid = valid_fold
# train_fold.extend(valid_fold)
pairs_trained = train_fold

random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)
torch.manual_seed(random_seed)  # cpu
if USE_CUDA:
    torch.cuda.manual_seed(random_seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_tokenizer.add_tokens(['[NUM]'])


_, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True, use_lm=True, use_group_num=True)

# Initialize models
encoder = PLMGraphMeanEncoderSeq(model_path=bert_path)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                     input_size=len(generate_nums) + len(var_nums), dropout=drop_out)
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                        embedding_size=embedding_size, dropout=drop_out)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=drop_out)

# the embedding layer is  only for generated number embeddings, operators, and paddings
encoder_bert_optimizer = AdamW(encoder.model.parameters(), lr=bert_learning_rate, weight_decay=weight_decay)
encoder_graph_optimizer = torch.optim.Adam(encoder.gcn.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_bert_scheduler = torch.optim.lr_scheduler.StepLR(encoder_bert_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
encoder_graph_scheduler = torch.optim.lr_scheduler.StepLR(encoder_graph_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

var_num_ids = []
for var in var_nums:
    if var in output_lang.word2index.keys():
        var_num_ids.append(output_lang.word2index[var])

best_val_acc = 0
best_equ_acc = 0
current_best_val_acc = (0, 0, 0)
for epoch in range(n_epochs):
    loss_total = 0
    random.seed(epoch+random_seed)  # for reproduction

    batches_dict = prepare_data_batch(train_pairs, batch_size, lm_tokenizer=bert_tokenizer, use_lm=True, use_group_num=True)

    id_batches = batches_dict['id_batches']
    input_batches = batches_dict['input_batches']
    input_lengths = batches_dict['input_lengths']
    attention_mask_batches = batches_dict['attention_mask_batches']
    token_type_ids_batches = batches_dict['token_type_ids_batches']
    output_batches = batches_dict['output_batches']
    output_lengths = batches_dict['output_lengths']
    nums_batches = batches_dict['nums_batches']
    num_stack_batches = batches_dict['num_stack_batches']
    num_pos_batches = batches_dict['num_pos_batches']
    num_size_batches = batches_dict['num_size_batches']
    ans_batches = batches_dict['ans_batches']
    graph_batches = batches_dict['num_graph_batches']

    logs_content = "epoch: {}".format(epoch + 1)
    add_log(log_file, logs_content)
    start = time.time()
    for idx in range(len(input_lengths)):
        step_size = len(input_batches[idx]) // grad_acc_steps
        for step in range(grad_acc_steps):
            start_idx = step * step_size
            end_idx = (step + 1) * step_size
            if step_size == 0:
                end_idx = len(input_batches[idx])

            if step == grad_acc_steps - 1:
                grad_acc = False
            else:
                grad_acc = True

            if step == 0:
                zero_grad = True
            else:
                zero_grad = False

                loss = train_lmgraph2tree(input_batches[idx][start_idx:end_idx],
                                          input_lengths[idx][start_idx:end_idx],
                                          attention_mask_batches[idx][start_idx:end_idx],
                                          token_type_ids_batches[idx][start_idx:end_idx],
                                          output_batches[idx][start_idx:end_idx],
                                          output_lengths[idx][start_idx:end_idx],
                                          num_stack_batches[idx][start_idx:end_idx],
                                          num_size_batches[idx][start_idx:end_idx],
                                          generate_num_ids, encoder, predict, generate, merge,
                                          encoder_bert_optimizer, encoder_graph_optimizer,
                                          predict_optimizer, generate_optimizer, merge_optimizer, output_lang,
                                          num_pos_batches[idx][start_idx:end_idx],
                                          graph_batches[idx][start_idx:end_idx],
                                          grad_acc=grad_acc, zero_grad=zero_grad, grad_acc_steps=grad_acc_steps,
                                          var_nums=var_num_ids)
                loss_total += loss

    encoder_bert_scheduler.step()
    encoder_graph_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    logs_content = "loss: {}".format(loss_total / len(input_lengths))
    add_log(log_file, logs_content)
    logs_content = "training time: {}".format(time_since(time.time() - start))
    add_log(log_file, logs_content)
    logs_content = "--------------------------------"
    add_log(log_file, logs_content)
    if epoch % 1 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        answer_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            # print(test_batch)
            # pairs: (id, input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack, ans, group)
            # input_batch, input_length,group,num_value,num_pos
            tokens_dict = bert_tokenizer(' '.join(test_batch['input_cell']))
            raw_input_ids = tokens_dict["input_ids"]
            tokens_dict["input_ids"] = []
            for t_id in raw_input_ids:
                if t_id == len(bert_tokenizer.vocab):
                    tokens_dict["input_ids"].append(1)
                else:
                    tokens_dict["input_ids"].append(t_id)

            num_pos = []
            for idx, t_id in enumerate(tokens_dict["input_ids"]):
                if t_id == 1:
                    num_pos.append(idx)

            # num_pos = []
            # for idx, t_id in enumerate(tokens_dict["input_ids"]):
            #     if t_id == bert_tokenizer.vocab['num']:
            #         num_pos.append(idx)

            input_batch = tokens_dict["input_ids"]
            input_length = tokens_dict["input_ids"].index(bert_tokenizer.vocab['[SEP]'])+1

            group_batch = []
            group_num = test_batch["group_num"]
            input_seq = test_batch['input_cell']
            pattern = re.compile("\[NUM]")
            # input_seg = []
            # for idx, s in enumerate(test_batch['input_cell']):
            #     pos = re.search(pattern, s)  # 搜索每个词的数字位置
            #     if pos and idx in group_num:
            #         input_seg.append(s)
            #         group_batch.append(len(input_seg)-1 + 1)  # 补偿CLS
            #     else:
            #         for ss in s:
            #             input_seg.append(ss)
            #             if idx in group_num:
            #                 group_batch.append(len(input_seg)-1 + 1)  # 补偿CLS

            # update group_num
            acc_count = 0
            temp_input_seq = []
            for idx, s in enumerate(input_seq):
                if s in ['',  '',  '', '', '', '', '', '']:
                    updated_group_num = []
                    for g_idx in group_num:
                        if g_idx == idx - acc_count:
                            continue
                        elif g_idx > idx - acc_count:
                            updated_group_num.append(g_idx - 1)
                        else:
                            updated_group_num.append(g_idx)
                    acc_count += 1
                    group_num = updated_group_num
                else:
                    if s != '' and s != '' and s != ' ':
                        temp_input_seq.append(s)
            input_seq = temp_input_seq

            input_seg = []
            seq_mapping = {}
            for idx, s in enumerate(input_seq):
                pos = re.search(pattern, s)  # 搜索每个词的数字位置
                if pos and idx in group_num:
                    input_seg.append(s)
                    seq_mapping[idx] = [len(input_seg)-1 + 1]
                else:
                    seq_mapping[idx] = []
                    # 利用tokenizer来校正group_num
                    lm_s = bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.encode(s)[1:-1])
                    for ss in lm_s:
                        input_seg.append(ss)
                        if idx in group_num:
                            seq_mapping[idx].append(len(input_seg)-1 + 1)
                            # new_group_num.append(len(input_seg)-1 + 1)  # 补偿CLS

            for idx in group_num:
                if idx < len(input_seq):
                    group_batch.extend(seq_mapping[idx])

            num_value_batch = test_batch['nums']
            num_pos_batch = num_pos
            batch_graph = get_single_example_graph(input_batch, input_length, group_batch, num_value_batch, num_pos_batch)

            test_res = evaluate_lmgraph2tree(tokens_dict["input_ids"], len(tokens_dict["input_ids"]), tokens_dict["attention_mask"], tokens_dict["token_type_ids"], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, num_pos, batch_graph, beam_size=beam_size, beam_search=beam_search,
                                        var_nums=var_num_ids)

            # batch_graph = get_single_example_graph(test_batch[1], test_batch[2], test_batch[9], test_batch[5], test_batch[6])
            # test_res = evaluate_graph2tree(test_batch[1], test_batch[2], generate_num_ids, encoder, predict, generate,
            #                          merge, output_lang, test_batch[6], batch_graph, beam_size=beam_size,
            #                         beam_search=beam_search, var_nums=var_num_ids)
            # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            try:
                val_ac, equ_ac, ans_ac, test_res_result, test_tar_result = compute_equations_result(test_res, test_batch[3], output_lang, test_batch[5], test_batch[7],
                                                                                                    ans_list=test_batch[8], tree=True, prefix=True)
                # print(test_res_result, test_tar_result)
            except Exception as e:
                # traceback.print_exc()
                # print(e)
                val_ac, equ_ac, ans_ac = False, False, False
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            if ans_ac:
                answer_ac += 1
            eval_total += 1
        logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
        add_log(log_file, logs_content)
        logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
        add_log(log_file, logs_content)
        logs_content = "testing time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "------------------------------------------------------"
        add_log(log_file, logs_content)
        all_acc_data.append((epoch,equation_ac, value_ac, eval_total))

        torch.save(encoder.state_dict(), os.path.join(save_dir, "graph2tree_encoder"))
        torch.save(predict.state_dict(), os.path.join(save_dir, "graph2tree_predict"))
        torch.save(generate.state_dict(), os.path.join(save_dir, "graph2tree_generate"))
        torch.save(merge.state_dict(),  os.path.join(save_dir, "graph2tree_merge"))
        if best_val_acc < value_ac:
            best_val_acc = value_ac
            current_best_val_acc = (equation_ac, value_ac, eval_total)
            torch.save(encoder.state_dict(), os.path.join(save_dir, "graph2tree_encoder_best_val_acc"))
            torch.save(predict.state_dict(), os.path.join(save_dir, "graph2tree_predict_best_val_acc"))
            torch.save(generate.state_dict(), os.path.join(save_dir, "graph2tree_generate_best_val_acc"))
            torch.save(merge.state_dict(),  os.path.join(save_dir, "graph2tree_merge_best_val_acc"))
        if best_equ_acc < equation_ac:
            best_equ_acc = equation_ac
            torch.save(encoder.state_dict(), os.path.join(save_dir, "graph2tree_encoder_best_equ_acc"))
            torch.save(predict.state_dict(), os.path.join(save_dir, "graph2tree_predict_best_equ_acc"))
            torch.save(generate.state_dict(), os.path.join(save_dir, "graph2tree_generate_best_equ_acc"))
            torch.save(merge.state_dict(),  os.path.join(save_dir, "graph2tree_merge_best_equ_acc"))
        if epoch == n_epochs - 1:
            last_acc_fold.append((equation_ac, value_ac, eval_total))
            best_val_acc_fold.append(current_best_val_acc)

a, b, c = 0, 0, 0
for bl in range(len(last_acc_fold)):
    a += last_acc_fold[bl][0]
    b += last_acc_fold[bl][1]
    c += last_acc_fold[bl][2]
    logs_content = "{}".format(last_acc_fold[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold)):
    a += best_val_acc_fold[bl][0]
    b += best_val_acc_fold[bl][1]
    c += best_val_acc_fold[bl][2]
    logs_content = "{}".format(best_val_acc_fold[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

logs_content = "{}".format(all_acc_data)
add_log(log_file, logs_content)