# coding: utf-8
from src.mv_train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.load_data import *
from src.num_transfer import *
from src.expression_tree import *
from src.log_utils import *
from src.calculation import *
from src.data_utils import read_json, get_pretrained_embedding_weight
from transformers import BertTokenizer, AdamW
# from src.expressions_transfer import *
import argparse

# USE_CUDA = torch.cuda.is_available()
# batch_size = 16
# grad_acc_steps = 8  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
# embedding_size = 128  # if pretrained_word2vec is true, the size should be equal to the dims of word2vec
# hidden_size = 768
# n_epochs = 80
# bert_learning_rate = 2e-5
# bert_path = "./pretrained_lm/chinese-bert-wwm"
# learning_rate = 1e-3
# weight_decay = 1e-5
# beam_size = 5
# beam_search = True
# n_layers = 2
# drop_out = 0.5
# random_seed = 1
# dataset_name = "Math23K"
# ckpt_dir = "Math23K"
# data_dir = "./dataset/math23k/"
# data_path = data_dir + "Math_23K.json"
# prefix = '23k_processed.json'
# var_nums = []
# use_share_decoder = True

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--grad_acc_steps', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--n_epochs', type=int, default=80)
parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
parser.add_argument('--bert_path', type=str, default="./pretrained_lm/chinese-bert-wwm")
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=2e-5)

parser.add_argument('--enable_beam_search', action='store_true')
parser.add_argument('--beam_size', type=int, default=5)

# parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--use_teacher_forcing', type=float, default=1.0)
parser.add_argument('--use_clip', action='store_true')
parser.add_argument('--gclip', type=float, default=0.0)
parser.add_argument('--dataset_name', type=str, default='Math23K')
parser.add_argument('--use_share_decoder', action='store_true')
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
batch_size = args.batch_size  # 16
grad_acc_steps = args.grad_acc_steps  # 8  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
embedding_size = args.embedding_size  # 128
hidden_size = args.hidden_size  # 768
n_epochs = args.n_epochs  # 80
bert_learning_rate = args.bert_learning_rate  # 5e-5
bert_path = args.bert_path  # "./pretrained_lm/chinese-bert-wwm"
learning_rate = args.learning_rate  # 1e-3
weight_decay = args.weight_decay  # 2e-5
beam_size = args.beam_size  # 5
use_teacher_forcing = args.use_teacher_forcing  # 1.0
use_clip = args.use_clip
gclip = args.gclip  # 0
beam_search = args.enable_beam_search  # True
# n_layers = args.n_layers  # 1
drop_out = args.drop_out  # 0.5
random_seed = args.random_seed  # 1
dataset_name = args.dataset_name  # "Math23K"
ckpt_dir = "Math23K"
data_dir = "./dataset/math23k/"
data_path = data_dir + "Math_23K.json"
prefix = '23k_processed.json'
var_nums = []
use_share_decoder = args.use_share_decoder

if dataset_name == "Math23K":
    ckpt_dir = "Math23K_mv_b2t_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K.json"
    prefix = '23k_processed.json'
elif dataset_name == "Math23K_char":
    ckpt_dir = "Math23K_char_mv_b2t_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K_char.json"
    prefix = '23k_processed.json'
elif dataset_name == "cm17k":
    var_nums = ['x', 'y']
    ckpt_dir = "cm17k_mv_b2t_val_test"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/cm17k/questions.json"

ckpt_dir = ckpt_dir + '_' + str(n_epochs) + '_' + str(batch_size) + '_' + str(embedding_size) + '_' + str(hidden_size) + \
           '_blr' + str(bert_learning_rate) + '_lr' + str(learning_rate) + '_wd' + str(weight_decay) + '_do' + str(drop_out)
if beam_search:
    ckpt_dir = ckpt_dir + '_' + 'beam_search' + str(beam_size)

save_dir = os.path.join("./models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

# data = load_math23k_data("../dataset/math23k/Math_23K.json")
# data = load_math23k_data(data_path)
# pairs, generate_nums, copy_nums = transfer_math23k_num(data)
pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "Math23K_char":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "cm17k":
    data = load_cm17k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_cm17k_num(data)


temp_pairs = []
for p in pairs:
    ept = ExpressionTree()
    ept.build_tree_from_infix_expression(p["out_seq"])
    p['out_seq'] = ept.get_prefix_expression()
    temp_pairs.append(p)
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(data_dir, prefix, data, pairs)

last_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

last_acc_fold_v1 = []
best_val_acc_fold_v1 = []
all_acc_data_v1 = []

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
                                                                copy_nums, tree=True, use_lm=True, use_group_num=False)

embedding_weight = None

# Initialize models
encoder = MVPLMEncoderSeq(model_path=bert_path)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                     input_size=len(generate_nums) + len(var_nums), dropout=drop_out)
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                        embedding_size=embedding_size, dropout=drop_out)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=drop_out)

if not use_share_decoder:
    predict_v1 = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                            input_size=len(generate_nums) + len(var_nums), dropout=drop_out)
    generate_v1 = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                               embedding_size=embedding_size, dropout=drop_out)
    merge_v1 = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=drop_out)
else:
    predict_v1 = None
    generate_v1 = None
    merge_v1 = None


# the embedding layer is  only for generated number embeddings, operators, and paddings
# encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
encoder_optimizer = AdamW(encoder.parameters(), lr=bert_learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
if not use_share_decoder:
    predict_v1_optimizer = torch.optim.Adam(predict_v1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_v1_optimizer = torch.optim.Adam(generate_v1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_v1_optimizer = torch.optim.Adam(merge_v1.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    predict_v1_optimizer = None
    generate_v1_optimizer = None
    merge_v1_optimizer = None

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
if not use_share_decoder:
    predict_v1_scheduler = torch.optim.lr_scheduler.StepLR(predict_v1_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
    generate_v1_scheduler = torch.optim.lr_scheduler.StepLR(generate_v1_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
    merge_v1_scheduler = torch.optim.lr_scheduler.StepLR(merge_v1_optimizer, step_size=max(n_epochs//4, 1), gamma=0.5)
else:
    predict_v1_scheduler = None
    generate_v1_scheduler = None
    merge_v1_scheduler = None

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    if not use_share_decoder:
        predict_v1.cuda()
        generate_v1.cuda()
        merge_v1.cuda()

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

best_val_acc_v1 = 0
best_equ_acc_v1 = 0
current_best_val_acc_v1 = (0, 0, 0)

for epoch in range(n_epochs):
    loss_total = 0
    random.seed(epoch+random_seed)  # for reproduction
    batch_dict = prepare_data_batch(train_pairs, batch_size, lm_tokenizer=bert_tokenizer,
                                    use_group_num=False, use_lm=True)

    id_batches = batch_dict['id_batches']
    input_batches = batch_dict['input_batches']
    input_lengths = batch_dict['input_lengths']
    attention_mask_batches = batch_dict['attention_mask_batches']
    token_type_ids_batches = batch_dict['token_type_ids_batches']
    output_batches = batch_dict['output_batches']
    output_lengths = batch_dict['output_lengths']
    nums_batches = batch_dict['nums_batches']
    num_stack_batches = batch_dict['num_stack_batches']
    num_pos_batches = batch_dict['num_pos_batches']
    num_size_batches = batch_dict['num_size_batches']
    ans_batches = batch_dict['ans_batches']

    logs_content = "epoch: {}".format(epoch + 1)
    add_log(log_file,logs_content)
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

            loss = train_mvlm2tree(input_batches[idx][start_idx:end_idx],
                                   input_lengths[idx][start_idx:end_idx],
                                   attention_mask_batches[idx][start_idx:end_idx],
                                   token_type_ids_batches[idx][start_idx:end_idx],
                                   output_batches[idx][start_idx:end_idx],
                                   output_lengths[idx][start_idx:end_idx],
                                   num_stack_batches[idx][start_idx:end_idx],
                                   num_size_batches[idx][start_idx:end_idx],
                                   generate_num_ids, encoder, predict, generate, merge,
                                   encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer,
                                   output_lang,
                                   num_pos_batches[idx][start_idx:end_idx], use_clip=use_clip, clip=gclip,
                                   grad_acc=grad_acc, zero_grad=zero_grad, grad_acc_steps=grad_acc_steps,
                                   var_nums=var_num_ids, use_share_decoder=use_share_decoder,
                                   predict_v1=predict_v1, generate_v1=generate_v1, merge_v1=merge_v1,
                                   predict_v1_optimizer=predict_v1_optimizer,
                                   generate_v1_optimizer=generate_v1_optimizer,
                                   merge_v1_optimizer=merge_v1_optimizer)
            loss_total += loss

    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    if not use_share_decoder:
        predict_v1_scheduler.step()
        generate_v1_scheduler.step()
        merge_v1_scheduler.step()

    logs_content = "loss: {}".format(loss_total / len(input_lengths))
    add_log(log_file,logs_content)
    logs_content = "training time: {}".format(time_since(time.time() - start))
    add_log(log_file,logs_content)
    logs_content = "--------------------------------"
    add_log(log_file,logs_content)
    if epoch % 1 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        answer_ac = 0

        value_ac_v1 = 0
        equation_ac_v1 = 0
        answer_ac_v1 = 0

        ensemble_value_ac = 0
        ensemble_equation_ac = 0
        ensemble_answer_ac = 0

        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
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
            #     if t_id == bert_tokenizer.vocab['[NUM]']:
            #         num_pos.append(idx)

            test_res, test_score, test_res_v1, test_score_v1 = evaluate_mvlm2tree(tokens_dict["input_ids"], len(tokens_dict["input_ids"]), tokens_dict["attention_mask"], tokens_dict["token_type_ids"], generate_num_ids, encoder, predict, generate,
                                                       merge, output_lang, num_pos, beam_size=beam_size, beam_search=beam_search,
                                                       var_nums=var_num_ids,use_share_decoder=use_share_decoder,
                                                       predict_v1=predict_v1, generate_v1=generate_v1,
                                                       merge_v1=merge_v1)
            import traceback
            try:
                val_ac, equ_ac, ans_ac, \
                test_res_result, test_tar_result = compute_equations_result(test_res, copy.deepcopy(test_batch['output_cell']),
                                                                            output_lang, copy.deepcopy(test_batch['nums']),
                                                                            copy.deepcopy(test_batch['num_stack']),
                                                                            ans_list=test_batch['ans'],
                                                                            tree=True, prefix=True)
                # print(test_res_result, test_tar_result)
            except Exception as e:
                # traceback.print_exc()
                # print(e)
                val_ac, equ_ac, ans_ac = False, False, False

                # v1
            try:
                val_ac_v1, equ_ac_v1, ans_ac_v1, \
                test_res_result_v1, test_tar_result_v1 = compute_equations_result(test_res_v1,
                                                                                  copy.deepcopy(test_batch['output_cell']),
                                                                                  output_lang, copy.deepcopy(test_batch['nums']),
                                                                                  copy.deepcopy(test_batch['num_stack']),
                                                                                  ans_list=test_batch['ans'],
                                                                                  tree=True, prefix=True)
                # print(test_res_result, test_tar_result)
            except Exception as e:
                # traceback.print_exc()
                # print(e)
                val_ac_v1, equ_ac_v1, ans_ac_v1 = False, False, False

            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            if ans_ac:
                answer_ac += 1

            if val_ac_v1:
                value_ac_v1 += 1
            if equ_ac_v1:
                equation_ac_v1 += 1
            if ans_ac_v1:
                answer_ac_v1 += 1

            if (val_ac and test_score > test_score_v1) or (val_ac_v1 and test_score <= test_score_v1):
                ensemble_value_ac += 1
            if (equ_ac and test_score > test_score_v1) or (equ_ac_v1 and test_score <= test_score_v1):
                ensemble_equation_ac += 1

            if ans_ac or ans_ac_v1:
                ensemble_answer_ac += 1

            eval_total += 1
        logs_content = "{}, {}, {}".format(equation_ac, value_ac, eval_total)
        add_log(log_file, logs_content)
        logs_content = "test_answer_acc: {} {}".format(float(equation_ac) / eval_total, float(value_ac) / eval_total)
        add_log(log_file, logs_content)
        logs_content = "test_answer_acc_v1: {} {}".format(float(equation_ac_v1) / eval_total, float(value_ac_v1) / eval_total)
        add_log(log_file, logs_content)
        logs_content = "test_answer_acc_ensemble: {} {}".format(float(ensemble_equation_ac) / eval_total, float(ensemble_value_ac) / eval_total)
        add_log(log_file, logs_content)
        logs_content = "testing time: {}".format(time_since(time.time() - start))
        add_log(log_file, logs_content)
        logs_content = "------------------------------------------------------"
        add_log(log_file, logs_content)
        all_acc_data.append((epoch, equation_ac, value_ac, eval_total))
        all_acc_data_v1.append((epoch, equation_ac_v1, value_ac_v1, eval_total))

        torch.save(encoder.state_dict(), os.path.join(save_dir, "seq2tree_encoder"))
        torch.save(predict.state_dict(), os.path.join(save_dir, "seq2tree_predict"))
        torch.save(generate.state_dict(), os.path.join(save_dir, "seq2tree_generate"))
        torch.save(merge.state_dict(),  os.path.join(save_dir, "seq2tree_merge"))
        if not use_share_decoder:
            torch.save(predict_v1.state_dict(), os.path.join(save_dir, "seq2tree_predict_v1"))
            torch.save(generate_v1.state_dict(), os.path.join(save_dir, "seq2tree_generate_v1"))
            torch.save(merge_v1.state_dict(),  os.path.join(save_dir, "seq2tree_merge_v1"))

        if best_val_acc < value_ac:
            best_val_acc = value_ac
            current_best_val_acc = (equation_ac, value_ac, eval_total)
            torch.save(encoder.state_dict(), os.path.join(save_dir, "seq2tree_encoder_best_val_acc"))
            torch.save(predict.state_dict(), os.path.join(save_dir, "seq2tree_predict_best_val_acc"))
            torch.save(generate.state_dict(), os.path.join(save_dir, "seq2tree_generate_best_val_acc"))
            torch.save(merge.state_dict(),  os.path.join(save_dir, "seq2tree_merge_best_val_acc"))
            if not use_share_decoder:
                torch.save(predict_v1.state_dict(), os.path.join(save_dir, "seq2tree_predict_v1_best_val_acc"))
                torch.save(generate_v1.state_dict(), os.path.join(save_dir, "seq2tree_generate_v1_best_val_acc"))
                torch.save(merge_v1.state_dict(),  os.path.join(save_dir, "seq2tree_merge_v1_best_val_acc"))

        # if best_equ_acc < equation_ac:
        #     best_equ_acc = equation_ac
        #     torch.save(encoder.state_dict(), os.path.join(save_dir, "seq2tree_encoder_best_equ_acc"))
        #     torch.save(predict.state_dict(), os.path.join(save_dir, "seq2tree_predict_best_equ_acc"))
        #     torch.save(generate.state_dict(), os.path.join(save_dir, "seq2tree_generate_best_equ_acc"))
        #     torch.save(merge.state_dict(),  os.path.join(save_dir, "seq2tree_merge_best_equ_acc"))
        #     if not use_share_decoder:
        #         torch.save(predict_v1.state_dict(), os.path.join(save_dir, "seq2tree_predict_v1_best_equ_acc"))
        #         torch.save(generate_v1.state_dict(), os.path.join(save_dir, "seq2tree_generate_v1_best_equ_acc"))
        #         torch.save(merge_v1.state_dict(),  os.path.join(save_dir, "seq2tree_merge_v1_best_equ_acc"))

        if best_val_acc_v1 < value_ac_v1:
            best_val_acc_v1 = value_ac_v1
            current_best_val_acc_v1 = (equation_ac_v1, value_ac_v1, eval_total)
            torch.save(encoder.state_dict(), os.path.join(save_dir, "seq2tree_encoder_best_val_acc_v1"))
            torch.save(predict.state_dict(), os.path.join(save_dir, "seq2tree_predict_best_val_acc_v1"))
            torch.save(generate.state_dict(), os.path.join(save_dir, "seq2tree_generate_best_val_acc_v1"))
            torch.save(merge.state_dict(),  os.path.join(save_dir, "seq2tree_merge_best_val_acc_v1"))
            if not use_share_decoder:
                torch.save(predict_v1.state_dict(), os.path.join(save_dir, "seq2tree_predict_v1_best_val_acc_v1"))
                torch.save(generate_v1.state_dict(), os.path.join(save_dir, "seq2tree_generate_v1_best_val_acc_v1"))
                torch.save(merge_v1.state_dict(),  os.path.join(save_dir, "seq2tree_merge_v1_best_val_acc_v1"))

        # if best_equ_acc_v1 < equation_ac_v1:
        #     best_equ_acc_v1 = equation_ac_v1
        #     torch.save(encoder.state_dict(), os.path.join(save_dir, "seq2tree_encoder_best_equ_acc_v1"))
        #     torch.save(predict.state_dict(), os.path.join(save_dir, "seq2tree_predict_best_equ_acc_v1"))
        #     torch.save(generate.state_dict(), os.path.join(save_dir, "seq2tree_generate_best_equ_acc_v1"))
        #     torch.save(merge.state_dict(),  os.path.join(save_dir, "seq2tree_merge_best_equ_acc_v1"))
        #     if not use_share_decoder:
        #         torch.save(predict_v1.state_dict(), os.path.join(save_dir, "seq2tree_predict_v1_best_equ_acc_v1"))
        #         torch.save(generate_v1.state_dict(), os.path.join(save_dir, "seq2tree_generate_v1_best_equ_acc_v1"))
        #         torch.save(merge_v1.state_dict(),  os.path.join(save_dir, "seq2tree_merge_v1_best_equ_acc_v1"))

        if epoch == n_epochs - 1:
            last_acc_fold.append((equation_ac, value_ac, eval_total))
            best_val_acc_fold.append(current_best_val_acc)

            last_acc_fold_v1.append((equation_ac_v1, value_ac_v1, eval_total))
            best_val_acc_fold_v1.append(current_best_val_acc_v1)

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

a, b, c = 0, 0, 0
for bl in range(len(last_acc_fold_v1)):
    a += last_acc_fold_v1[bl][0]
    b += last_acc_fold_v1[bl][1]
    c += last_acc_fold_v1[bl][2]
    logs_content = "{}".format(last_acc_fold_v1[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

a, b, c = 0, 0, 0
for bl in range(len(best_val_acc_fold_v1)):
    a += best_val_acc_fold_v1[bl][0]
    b += best_val_acc_fold_v1[bl][1]
    c += best_val_acc_fold_v1[bl][2]
    logs_content = "{}".format(best_val_acc_fold_v1[bl])
    add_log(log_file, logs_content)
logs_content = "{} {}".format(a / float(c), b / float(c))
add_log(log_file, logs_content)
logs_content = "------------------------------------------------------"
add_log(log_file, logs_content)

logs_content = "{}".format(all_acc_data)
add_log(log_file, logs_content)

logs_content = "{}".format(all_acc_data_v1)
add_log(log_file, logs_content)

