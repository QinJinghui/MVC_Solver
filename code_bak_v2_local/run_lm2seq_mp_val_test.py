# coding: utf-8
from src.train_and_evaluate import *
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
# bert_learning_rate = 5e-5
# bert_path = "./pretrained_lm/chinese-bert-wwm"
# learning_rate = 1e-3
# weight_decay = 2e-5
# beam_size = 5
# use_teacher_forcing = 1.0
# gclip = 0
# beam_search = True
# n_layers = 1
# drop_out = 0.5
# random_seed = 1
# dataset_name = "Math23K"
# ckpt_dir = "Math23K"
# data_dir = "./dataset/math23k/"
# data_path = data_dir + "Math_23K.json"
# prefix = '23k_processed.json'
# var_nums = []

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--grad_acc_steps', type=int, default=8)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--n_epochs', type=int, default=80)
parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
parser.add_argument('--bert_path', type=str, default="./pretrained_lm/chinese-bert-wwm")
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=2e-5)

parser.add_argument('--enable_beam_search', action='store_true')
parser.add_argument('--beam_size', type=int, default=5)

parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--use_teacher_forcing', type=float, default=1.0)
parser.add_argument('--use_clip', action='store_true')
parser.add_argument('--gclip', type=float, default=0.0)
parser.add_argument('--dataset_name', type=str, default='Math23K')
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
n_layers = args.n_layers  # 1
drop_out = args.drop_out  # 0.5
random_seed = args.random_seed  # 1
dataset_name = args.dataset_name  # "Math23K"
ckpt_dir = "Math23K"
data_dir = "./dataset/math23k/"
data_path = data_dir + "Math_23K.json"
prefix = '23k_processed.json'
var_nums = []

if dataset_name == "Math23K":
    ckpt_dir = "Math23K_b2s_mp_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K.json"
    prefix = '23k_processed.json'
elif dataset_name == "Math23K_char":
    ckpt_dir = "Math23K_char_mp_b2s_val_test"
    data_dir = "./dataset/math23k/"
    data_path = data_dir + "Math_23K_char.json"
    prefix = '23k_processed.json'
elif dataset_name == "cm17k":
    var_nums = ['x', 'y']
    ckpt_dir = "cm17k_b2s_mp_val_test"
    data_dir = "./dataset/cm17k/"
    data_path = data_dir + "questions.json"
    prefix = '17k.json'

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
    # ept = ExpressionTree()
    # ept.build_tree_from_infix_expression(p["out_seq"])
    # p['out_seq'] = ept.get_prefix_expression()
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
    torch.cuda.manual_seed(random_seed) # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_tokenizer.add_tokens(['[NUM]'])

_, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                    copy_nums, tree=False, use_lm=True,
                                                    use_group_num=False)
print(output_lang.index2word)

embedding_weight = None

# Initialize models
encoder = PLMMeanEncoderSeq(model_path=bert_path)
decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size,
                         input_size=output_lang.n_words, output_size=output_lang.n_words,
                         n_layers=n_layers, dropout=drop_out)
# the embedding layer is  only for generated number embeddings, operators, and paddings

# encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
encoder_optimizer = AdamW(encoder.parameters(), lr=bert_learning_rate, weight_decay=weight_decay)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

# opt scheduler
encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)


# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

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
    batches_dict = prepare_data_batch(train_pairs, batch_size, lm_tokenizer=bert_tokenizer, use_group_num=False, use_lm=True)

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

            loss = train_lm2seq(input_batches[idx][start_idx:end_idx],
                                input_lengths[idx][start_idx:end_idx],
                                attention_mask_batches[idx][start_idx:end_idx],
                                token_type_ids_batches[idx][start_idx:end_idx],
                                output_batches[idx][start_idx:end_idx],
                                output_lengths[idx][start_idx:end_idx],
                                nums_batches[idx][start_idx:end_idx],
                                num_stack_batches[idx][start_idx:end_idx],
                                copy_nums, generate_num_ids, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                output_lang, use_clip=use_clip, clip=gclip, use_teacher_forcing=use_teacher_forcing,
                                beam_size=1,
                                grad_acc=grad_acc, zero_grad=zero_grad, grad_acc_steps=grad_acc_steps,
                                beam_search=beam_search, var_nums=var_num_ids)

            loss_total += loss

    encoder_scheduler.step()
    decoder_scheduler.step()
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
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            # pairs: (id, input_seq, input_len, eq_segs, eq_len, nums, num_pos, num_stack, ans)
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
            test_res = evaluate_lm2seq(tokens_dict["input_ids"], len(tokens_dict["input_ids"]),
                                       tokens_dict["attention_mask"], tokens_dict["token_type_ids"],
                                       test_batch['nums'], copy_nums, generate_num_ids, encoder, decoder, output_lang,
                                        beam_size=beam_size, beam_search=beam_search, max_length=MAX_OUTPUT_LENGTH,
                                       var_nums=var_num_ids)
            import traceback
            try:
                val_ac, equ_ac, ans_ac, \
                test_res_result, test_tar_result = compute_equations_result(test_res,
                                                                            test_batch["output_cell"],
                                                                            output_lang,
                                                                            test_batch["nums"],
                                                                            test_batch['num_stack'],
                                                                            ans_list=test_batch['ans'],
                                                                            tree=False, prefix=False)
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

        torch.save(encoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_encoder"))
        torch.save(decoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_decoder"))
        if best_val_acc < value_ac:
            best_val_acc = value_ac
            current_best_val_acc = (equation_ac, value_ac, eval_total)
            torch.save(encoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_encoder_best_val_acc"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_decoder_best_val_acc"))
        if best_equ_acc < equation_ac:
            best_equ_acc = equation_ac
            torch.save(encoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_encoder_best_equ_acc"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "lm2seq_attn_decoder_best_equ_acc"))
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
