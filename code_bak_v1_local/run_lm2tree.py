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
# from src.expressions_transfer import *
from src.data_utils import get_pretrained_embedding_weight
from transformers import BertTokenizer, AdamW

USE_CUDA = torch.cuda.is_available()
batch_size = 16
grad_acc_steps = 8  # 使用grad_acc_steps步来完成batch_size的训练，每一步：batch_size // grad_acc_steps
embedding_size = 128
hidden_size = 768
n_epochs = 80
bert_learning_rate = 5e-5
bert_path = "./pretrained_lm/chinese-bert-wwm"
learning_rate = 1e-3
weight_decay = 2e-5
beam_size = 5
beam_search = True
fold_num = 5
n_layers = 2
drop_out = 0.5
random_seed = 1
var_nums = []
dataset_name = "mawps"
ckpt_dir = "Math23K"
data_path = "../dataset/math23k/Math_23K.json"

if dataset_name == "Math23K":
    var_nums = []
    ckpt_dir = "Math23K_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/math23k/Math_23K.json"
elif dataset_name == "Math23K_char":
    var_nums = []
    ckpt_dir = "Math23K_char_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/math23k/Math_23K_char.json"
elif dataset_name == "ALG514":
    var_nums = ['x','y']
    ckpt_dir = "ALG514_b2t"
    bert_path = "../pretrained_lm/bert-base-uncased"
    data_path = "./dataset/alg514/questions.json"
elif dataset_name == "mawps":
    var_nums = []
    ckpt_dir = "mawps_b2t"
    bert_path = "./pretrained_lm/bert-base-uncased"
    data_path = "./dataset/mawps/mawps_combine.json"
elif dataset_name == "hmwp":
    var_nums = ['x', 'y']
    ckpt_dir = "hmwp_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/hmwp/hmwp.json"
elif dataset_name == "cm17k":
    var_nums = ['x', 'y']
    ckpt_dir = "cm17k_b2t"
    bert_path = "./pretrained_lm/chinese-bert-wwm"
    data_path = "./dataset/cm17k/questions.json"

ckpt_dir = ckpt_dir + '_' + str(n_epochs) + '_' + str(batch_size) + '_' + str(embedding_size) + '_' + str(hidden_size)
if beam_search:
    ckpt_dir = ckpt_dir + '_' + 'beam_search'


save_dir = os.path.join("./models", ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = os.path.join(save_dir, 'log')
create_logs(log_file)

for fold_id in range(fold_num):
    if not os.path.exists(os.path.join(save_dir, 'fold-'+str(fold_id))):
        os.mkdir(os.path.join(save_dir, 'fold-'+str(fold_id)))

pairs = None
generate_nums = None
copy_nums = None
if dataset_name == "Math23K":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "Math23K_char":
    data = load_math23k_data(data_path)
    pairs, generate_nums, copy_nums = transfer_math23k_num(data)
elif dataset_name == "ALG514":
    data = load_alg514_data(data_path)
    pairs, generate_nums, copy_nums = transfer_alg514_num(data)
elif dataset_name == "mawps":
    data = load_mawps_data(data_path)
    pairs, generate_nums, copy_nums = transfer_mawps_num(data)
elif dataset_name == "hmwp":
    data = load_hmwp_data(data_path)
    pairs, generate_nums, copy_nums = transfer_hmwp_num(data)
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

fold_size = int(len(pairs) / fold_num)
fold_pairs = []
for split_fold in range((fold_num - 1)):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * (fold_num - 1)):])

last_acc_fold = []
best_val_acc_fold = []
all_acc_data = []

for fold in range(fold_num):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(fold_num):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

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
                                                        copy_nums, tree=True, use_lm=True, use_group_num=False)

    embedding_weight = None

    # Initialize models
    encoder = PLMEncoderSeq(model_path=bert_path)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                         input_size=len(generate_nums) + len(var_nums), dropout=drop_out)
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums) - len(var_nums),
                            embedding_size=embedding_size, dropout=drop_out)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size, dropout=drop_out)

    # the embedding layer is  only for generated number embeddings, operators, and paddings
    encoder_optimizer = AdamW(encoder.parameters(), lr=bert_learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=max(n_epochs//4,1), gamma=0.5)

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
    current_save_dir = os.path.join(save_dir, 'fold-'+str(fold))
    current_best_val_acc = (0,0,0)
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

        logs_content = "fold: {}".format(fold+1)
        add_log(log_file, logs_content)
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

                loss = train_lm2tree(input_batches[idx][start_idx:end_idx],
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
                                     num_pos_batches[idx][start_idx:end_idx],
                                     grad_acc=grad_acc, zero_grad=zero_grad, grad_acc_steps=grad_acc_steps,
                                     var_nums=var_num_ids)
                loss_total += loss

        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()

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

                test_res = evaluate_lm2tree(tokens_dict["input_ids"], len(tokens_dict["input_ids"]), tokens_dict["attention_mask"], tokens_dict["token_type_ids"], generate_num_ids, encoder, predict, generate,
                                            merge, output_lang, num_pos, beam_size=beam_size, beam_search=beam_search,
                                            var_nums=var_num_ids)
                import traceback
                try:
                    val_ac, equ_ac, ans_ac, \
                    test_res_result, test_tar_result = compute_equations_result(test_res, test_batch['output_cell'],
                                                                                output_lang, test_batch['nums'],
                                                                                test_batch['num_stack'],
                                                                                ans_list=test_batch['ans'],
                                                                                tree=True, prefix=True)
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
            all_acc_data.append((fold, epoch,equation_ac, value_ac, eval_total))

            torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder"))
            torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict"))
            torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate"))
            torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge"))
            if best_val_acc < value_ac:
                best_val_acc = value_ac
                current_best_val_acc = (equation_ac, value_ac, eval_total)
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_val_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_val_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_val_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_val_acc"))
            if best_equ_acc < equation_ac:
                best_equ_acc = equation_ac
                torch.save(encoder.state_dict(), os.path.join(current_save_dir, "seq2tree_encoder_best_equ_acc"))
                torch.save(predict.state_dict(), os.path.join(current_save_dir, "seq2tree_predict_best_equ_acc"))
                torch.save(generate.state_dict(), os.path.join(current_save_dir, "seq2tree_generate_best_equ_acc"))
                torch.save(merge.state_dict(),  os.path.join(current_save_dir, "seq2tree_merge_best_equ_acc"))
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
