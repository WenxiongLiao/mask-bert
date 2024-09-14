import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_prompt_prefix(tokenizer,few_shot_text,few_shot_labels,id2label):
    label_names = list(id2label.values())
    label_str = ''
    for name in label_names[:-1]:
        label_str = label_str + name + ', '
    label_str  = label_str + 'or ' + label_names[-1] + '\n\n'
    
    prompt_prefix = '''You are a text classification assistant. Please determine the category of the text based on the input text.
Please classify the following text into one of ''' + label_str
    example_num = len(few_shot_text)
    for i in range(example_num,0,-1):
        prompt =  prompt_prefix
        for j in range(i):
            example = 'Input: {input_str}\nOutput: {label}\n\n'.format(input_str = few_shot_text[j],label = few_shot_labels[j])
            prompt = prompt + example
            
        prompt_tokens = tokenizer(prompt,return_tensors = 'pt')    
        if len(prompt_tokens.input_ids[0]) < 1800:
            break
    
    return prompt
    



def get_correct_num(tokenizer,prompts,result,batch_test_labels,novel_label_names):
    correct_num = 0
    for i in range(len(batch_test_labels)):
        res = tokenizer.decode(result[i], skip_special_tokens=True)
        assert res.find(prompts[i]) == 0, print('data error')
        res = res.replace(prompts[i], '')
        
        find_idxs = []
        for name in novel_label_names:
            find_idxs.append(res.lower().find(name.lower()))
        if all(ele == -1 for ele in find_idxs):
            # 如果无有效答案，则随机选择
            correct_num = correct_num + 1 / len(novel_label_names)
        else:
            filter_list = [x for x in find_idxs if x != -1]
            if batch_test_labels[i] == novel_label_names[find_idxs.index(min(filter_list))]:
                # 如果有有效答案，则认为最先输出的为正确答案
                correct_num = correct_num + 1
            
        
    return correct_num

def get_predict_acc(model,tokenizer,ellipsis,novel_few_shot_path,novel_test_path,max_length,id2label,novel_labels,device,batch_size = 8):
    model.eval()
    
    few_shot_data = pd.read_csv(novel_few_shot_path)
    few_shot_data = few_shot_data.sample(frac = 1.0)
    id2label = {nl:id2label[nl] for nl in novel_labels}
    
    max_length = int(max_length * 0.7)
    few_shot_labels = [id2label[l] for l in few_shot_data.labels]
    few_shot_text = [t if len(t.split()) < max_length else ' '.join(t.split()[0: max_length]) + ellipsis for t in few_shot_data.text]
    
    test_data = pd.read_csv(novel_test_path)
    if novel_test_path.find('dbpedia14')!= -1 or novel_test_path.find('PubMed20k')!= -1:
        test_data = test_data.sample(frac = 0.1)
        batch_size = 4
        if novel_test_path.find('dbpedia14')!= -1:
            batch_size = 2
    test_labels = [id2label[l] for l in test_data.labels]
    test_text = [t if len(t.split()) < max_length else ' '.join(t.split()[0: max_length]) + ellipsis for t in test_data.text]
    test_num = len(test_text)
    
    prompt_prefix = get_prompt_prefix(tokenizer,few_shot_text,few_shot_labels,id2label)
    batch_num = int(test_num / batch_size) if test_num%batch_size == 0 else int(test_num / batch_size) + 1
    correct_num = 0
    import time 
    begin = time.time()
    for i in range(batch_num):
        batch_test_text = test_text[i * batch_size : (i + 1)* batch_size]
        batch_test_labels = test_labels[i * batch_size : (i + 1)* batch_size]
        
        prompts = [prompt_prefix +'Input: {text}\nOutput: '.format(text = text) for text in batch_test_text]
        model_input = tokenizer(prompts,padding=True,truncation=True,return_tensors = 'pt').to(device)
        with torch.no_grad():
            result = model.generate(**model_input, do_sample=True,max_new_tokens=15, temperature=0.1,top_k = 1)

        correct_num = correct_num + get_correct_num(tokenizer,prompts,result,batch_test_labels,list(id2label.values()))
    print('=========>>',str(time.time() - begin))
    acc = correct_num /  test_num
    print(f'acc :{acc}')
    
    return acc

def load_LLM(model_path,tokenizer_path,device):
    if tokenizer_path == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,padding_side='left')

        
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_4bit=True,torch_dtype = torch.bfloat16)
    
    return model,tokenizer

