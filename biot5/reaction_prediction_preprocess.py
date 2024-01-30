import json
import os
import tqdm


def split_train_valid_test_instances(data):
    train = []
    test = []
    for i in range(len(data)):
        if data[i]['metadata']['split'] == 'train':
            train.append(data[i])
        elif data[i]['metadata']['split'] == 'test':
            test.append(data[i])
        else:
            raise ValueError('Invalid split value')
        
    valid_len = int(len(train) * 0.2)
    valid = train[-valid_len:]
    
    return train, valid, test

def wrap_selfies_in_special_tokens(selfies):
    return '<bom>' + selfies + '<eom>'

def preprocess_instance(instance):
    
    if instance['metadata']['task'] in ['forward reaction prediction', 'retrosynthesis']:
        input_selfies = wrap_selfies_in_special_tokens(instance['input'])
    elif instance['metadata']['task'] == 'reagent prediction':
        input_selfies = instance['input'].split('>>')
        input_selfies = [wrap_selfies_in_special_tokens(input_selfies[0]), wrap_selfies_in_special_tokens(input_selfies[1])]
        input_selfies = '>>'.join(input_selfies)
    else:
        raise ValueError('Invalid task value')
    input = instance['instruction'] + '\n\n' + input_selfies
    output = [wrap_selfies_in_special_tokens(instance['output'])]
    out_instance = {
        'input': input,
        'output': output
    }
    return out_instance


def preprocess_data_split(split):
    processed_instances = []
    iter_bar = tqdm.tqdm(split)
    for i in range(len(iter_bar)):
        split_instance = split[i]
        instance = preprocess_instance(split_instance)
        instance['id'] = i
        processed_instances.append(instance)

    json_data = {
        "Contributors": [""],
        "Categories": [""],
        "Reasoning": [""],
        "URL": [""],
        "Instruction_language": [""],
        "Domains": [""],
        "Positive Examples": [],
        "Negative Examples": [],
        "Source": [""],
        "Definition": [""],
        "Input_language": [""],
        "Output_language": [""],
        "Instance License": ["Unknown"],
        "Instances": processed_instances
    }
    return json_data



def get_unique_instructions(data):
    instructions = []
    for i in range(len(data)):
        instructions.append(data[i]['instruction'])
    return set(instructions)

data_dir = 'biot5/data/tasks/Molecule-oriented_Instructions'
save_dir = 'biot5/data/tasks'

files = ['forward_reaction_prediction.json', 'reagent_prediction.json', 'retrosynthesis.json']

iter_bar = tqdm.tqdm(files)
for file in iter_bar:

    data_path = os.path.join(data_dir, file)

    # read json data
    with open(data_path, 'r') as f:
        data = json.load(f)

    train, valid, test = split_train_valid_test_instances(data)

    train = preprocess_data_split(train)
    valid = preprocess_data_split(valid)
    test = preprocess_data_split(test)

    # save train, valid, test json data
    train_path = os.path.join(save_dir, file.replace('.json', '_train.json'))
    valid_path = os.path.join(save_dir, file.replace('.json', '_valid.json'))
    test_path = os.path.join(save_dir, file.replace('.json', '_test.json'))

    with open(train_path, 'w') as f:
        json.dump(train, f, indent=2)
    with open(valid_path, 'w') as f:
        json.dump(valid, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test, f, indent=2)

a = 17
