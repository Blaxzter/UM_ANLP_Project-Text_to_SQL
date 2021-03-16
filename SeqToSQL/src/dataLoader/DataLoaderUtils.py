def get_question_answers(request, tokenizer):
    """
    Function that pads the qa in the request to the same length
    """
    input_list = []

    table_name = request['table_name']  # should be name not id
    space_token = ' '
    columns = request['columns']
    req_question = request['question']  # might need to be tokenized
    max_len = 0
    for i, col in enumerate(columns):
        col_type = request['types'][i]  # infere type somehow
        column_representation = col_type + space_token + table_name + space_token + col
        embedding = tokenizer.encode_plus(
            column_representation,
            req_question,
            add_special_tokens = True,
        )
        if max_len < len(embedding['input_ids']):
            max_len = len(embedding['input_ids'])

    for i, col in enumerate(columns):
        col_type = request['types'][i]  # infere type somehow
        column_representation = col_type + space_token + table_name + space_token + col
        embedding = tokenizer.encode_plus(
            column_representation,
            req_question,
            add_special_tokens = True,
            max_length = max_len,
            padding = 'max_length',
            return_overflowing_tokens = True,
            return_attention_mask = True,
        )
        input_list.append(embedding)
    return input_list


def get_question_answers_def_length(request, tokenizer, pad_max_length):
    input_list = []

    table_name = request['table_name']  # should be name not id
    space_token = ' '
    columns = request['columns']
    req_question = request['question']  # might need to be tokenized
    for i, col in enumerate(columns):
        col_type = request['types'][i]  # infere type somehow
        column_representation = col_type + space_token + table_name + space_token + col
        embedding = tokenizer.encode_plus(
            column_representation,
            req_question,
            add_special_tokens = True,
            max_length = pad_max_length,
            padding = 'max_length',
            truncation = True,
            return_overflowing_tokens = True,
            return_attention_mask = True,
        )
        input_list.append(embedding)
    return input_list


def get_question_answers_for_where_value_def_length(request, tokenizer, pad_max_length):
    input_list = []
    target_list = []
    cond_dict = {0: "equal to ", 1: "less than ", 2: "more than ", 3: "OP "}

    # agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    # cond_ops = ['=', '>', '<', 'OP']

    space_token = ' '
    req_question = request['question']

    conditions = request['sql']['conds']
    for i, cond in enumerate(conditions):
        column_name = cond[-1]
        opp_name = cond_dict[cond[1]]
        target = cond[2]
        value_question = column_name + space_token + opp_name

        embedding = tokenizer.encode_plus(
            text = value_question,
            text_pair = req_question,
            add_special_tokens = True,
            max_length = pad_max_length,
            padding = 'max_length',
            truncation = True,
            return_overflowing_tokens = True,
            return_attention_mask = True,
        )

        input_list.append(embedding)
        encoded_target = tokenizer.encode(
            text = str(target).lower() if (str(target).lower() in req_question) else str(target),
            add_special_tokens = False
        )
        startIdx = -1
        endIdx = -1

        sll = len(encoded_target)
        for ind in (i for i, e in enumerate(embedding['input_ids']) if e == encoded_target[0]):
            if embedding['input_ids'][ind:ind + sll] == encoded_target:
                startIdx = ind
                endIdx = ind + sll
                break

        target_list.append([startIdx, endIdx])

    return input_list, target_list
