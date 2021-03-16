import json

from tqdm import tqdm


def read_json_data_from_file(file: str):
    ret_data = []
    with open(file) as json_file:
        # Get next line from file
        lines = json_file.readlines()
        for line in tqdm(lines):
            if not line:
                break

            data = json.loads(line)
            ret_data.append(data)
    return ret_data


def convert_to_id_dict(data, id_key: str):
    ret_dict = {}
    for element in data:
        if id_key in element:
            ret_dict[element[id_key]] = element
        else:
            print(f'Element {element} doenst contain key {id_key}')
    return ret_dict


def get_table_column(data_list, tables_dict):
    ret_list = []
    for element in data_list:
        current_table = tables_dict[element['table_id']]
        columns = current_table['header']
        # Replace the index
        element['columns'] = columns
        element['types'] = current_table['types']
        element['sql']['sel_name'] = columns[element['sql']['sel']]

        if 'page_title' in current_table:
            element['table_name'] = current_table['page_title']
        elif 'section_title' in current_table:
            element['table_name'] = current_table['section_title']
        elif 'caption' in current_table:
            element['table_name'] = current_table['caption']
        elif 'name' in current_table:
            element['table_name'] = current_table['name']

        # For the where conditions
        for cond in element['sql']['conds']:
            cond.append(columns[cond[0]])
        ret_list.append(element)
    return ret_list
