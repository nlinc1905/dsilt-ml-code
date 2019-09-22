import os
import json
import pandas as pd


json_dir = 'dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/mail_jsons/' 


jsons_list = []
for (root, dirs, file_names) in os.walk(json_dir):
    for file in file_names:
        #print(json_dir+file)
        with open(json_dir+file) as f:
            f_data = json.load(f)
            jsons_list.append(f_data)


d = pd.DataFrame(jsons_list)


# clean data
del jsons_list
d.drop('id', axis=1, inplace=True)
d = d[(d['from'] != '') & (d['to'] != '') & (d['date'] != '')]
d.reset_index(inplace=True).rename(columns={'index': 'id'}, inplace=True)
d['date'] = d['date'].astype('datetime64[ns]')


def unique_entities(df_col):
    return list(set([entity for sublist in df_col for entity in sublist]))


print(len(unique_entities(d['from'])))
print(len(unique_entities(d['to'])))
print(len(unique_entities(d['cc_email'])))
print(len(unique_entities(d['bcc_email'])))


d.to_json('emails.json', orient='records', date_format='epoch', date_unit='ms')
