import re
import time
import os
from dateutil.parser import parse
import json


mail_dir = 'dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/maildir/' 
json_dir = 'dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/mail_jsons/' 
folders_to_process = ['inbox', '_sent_mail']


def high_level_message_cleaning(message_text):
    message_text = re.sub(r"=[0-9]+", " ", message_text)
    message_text = re.sub(r"=\n", "\n", message_text)
    return message_text


def split_by_reply(message_text):
    return message_text.split("-----Original Message-----")


def parse_field(message_text, field_name, sep_char="\n"):
    try:
        regex_search_string = r"" + field_name + r":([^\r\n]+)"
        field_text = re.search(regex_search_string, message_text).groups()[0]
        field_text = ''.join(field_text.partition(sep_char)[:1]).strip()
    except:
        field_text = '' # In case a field is blank or null (like subject)
    return field_text


def parse_text(message_text, sep_char="---"):
    try:
        regex_search_string = "(\.pst[^-]*|\.nsf[^-]*)"
        text = re.search(regex_search_string, message_text).groups()[0][4:]
    except:
        regex_search_string = r"Subject:(.*)([^\r]+)"
        text = re.search(regex_search_string, message_text).groups()[1]
    text = ''.join(text.partition(sep_char)[:1]).strip()
    return text


def split_email_addresses(email_address_string):
    email_address_string = email_address_string.replace("'", "")
    email_address_string = email_address_string.replace(".com,", ";").replace(">,", ">;")
    # Remove anything <like_this>
    email_address_string = re.sub(r"(?<=\<)(.*?)(?=\>)", "", email_address_string)
    email_address_string = email_address_string.replace("<>", "")
    # Remove anything [like_this]
    email_address_string = re.sub(r"(?<=\[)(.*?)(?=\])", "", email_address_string)
    email_address_string = email_address_string.replace("[]", "")
    return [s.strip() for s in email_address_string.split(";")]


def format_date(date_string):
    try:
        return time.asctime(parse(date_string).timetuple())
    except:
        return ''  # Give up - no easy way to clean date strings


def process_one_email_file(f):
    message_text = open(f).read()
    message_text = high_level_message_cleaning(message_text)
    message_text_split = split_by_reply(message_text)
    message_text_list = []
    for i,m in enumerate(message_text_split):
        message_dict = {}
        if i == 0:
            message_dict['id'] = parse_field(m, "Message-ID")
            message_dict['date'] = format_date(parse_field(m, "Date"))
            message_dict['from_email'] = split_email_addresses(parse_field(m, "From"))
            message_dict['to_email'] = split_email_addresses(parse_field(m, "To"))
            message_dict['subject'] = parse_field(m, "Subject")
            message_dict['from'] = parse_field(m, "X-From", "<")
            message_dict['to'] = split_email_addresses(parse_field(m, "X-To"))
            message_dict['cc_email'] = split_email_addresses(parse_field(m, "X-cc"))
            message_dict['bcc_email'] = split_email_addresses(parse_field(m, "X-bcc"))
            message_dict['text'] = parse_text(m)
        else:
            message_dict['id'] = ''
            message_dict['date'] = format_date(parse_field(m, "Sent"))
            message_dict['from_email'] = ['']
            message_dict['to_email'] = ['']
            message_dict['subject'] = parse_field(m, "Subject")
            message_dict['from'] = split_email_addresses(parse_field(m, "From"))
            message_dict['to'] = split_email_addresses(parse_field(m, "To"))
            message_dict['cc_email'] = ['']
            message_dict['bcc_email'] = ['']
            message_dict['text'] = parse_text(m)
        message_text_list.append(message_dict)
    return message_text_list


'''
one_test = 'dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/maildir/allen-p/inbox/1'
email_output = process_one_email_file(one_test)
print(email_output)
'''


def process_one_employees_emails(employee_folder):
    employee_emails = []
    for (root, dirs, file_names) in os.walk(employee_folder):
        if root.split("/")[-1].lower() not in folders_to_process:
            continue
        else:
            print("Processing folder:", root.split("/")[-1].lower())
            folder_emails = []
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                email_output = process_one_email_file(file_path)
                folder_emails += email_output
            employee_emails += folder_emails
    for i, email in enumerate(employee_emails):
        with open(json_dir+root.split("/")[-2:-1][0]+"_"+str(i)+'.json', 'w') as file:
            json.dump(email, file)


'''
one_test = 'dsilt-ml-code/12 Natural Language Processing/Enron Emails with Python/maildir/allen-p/' 
process_one_employees_emails(one_test)
'''


for (root, dirs, file_names) in os.walk(mail_dir):
    empl = root.split("/")[-1].split(os.sep)[0].lower()+"_"
    employee_emails = []
    if root.split(os.sep)[-1] not in folders_to_process:
        continue
    else:
        print("Processing folder:", empl, root.split(os.sep)[-1])
        folder_emails = []
        for file_name in file_names:
            try:
                file_path = os.path.join(root, file_name)
                email_output = process_one_email_file(file_path)
                folder_emails += email_output
            except:
                next # If the email doesn't process, skip it
        employee_emails += folder_emails
    for i, email in enumerate(employee_emails):
        with open(json_dir+empl+str(i)+'.json', 'w') as file:
            json.dump(email, file)

