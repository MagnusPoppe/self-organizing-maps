import json
import os

def input_number(text:str) -> int:
    epochs = input("\t"+text)
    while not epochs.isdigit():
        print("\"" + epochs + "\" is not a number. Write a number!")
        epochs = input("\t"+text)
    return int(epochs)

def input_json_file(text:str="file: configurations/") -> str:
    print(os.listdir("configurations/"))
    json_file = input("\tfile: configurations/")
    while json_file[-5:] != ".json":
        print("\"" + json_file + "\" is no a json file!")
        json_file = input("\t"+text)
    return json_file

def input_json(text:str) -> dict:
    """ Reads raw json """
    inn = input("\t"+text+"=")
    not_valid_json=True
    out = None
    while not_valid_json:
        try:
            out = json.loads(inn)
            not_valid_json=False
        except Exception:
            not_valid_json = True
            print("\"" + inn + "\" is not json!")
            inn = input("\t"+text+"=")
    return out