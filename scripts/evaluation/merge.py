'''
Autor: Biranavan Parameswaran
'''
import os
import json

directory = "./logs_tb/wcid_logs_summary"

merged_data = {}
# merge all the json files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)

        with open(file_path, "r") as file:
            json_data = json.load(file)

        model_name = json_data["name"]
        model_data = json_data

        if model_name in merged_data:
            merged_data[model_name].update(model_data)
        else:
            merged_data[model_name] = model_data

output_file = "merged_data.json"
with open(output_file, "w") as file:
    json.dump(merged_data, file, indent=4)

print(f"Merged data has been written to {output_file}.")
