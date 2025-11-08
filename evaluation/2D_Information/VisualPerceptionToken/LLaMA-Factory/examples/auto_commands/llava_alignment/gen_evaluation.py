import yaml, copy

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

file_path = '/root/LLaMA-Factory/examples/evaluation/qwen2vl_llama_alignment.yaml'
base_parameters = read_yaml(file_path)

command_files = [
    "commands_1.sh",
    "commands_2.sh",
    ]
files = []
for command_file in command_files:
    file = open(command_file, 'w')
    file.write("cd /root/LLaMA-Factory\n\n")
    files.append(file)

step_size = 10000

for index, start in enumerate(range(0, 600000, step_size)):
    end = start + step_size
    parameters_copy = copy.deepcopy(base_parameters)
    parameters_copy["output_dir"] = parameters_copy["output_dir"]+f"_{start}_{end}"
    parameters_copy["sample_range"] = f"{start},{end}"

    command = "llamafactory-cli train "
    for k,v in parameters_copy.items():
        command = command + f"--{k} {v} "
    command = command + "\n\n"
    files[index%len(files)].write(command)

for command_file in command_files:
    print(f"sh {command_file} >> {command_file}.log 2>&1 &")