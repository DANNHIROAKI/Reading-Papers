# 定义输入和输出文件名
input_file = 'output.txt'
output_file = 'filtered_output.txt'

# 打开输入文件并准备输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    lines = infile.readlines()  # 读取所有行
    for i in range(len(lines)):
        line = lines[i]
        # 检查是否为标题行
        if line.startswith('title = {'):
            title = line.strip()
            # 检查标题中是否包含“LLM”或“Language Model”
            if 'LLM' in title or 'Language Model' in title:
                outfile.write(title + '\n')  # 写入标题
                if i + 1 < len(lines) and lines[i + 1].startswith('url = {'):
                    outfile.write(lines[i + 1])  # 写入对应的URL

# 提示完成
print("筛选完成，结果已保存到 filtered_output.txt 文件。")
