# 定义输入和输出文件名
input_file = 'acm.txt'
output_file = 'output.txt'

# 打开输入文件并准备输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 检查行是否为空或以 title 或 url 开头
        if line.strip() == '' or line.startswith('title = {') or line.startswith('url = {'):
            # 写入输出文件
            outfile.write(line)

# 提示完成
print("处理完成，结果已保存到 output.txt 文件。")
