def remove_error_prefix(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    modified_lines = []
    for line in lines:
        if line.strip().startswith('ERROR '):
            modified_lines.append(line.replace('ERROR ', '', 1))
        else:
            modified_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(modified_lines)
    
    print(f"已处理完成，结果保存到 {output_file}")

# 使用示例
input_file = "./qwen2.5_72b_472_zh_overlap_dir_none_True_zh.txt"
output_file = "./qwen2.5_72b_472_zh_overlap_dir_none_True_zh_cleaned.txt"
remove_error_prefix(input_file, output_file)
