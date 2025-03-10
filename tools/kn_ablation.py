import os
import re
import pandas as pd

def extract_metrics_from_log(file_path):
    """从日志文件中提取最后一个Best指标"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # 查找最后一个包含Best的行
            matches = re.findall(r'Best SRCC: ([\d.]+), Best PLCC: ([\d.]+)', content)
            if matches:
                return float(matches[-1][0]), float(matches[-1][1])
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
    return None, None

def parse_k_n_from_dirname(dirname):
    """从目录名解析k和n的值"""
    match = re.match(r'k(\d+)_n(\d+)_', dirname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    # 创建空的DataFrame用于存储结果
    k_values = [1, 2, 4, 8]
    n_values = [1, 2, 4, 8, 16, 32, 64]
    
    srcc_df = pd.DataFrame(index=k_values, columns=n_values)
    plcc_df = pd.DataFrame(index=k_values, columns=n_values)
    
    # 遍历日志目录
    base_dir = 'exp_log/kn'
    for dirname in os.listdir(base_dir):
        log_path = os.path.join(base_dir, dirname, 'train.log')
        if os.path.exists(log_path):
            k, n = parse_k_n_from_dirname(dirname)
            if k in k_values and n in n_values:
                srcc, plcc = extract_metrics_from_log(log_path)
                if srcc is not None and plcc is not None:
                    srcc_df.loc[k, n] = srcc
                    plcc_df.loc[k, n] = plcc
    
    # 保存结果到CSV文件
    srcc_df.to_csv('paper/best_srcc.csv')
    plcc_df.to_csv('paper/best_plcc.csv')
    
    # 打印结果
    print("\nBest SRCC:")
    print(srcc_df)
    print("\nBest PLCC:")
    print(plcc_df)

if __name__ == "__main__":
    main()