import glob
import re
import os

# 1. 自动寻找所有的 .out 日志文件
# 根据你的截图，文件名格式类似 logs_2010834_18.out
log_files = glob.glob("logs_*.out")

if not log_files:
    print("❌ 错误：在当前目录下没有找到 logs_*.out 文件，请确保脚本在正确目录下运行。")
    exit(1)

success_rates = []
task_results = []
missing_files = []

print("="*50)
print("📊 正在解析各个任务的评估成绩...")
print("="*50)

# 2. 遍历每个日志文件
for f in log_files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 使用正则表达式精准抓取任务名和成功率
            # 匹配格式: 
            # Task: CoffeePressButton
            # Success rate: 0.9400 (94%)
            task_match = re.search(r"Task:\s+([a-zA-Z]+)", content)
            sr_match = re.search(r"Success rate:\s+[0-9.]+\s+\(([0-9.]+)%\)", content)
            
            if task_match and sr_match:
                task_name = task_match.group(1)
                sr_percent = float(sr_match.group(1))
                
                task_results.append({"file": f, "task": task_name, "sr": sr_percent})
                success_rates.append(sr_percent)
            else:
                # 记录可能因为报错中断而没有输出最终成绩的文件
                missing_files.append(f)
                
    except Exception as e:
        print(f"⚠️ 读取文件 {f} 时发生错误: {e}")

# 3. 按任务名称字母顺序排序，方便查看
task_results.sort(key=lambda x: x["task"])

# 4. 打印每个任务的成绩
for res in task_results:
    print(f"✅ {res['task'].ljust(25)} : {res['sr']:>6.2f}%  (来源: {res['file']})")

print("-" * 50)

# 5. 打印异常文件警告（如果有）
if missing_files:
    print(f"⚠️ 注意: 有 {len(missing_files)} 个文件没有找到 'FINAL RESULTS' (可能运行报错或超时):")
    for mf in missing_files:
        print(f"   - {mf}")
    print("-" * 50)

# 6. 计算并输出最终平均成功率
if success_rates:
    average_sr = sum(success_rates) / len(success_rates)
    print(f"🎯 共成功统计了 {len(success_rates)} 个任务")
    print(f"🏆 最终整体平均成功率 (Average SR): {average_sr:.2f}%")
else:
    print("❌ 没有提取到任何成功率数据，请检查日志文件内容。")

print("="*50)