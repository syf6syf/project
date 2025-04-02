import pickle
file_path = '/home/syf/pycharm_xm/Opt-GDBA/Opt-GDBA-main/MUTAG_triggersize_4'
with open(file_path, 'rb') as f:
    loaded_data = pickle.load(f) # 反序列化
print(loaded_data)
with open('/home/syf/pycharm_xm/Opt-GDBA/Opt-GDBA-main/result.py', 'w') as f:
    f.write(str(loaded_data))
