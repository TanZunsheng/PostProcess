import os
import subprocess
import shutil

base_dir = '/work/2024/tanzunsheng/PENCIData'
script_path = '/work/2024/tanzunsheng/Code/BrainOmniPostProcess/generate_metadata_fast.py'

# 包含 Broderick2018 的 4 个子数据集
datasets = [
    'HBN_EEG', 
    'SEED-DV', 
    'THINGS-EEG', 
    'ThingsEEG', 
    'Grootswagers2019', 
    'Brennan_Hale2019',
    'Broderick2018/Broderick2018_CocktailParty',
    'Broderick2018/Broderick2018_NaturalSpeech',
    'Broderick2018/Broderick2018_NaturalSpeechReverse',
    'Broderick2018/Broderick2018_SpeechInNoise'
]

print('清理旧的 metadata 文件夹...')
# 清理一级目录和二级的 metadata
for ds in datasets:
    ds_name = os.path.basename(ds)
    ds_path = os.path.join(base_dir, ds)
    parent_dir = os.path.dirname(ds_path)  # e.g. PENCIData/ 或 PENCIData/Broderick2018/
    
    # 删目标目录旁边的 {ds_name}-metadata（与当前生成逻辑一致）
    target_meta = os.path.join(parent_dir, f'{ds_name}-metadata')
    if os.path.exists(target_meta):
        shutil.rmtree(target_meta)
        print(f'  删除 {target_meta}')

    # 删数据集目录内部遗留的 metadata/（旧逻辑产物）
    legacy_meta = os.path.join(ds_path, 'metadata')
    if os.path.exists(legacy_meta):
        shutil.rmtree(legacy_meta)
        print(f'  删除 {legacy_meta}')
        
    # 如果处理过合并的 Broderick2018-metadata 也删掉
    if 'Broderick2018' in ds_name:
        brod_meta = os.path.join(base_dir, 'Broderick2018-metadata')
        if os.path.exists(brod_meta):
            shutil.rmtree(brod_meta)
            print(f'  删除 {brod_meta}')

print('\\n重新并发生成具有缩进格式化(indent)且拆分子数据集的 metadata...')
procs = []
for ds in datasets:
    ds_path = os.path.join(base_dir, ds)
    if os.path.isdir(ds_path):
        print(f'启动处理: {ds}')
        p = subprocess.Popen(
            ['/work/2024/tanzunsheng/anaconda3/envs/EEG/bin/python', script_path, ds_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        procs.append((ds, p))

for ds, p in procs:
    out, err = p.communicate()
    ds_name = os.path.basename(ds)
    if p.returncode == 0:
        print(f'{ds_name} 处理成功.')
        # 现在 generate_metadata_fast.py 已直接生成到父目录旁边的 {ds_name}-metadata，无需再 move
    else:
        print(f'{ds_name} 处理失败!')
        print(err)

print('所有后台任务完成。')
