### Define next the folder to create the hooks files and the warning file to read the modules from
output_hooks_dir = 'hooks'
warning_file = 'build/VCToolbox/warn-VCToolbox.txt'

import re
import os
import shutil

shutil.rmtree(output_hooks_dir, ignore_errors=True)
os.makedirs(output_hooks_dir, exist_ok=True)

with open(warning_file) as file:
    files_content = file.readlines()

clean_content = []
for line in files_content:
    if re.search('missing module named',line):
        temp_line = re.sub('.*imported by ','',line)
        temp_line = re.sub('\n',', ',temp_line)
        clean_content.append(temp_line)
clean_content = list(set(clean_content))
joined_content = ''.join(clean_content)
clean_content = list(set(joined_content.split('), ')))

modules_toplevel = []
for line in clean_content:
    if re.search('top-level',line):
        temp_mod = re.sub(' \(.*','',line)
        temp_mod = re.sub('\..*','',temp_mod)
        modules_toplevel.append(temp_mod)
modules_toplevel = list(set(modules_toplevel))

modules_conditional = []
for line in clean_content:
    if re.search('conditional',line):
        temp_mod = re.sub(' \(.*','',line)
        temp_mod = re.sub('\..*','',temp_mod)
        modules_conditional.append(temp_mod)
modules_conditional = list(set(modules_conditional))

modules_delayed = []
for line in clean_content:
    if re.search('delayed',line):
        temp_mod = re.sub(' \(.*','',line)
        temp_mod = re.sub('\..*','',temp_mod)
        modules_delayed.append(temp_mod)
modules_delayed = list(set(modules_delayed))

modules_optional = []
for line in clean_content:
    if re.search('optional',line):
        temp_mod = re.sub(' \(.*','',line)
        temp_mod = re.sub('\..*','',temp_mod)
        modules_optional.append(temp_mod)
modules_optional = list(set(modules_optional))

all_modules = modules_toplevel + modules_conditional + modules_delayed + modules_optional
all_modules = list(set(all_modules))

print(all_modules)
print('Number of found modules:', len(all_modules))

### Optional: remove any of the modules
remove_list = [
    '/usr/local/lib/python3',
    '/Users/sotomi/envs/pyinstaller-env/lib/python3',
    'zipimport',
    'test',
]
add_list = [
    'sacremoses',
    'librosa',
]
for pkg in remove_list:
    if pkg in all_modules:
        all_modules.remove(pkg)
for pkg in add_list:
    if pkg not in all_modules:
        all_modules.append(pkg)

print('Total number of requested modules:', len(all_modules))

### Optional: Change all_modules by any of the other lists, e.g. modules_toplevel
for module in all_modules:
    output_content = 'from PyInstaller.utils.hooks import collect_all\n\ndatas, binaries, hiddenimports = collect_all(\''+module+'\')'
    with open(output_hooks_dir+'/hook-'+str(module)+'.py', 'w') as f:
        f.write(output_content)
