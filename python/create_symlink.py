import os
import sys
import shutil

if len(sys.argv) != 3:
    print("Usage: python create_symlink.py <target> <link>")
    sys.exit(1)

target = sys.argv[1]
link = sys.argv[2]

# 如果 link 已存在，删除它 (If the link exists, delete it)
if os.path.islink(link) or os.path.isfile(link):
    os.remove(link)  # 删除符号链接 (symbolic link) / 文件 (file)
elif os.path.isdir(link):
    shutil.rmtree(link)  # 删除目录 (directory)

os.symlink(target, link, target_is_directory=True)
print(f"Created symlink: {link} -> {target}")
