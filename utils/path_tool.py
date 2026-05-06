'''
🤪🤪🤪Author: JY
Date: 2026-04-27 10:55:05
LastEditTime: 2026-04-27 11:05:01
'''
'''
为整个工程提供统一的绝对路径
'''
import os


def get_project_root() -> str:
    #当前文件的绝对路径  utils
    current_file = os.path.dirname(os.path.abspath(__file__))
    #项目根目录,先获取当前文件的目录,再获取目录的父目录  Agent  这里我只需要回到Agent就行
    current_dir = os.path.dirname(current_file)
    #项目根目录    python_ai
    # project_root = os.path.dirname(current_dir)
    return current_dir

def get_abs_path(relative_path: str) -> str:
    '''传递相对路径，得到绝对路径'''

    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path


if __name__ == "__main__":
    print(get_project_root())
    print(get_abs_path(r"config\config.json"))