import os
import functools
import shutil
from typing import Callable, Tuple, List, Any

def find_dirs(root, opt=0):
    return sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f)) and (f == 'SPL') == bool(opt)])

def Experiment(func: Callable[[str],Any]):
    @functools.wraps(func)
    def wrapper(root,epoch):
        copy = os.path.join(root,'default')
        root = os.path.join(root,func.__name__)
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root,'A'), exist_ok=True)
        os.makedirs(os.path.join(root,'B'), exist_ok=True)
        os.makedirs(os.path.join(root,'SPL'), exist_ok=True)
        files = sorted([f for f in os.listdir(copy) if f.endswith((".csv",".xlsx"))])
        # for file in files:
        #     path = os.path.join(copy, file)
        #     if 'A' in file:
        #         shutil.copy(src=path,dst=os.path.join(root,'A',file))
        #     elif 'B' in file:
        #         shutil.copy(src=path, dst=os.path.join(root,'B',file))
        #     elif 'level' in file:
        #         shutil.copy(src=path, dst=os.path.join(root, 'SPL',file))
        func(root,epoch)
    return wrapper

def DirectoryProcess(func: Callable[[str], Any]):
    @functools.wraps(func)
    def wrapper(root: str) -> Tuple[List[Any], List[Any]]:
        lst1 = []
        lst2 = []

        # `opt` 값 결정: 함수 이름에 'xlsx'가 포함되어 있는지 확인 xlsx 파일은 SPL기록 파일임
        dirs = find_dirs(root, opt=1 if 'xlsx' in func.__name__ else 0)

        for dir in dirs:
            tmp = root
            root = os.path.join(root, dir)
            result = func(root)
            root = tmp

            # 디렉토리 하나일 때

            if len(dirs) == 1:
                return result

            if isinstance(result, tuple) and len(result) == 2:
                l1, l2 = result
                lst1.append(l1)
                lst2.append(l2)
            else:
                raise ValueError("The function must return a tuple of length 2.")

        return lst1, lst2

    return wrapper
