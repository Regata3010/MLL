from setuptools import find_packages,setup
from typing import List

HYPEN = "-e ."
def get_requirements(file_path:str)->List[str]:
    ''' this function return requirements'''
    
    requirements = []
    with open(file_path) as file_obj:
       requirements = file_obj.readlines()
       [req.replace("\n"," ")for req in requirements] 
       
       if HYPEN in requirements:
           requirements.remove(HYPEN)
           
    return requirements

setup(
    name = 'MLPROJECT',
    version = '0.0.1',
    author = 'Arav Pandey',
    author_email='aravpandey3010@gmail.com',
    package = find_packages(),
    install_requires = get_requirements('requirements.txt')
)