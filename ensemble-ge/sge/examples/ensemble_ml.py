import importlib
import re
import warnings
from time import gmtime, strftime
from sge.utilities.classifiers import *
from sge.utilities.preprocessing import *
from sge.utilities.ensembles import *
import random
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier
import numpy as np
import multiprocessing as mp
from sklearn.pipeline import Pipeline
import pmlb


KEYWORDS = ['preprocessing', 'classifier','ensemble',]  # Added 'ensemble' to the list of keywords
TIMEOUT = 5*60

def exec_timeout(func, args, timeout):
    # print(args)
    pool = mp.Pool(1, maxtasksperchild=1)
    result = pool.apply_async(func, args)
    pool.close()

    try:
        # print("I")
        s = result.get(timeout)
        # print(s)
        return s
    except mp.TimeoutError:
        print("Exec1")
        pool.terminate()
        return -1.0, -1.0, None

def customwarn(message, category, filename, lineno, file=None, line=None):
    with open("log.txt", "a+") as file:
        file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" : "+  warnings.formatwarning(message, category, filename, lineno)+"\n")

warnings.showwarning = customwarn

class EnsembleML():
    def __init__(self, invalid_fitness=9999999):
        # self.problem = problem
        self.best_pipeline = None
        self.best_fitness = float('inf')

    def process_float(self, value):
        min_value, max_value = map(float, value.replace("'", "").replace('RANDFLOAT(','').replace(')','').split(','))
        return random.uniform(min_value, max_value)

    def process_int(self, value):
        min_value, max_value = map(int, value.replace("'", "").replace('RANDINT(','').replace(')','').split(','))
        return random.randint(min_value, max_value)

    def parse_phenotype(self, phenotype):
        modules = []
        new_module = None
        phenotype = phenotype.replace('  ', ' ').replace('\n', '')

        for pair in phenotype.split(' '):
            keyword, value = pair.split(':')

            if keyword in KEYWORDS:
                if new_module is not None:
                    modules.append(new_module)
                new_module = {'module': keyword, 'module_function': eval(value)}
                # print(new_module)  # Debugging statement

            else:
                try:
                    if 'random' == value:
                        new_module[keyword] = 'random'
                    elif 'RANDFLOAT' in value:
                        new_module[keyword] = self.process_float(value)
                    elif 'RANDINT' in value:
                        new_module[keyword] = self.process_int(value)
                    else:
                        new_module[keyword] = eval(value)
                except NameError:
                    new_module[keyword] = value

        if new_module is not None:
            modules.append(new_module)
            # print(f"Last module: {new_module}")  # Debugging statement

        return modules
    
    def assemble_pipeline(self, modules):

        reordered_config = []
        for module_type in KEYWORDS:
            for module in modules:
                if module['module'] == module_type:
                    reordered_config.append(module)
        pipeline_methods = []
        clsf = []
        pre = []
        en = None
        # print(modules)
        for module in reordered_config:
            # print(module)
            # module_name = module.get('module')
            # print(module['module_function'](module))
            # print(module)
            if module['module'] == 'preprocessing':
                # pre.append(module['module_function'](module))
                pipeline_methods.append(module['module_function'](module))
            # #     # pre.append(module['module_function'](module))
                # print(module['module_function'](module))
            
            if module['module'] == 'classifier':
                # print(module['module_function'](module))
                clsf.append((f"{module['module']}{random.random()}",module['module_function'](module)))
                # print(clsf)
            #     # clasif.append(module['module_function'](module))
            if module['module'] == 'ensemble':
                # print(module['module'])
                # print(module['module_function'](clsf))
                en = module['module_function'](clsf)
                # print(en)
            # print(module['module_function'](module))


            pipeline_methods.append(en)
            pipeline_methods = [item for item in pipeline_methods if item is not None]
            # print(pipeline_methods)


        try:
            # print("IN")
            pipeline = make_pipeline(*pipeline_methods)
            # print(pipeline)
        except Exception:
            print("EX1")
            return -0.5, -0.5, None

        try:
            # print("IN-1")
            # print(ensemble_pipeline)
            # print(pipeline)
            cv = StratifiedKFold(n_splits=3, shuffle=True)
            X, y = pmlb.fetch_data('breast_cancer', return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
            # print(f'scores = {scores}')
        except ValueError as ve:
            # print(pipeline)
            # print("V1")
            return -0.5, -0.5, None
        except MemoryError:
            print("M1")
            return -0.5, -0.5, None

        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        f1_val_test = metrics.f1_score(y_test, y_pred_test, average='weighted')

        # print(np.mean(scores), f1_val_test,pipeline)

        return np.mean(scores), f1_val_test, pipeline.__str__()

    def evaluate(self, individual):
        # print(individual)
        # x = self.parse_individual(individual)
        # x = self.split_config(individual)
        # print(x)
        pipeline_modules = self.parse_phenotype(individual)
        # print(pipeline_modules)
        try:
            f1_val, f1_val_test, pipeline = exec_timeout(func=self.assemble_pipeline, args=[pipeline_modules], timeout=TIMEOUT)
            # f1_val, f1_val_test, pipeline = self.assemble_pipeline([pipeline_modules])
        except:
            # print("EX-2")
            return 9999, {'individual': individual, 'invalid': 1}
    
        fitness = 1 - f1_val

        invalid = 0
        if f1_val < 0:
            invalid = 1

        return fitness, {'individual': individual, 'f1score_val': f1_val, 'f1score_test': f1_val_test, 'pipeline': pipeline, 'invalid': invalid}
    
    


if __name__ == "__main__":
    import sge
    eval_func = EnsembleML()
    # print(eval_func)
    sge.evolutionary_algorithm(evaluation_function=eval_func)
