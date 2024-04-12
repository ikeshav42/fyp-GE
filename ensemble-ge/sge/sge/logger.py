import numpy as np
from sge.parameters import params
import json
import os



def evolution_progress(generation, pop):
    # print(generation)
    # print(pop)
    fitness_samples = [i['fitness'] for i in pop]
    print(fitness_samples)
    val1 = [i['other_info'] for i in pop]
    f1score_test = []
    f1score_val = []
    for i in val1:
        if i['invalid'] == 0:
            f1score_val.append(i['f1score_val'])
            f1score_test.append(i['f1score_test'])
    # f1score_val = val1[0]['f1score_val']
    # f1score_test = val1[0]['f1score_test']
    if f1score_val:
        max_f1score_val = np.max(f1score_val)
    else:
        max_f1score_val = -1  # or any default value you prefer
    # data = '%4d\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples))
    data = '%4d\t%.6e\t%.6e\t%.6e\t%.4f\t%.4f\t%.4f\t%.4f' % (
    generation, 
    np.min(fitness_samples), 
    np.mean(fitness_samples), 
    np.std(fitness_samples),
    max_f1score_val,
    np.mean(f1score_val),
    np.max(f1score_test),
    np.mean(f1score_test)
)
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    c = json.dumps(population)
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)
    # pass


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()

def get_best(bf,bg,bp,f1v):
    fit = bf
    gen = bg
    pipe = bp
    f1_val = f1v
    print(f'Best fitness : {fit}')
    print(f'Best Gen : {gen}')
    print(f'Best Pipeline : {pipe}')
    print(f'Best F1 : {f1_val}')
