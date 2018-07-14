import os
import re
import shutil
import glob

frameworks = ['tf', 'pytorch', 'dynet', 'dy']
frameworks = '({})'.format("|".join(frameworks))

tasks = ['classify', 'tagger', 'lm', 'seq2seq']
tasks = '({})'.format("|".join(tasks))

pid = r'[0-9]{2,4}'

pytorch_model = re.compile(r".*\.pyt$")
log = re.compile(r".*\.log$")
tf_model_dir = re.compile(r"tf-{}-{}".format(tasks, pid))
model_file = re.compile(r"{}-model-{}-{}.*".format(tasks, frameworks, pid))
checkpoint = re.compile("checkpoint")
conll = re.compile("^conll(-(bio|iobes)-)?results.conll$")
twpos = re.compile("^twposresults.conll$")

pyc = re.compile(r".*\.pyc$")
pycache = re.compile(r"^.*/__pycache__/.*$")
test_file = re.compile(r"^.*/test_data/.*$")

res = [
    log,
    pytorch_model,
    tf_model_dir,
    model_file,
    checkpoint,
    conll,
    twpos,
]

def delete(file_):
    try:
        if os.path.isdir(file_):
            shutil.rmtree(file_)
        else:
            os.remove(file_)
    except OSError:
        pass

base = os.path.dirname(os.path.realpath(__file__))


for path, dirs, files in os.walk('.'):
    for file_name in files:
        file_ = os.path.join(path, file_name)
        # Skip test files
        if test_file.match(file_):
            continue
        # Delete files that are generated by training
        if any(r.match(file_name) for r in res):
            delete(file_)
        # Delete .pyc files not in __pycache__ (created by python 27)
        if pyc.match(file_name) and not pycache.match(file_):
            delete(file_)