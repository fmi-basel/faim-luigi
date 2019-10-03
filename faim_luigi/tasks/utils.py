import hashlib
import json

RUN_HASH_LENGTH = 24


def task_to_hash(task):
    '''snippet from luigi.task.task_id_str

    '''
    params = task.to_str_params(only_significant=True, only_public=True)
    param_str = json.dumps(params, separators=(',', ':'), sort_keys=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

    if len(param_hash) > RUN_HASH_LENGTH:
        return param_hash[:RUN_HASH_LENGTH]
    if len(param_hash) < RUN_HASH_LENGTH:
        return '_' * (len(param_hash) - RUN_HASH_LENGTH) + param_hash
    return param_hash
