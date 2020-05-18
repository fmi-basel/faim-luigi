import os
import luigi


def dump_task_desc(filename: str, task: luigi.Task) -> None:
    '''dump task parametrization to a file.
    If the folder does not exist, it will create it.
    '''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fout:
        fout.write(str(task))
