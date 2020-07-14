'''Base class for calling knime workflows as part of luigi.

Prerequirements for the workflow:
- Use global flow variables if any parametrization of the call is needed

More information:
- https://www.knime.com/faq#q12
'''
import os
import abc
import subprocess

import luigi

# NOTE This has been tested on Linux only. KNIME might require
# additional options when running on windows or macos.
HEADLESS_ARGS = [
    '-nosplash', '-application', 'org.knime.product.KNIME_BATCH_APPLICATION',
    '--launcher.suppressErrors', '-nosave', '-reset'
]


def _check_if_knime_is_available(executable='knime'):
    '''
    '''
    import shutil
    if shutil.which(executable) is None:
        return False
    return True


def format_workflow_variable_arg(var_name: str,
                                 var_value: str,
                                 var_type: str = 'String') -> str:
    '''generates CLI argument for setting a variable of a knime workflow.

    Note that the knime workflow needs to have the variable with the given
    name as *global* workflow variable.

    '''
    return '-workflow.variable={},{},{}'.format(var_name, var_value, var_type)


def format_workflow_arg(path_to_workflow: str):
    '''generates CLI argument for setting the workflow.

    '''
    if os.path.isdir(path_to_workflow):
        return '-workflowDir={}'.format(path_to_workflow)
    if os.path.splitext(path_to_workflow)[1].lower() in ['.zip', '.knwf']:
        return '-workflowFile={}'.format(path_to_workflow)

    raise RuntimeError(
        'Could not determine CLI argument for workflow: {}'.format(
            path_to_workflow))


class KnimeWrapperTaskBase(luigi.Task):
    '''executes a given KNIME workflow as luigi Task.

    '''

    knime_executable = luigi.Parameter(default='knime')
    '''path to KNIME executable.
    '''
    @property
    @abc.abstractmethod
    def workflow(self) -> str:
        '''provides the workflow*=<location> argument.

        Example implementation could be:

        workflow_path = luigi.Parameter()
        ...
        def workflow(self):
            return format_workflow_arg(self.workflow_path)

        '''

    @property
    def workflow_args(self) -> list:
        '''provides a list of extra workflow arguments. This should cover any
        -workflow.variable=... arguments.

        See also: format_workflow_variable_arg

        '''
        return []

    def compose_call(self) -> list:
        '''creates the full CLI command to run KNIME in headless mode with the
        given workflow.

        '''
        return [
            self.knime_executable,
        ] + HEADLESS_ARGS + [
            self.workflow,
        ] + self.workflow_args

    def run(self):
        '''executes a given KNIME workflow as luigi Task.

        '''
        if not _check_if_knime_is_available(self.knime_executable):
            raise RuntimeError('KNIME executable not found!')
        # NOTE consider making the call interruptible.
        subprocess.run(self.compose_call(), check=True)
