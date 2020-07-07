import luigi

from faim_luigi.tasks.knime import KnimeWrapperTaskBase
from faim_luigi.tasks.knime import format_workflow_arg
from faim_luigi.tasks.knime import format_workflow_variable_arg


class SomeKnimeWorkflowRunner(KnimeWrapperTaskBase):
    '''executes a knime workflow that has the following
    global flow variables:

      output_file : path to write output
      input_folder : path to input folder

    '''
    workflow_path = luigi.Parameter()
    output_path = luigi.Parameter()
    input_folder = luigi.Parameter()

    @property
    def workflow(self) -> str:
        return format_workflow_arg(self.workflow_path)

    @property
    def workflow_args(self) -> list:
        return super().workflow_args + [
            format_workflow_variable_arg('input_folder', self.input_folder),
            format_workflow_variable_arg('output_file', self.output_path),
        ]

    def output(self):
        '''
        '''
        return luigi.LocalTarget(self.output_path)
