"""Base class for calling knime workflows via knimepy as part of luigi.

Requirement for the workflow:
- Containerized inputs

More information:
- https://github.com/knime/knimepy
- https://www.knime.com/blog/knime-and-jupyter
"""
import abc
import knime
import pandas as pd

import luigi


class KnimeRunnerTaskBase(luigi.Task):
    """executes a given KNIME workflow as luigi Task.

    """
    knime_executable = luigi.Parameter(default='knime')
    knime_workflow = luigi.Parameter

    @property
    @abc.abstractmethod
    def workflow_path(self) -> str:
        """provides the workflow*=<location> argument.

        Example implementation could be:

        workflow_path = luigi.Parameter()
        ...
        def workflow(self):
            return format_workflow_arg(self.workflow_path)

        """

    @property
    def input_df(self) -> pd.DataFrame:
        """provides a list of extra workflow arguments. This should cover any
        -workflow.variable=... arguments.

        See also: format_workflow_variable_arg

        """
        return pd.DataFrame({})

    def run(self):
        """executes a given KNIME workflow as luigi Task.

        """
        knime.executable_path = self.knime_executable
        with knime.Workflow(workflow_path=self.workflow_path) as wf:
            print(wf.data_table_inputs)
            wf.data_table_inputs[0] = self.input_df()
            wf.execute()

    def output(self):
        """TODO retrieve output LocalTarget?

        """