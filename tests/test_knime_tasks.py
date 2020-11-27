import luigi
from queue import Queue
import pandas as pd

from faim_luigi.tasks.knimepy import KnimeRunnerTaskBase

FAILQUEUE = Queue()


def test_dummy_workflow(tmpdir):
    # create list of files
    # start workflow with input list of files
    result = luigi.build([
        DummyKnimeWorkflowTask(
            dummy_workflow_location='tests/res/MinimalContainerWorkflow',
            filelist=[str(tmpdir / 'file1.test'),
                      str(tmpdir / 'file2.test')])
    ],
        local_scheduler=True,
        detailed_summary=True)
    if result.status not in [
        luigi.execution_summary.LuigiStatusCode.SUCCESS,
        luigi.execution_summary.LuigiStatusCode.SUCCESS_WITH_RETRY
    ]:
        raise RuntimeError(
            'Luigi failed to run the workflow! Exit code: {}'.format(result))
    assert (tmpdir / 'results.csv').exists()


class DummyKnimeWorkflowTask(KnimeRunnerTaskBase):
    dummy_workflow_location = luigi.Parameter()
    filelist = luigi.Parameter()

    @property
    def workflow_path(self) -> str:
        return self.dummy_workflow_location

    def input_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.filelist, columns=["filepath"])


@DummyKnimeWorkflowTask.event_handler(luigi.Event.FAILURE)
def fail(task, exception):
    FAILQUEUE.put(task, exception)
