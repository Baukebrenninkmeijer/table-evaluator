from IPython.core.display import HTML, display, Markdown
from IPython import get_ipython
import ipywidgets as widgets
from typing import Dict


class EvaluationResult(object):
    def __init__(self, name, content, prefix=None, appendix=None, notebook=False):
        self.name = name
        self.prefix = prefix
        self.content = content
        self.appendix = appendix
        self.notebook = notebook

    def show(self):
        if self.notebook:
            output = widgets.Output()
            with output:
                display(Markdown(f'## {self.name}'))
                if self.prefix: display(Markdown(self.prefix))
                display(self.content)
                if self.appendix: display(Markdown(self.appendix))
            return output

        else:
            print(f'\n{self.name}')
            if self.prefix: print(self.prefix)
            print(self.content)
            if self.appendix: print(self.appendix)


def visualize_notebook(table_evaluator, overview, privacy_metrics, ml_efficacy, statistical):
    dashboards = []
    for tab in [overview, privacy_metrics, ml_efficacy, statistical]:
        plots = []
        for evaluation_report in tab:
            evaluation_report.notebook = True
            plots.append(evaluation_report.show())
        if plots:
            dashboards.append(widgets.VBox(plots))
    display(HTML(f'<h1 style="text-align: center">Synthetic Data Report</h1>'))
    tab = widgets.Tab(dashboards)
    tab.set_title(0, 'Overview')
    tab.set_title(1, 'Privacy Metrics')
    tab.set_title(2, 'ML Efficacy')
    tab.set_title(3, 'Statistical Metrics')
    display(tab)


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__  #This works due to the except below
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
