try:
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.core.display import HTML, Markdown
    from IPython.display import display
except ImportError:
    print('IPython not installed.')


class EvaluationResult:
    def __init__(
        self,
        name: str,
        content: str,
        prefix: str | None = None,
        appendix: str | None = None,
        *,
        notebook: bool = False,
    ) -> None:
        self.name = name
        self.prefix = prefix
        self.content = content
        self.appendix = appendix
        self.notebook = notebook

    def show(self) -> widgets.Output | None:
        if self.notebook:
            output = widgets.Output()
            with output:
                display(Markdown(f'## {self.name}'))
                if self.prefix:
                    display(Markdown(self.prefix))
                display(self.content)
                if self.appendix:
                    display(Markdown(self.appendix))
            return output

        print(f'\n{self.name}')
        if self.prefix:
            print(self.prefix)
        print(self.content)
        if self.appendix:
            print(self.appendix)
        return None


def visualize_notebook(
    overview: list[EvaluationResult],
    privacy_metrics: list[EvaluationResult],
    ml_efficacy: list[EvaluationResult],
    statistical: list[EvaluationResult],
) -> None:
    dashboards = []
    for tab in [overview, privacy_metrics, ml_efficacy, statistical]:
        plots = []
        for evaluation_report in tab:
            evaluation_report.notebook = True
            plots.append(evaluation_report.show())
        if plots:
            dashboards.append(widgets.VBox(plots))
    display(HTML('<h1 style="text-align: center">Synthetic Data Report</h1>'))
    tab = widgets.Tab(dashboards)
    tab.set_title(0, 'Overview')
    tab.set_title(1, 'Privacy Metrics')
    tab.set_title(2, 'ML Efficacy')
    tab.set_title(3, 'Statistical Metrics')
    display(tab)


def isnotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # This works due to the except below
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        if shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False
    except NameError:
        return False  # Probably standard Python interpreter
