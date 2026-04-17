import pickle
import sys
import time
from contextlib import contextmanager

import psutil

from ..basics import *

__all__ = [
    "setDisplayOptions",
    "jsonLoader",
    "yamlLoader",
    "pklLoader",
    "pklDumper",
    "pipeCellOutput",
    "Timer",
]


def setDisplayOptions():
    np.set_printoptions(precision=4, linewidth=105)
    pd.options.display.max_rows = 7
    pd.options.display.max_columns = 13
    pd.options.display.precision = 4
    pd.options.display.float_format = "{:.4f}".format
    plt.rcParams.update({"font.family": "Charis SIL"})


def jsonLoader(file):
    with open(file, "r") as f:
        return json.load(f)


def yamlLoader(file):
    with open(file, "r") as f:
        return yaml.load(f, yaml.SafeLoader)


def pklLoader(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def pklDumper(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


@contextmanager
def pipeCellOutput(file: str | None = None, mode="w"):
    pipe = file if file else os.devnull
    Path(file).parent.mkdir(parents=True, exist_ok=True) if file else None

    sys.modules["IPython.display"] = None  # type:ignore
    bak_stdout = sys.stdout
    bak_stderr = sys.stderr

    with open(pipe, mode) as f:
        sys.stdout = f
        sys.stderr = f
        try:
            yield
        finally:
            sys.stdout = bak_stdout
            sys.stderr = bak_stderr
            try:
                del sys.modules["IPython.display"]
            except KeyError:
                pass

    print(f"cell output piped to ./{file}") if file else None


class Timer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.t_start = time.time()
        self.t_start_process = psutil.Process(os.getpid()).cpu_times()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        t_end = time.time()
        t_wall = t_end - self.t_start
        if t_wall < 0.1:
            return

        t_end_process = psutil.Process(os.getpid()).cpu_times()
        t_user = t_end_process.user - self.t_start_process.user
        t_system = t_end_process.system - self.t_start_process.system
        t_total = t_user + t_system

        st_wall, st_user, st_system, st_total = map(self._format, [t_wall, t_user, t_system, t_total])
        if self.verbose:
            print(f"*/* cpu times: user {st_user}, system {st_system}, total {st_total} */*")
        print(f"*/* wall time: {st_wall} */*")

    def _format(self, ss):
        mm, ss = divmod(ss, 60)
        return f"{int(mm)}m {ss:.1f}s" if mm else f"{ss:.1f}s"
