import os
from types import SimpleNamespace


class ProcessLocal(object):
    def __init__(self):
        self._local = SimpleNamespace()

    def _thread_init(self):
        pid = os.getpid()
        if not hasattr(self._local, '_pid'):
            self._local._pid = pid

        if self._local._pid != pid:
            # not hit by coverage (multiprocessing)
            self._local = SimpleNamespace()  # pragma: no cover

    def __getattr__(self, item):
        if item == '_local':
            return super(ProcessLocal, self).__getattr__(item)
        self._thread_init()
        return getattr(self._local, item)

    def __setattr__(self, item, value):
        if item == '_local':
            super(ProcessLocal, self).__setattr__(item, value)
        else:
            self._thread_init()
            setattr(self._local, item, value)
