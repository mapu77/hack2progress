from threading import Thread


class FunctionThread(Thread):
    def __init__(self, target, *args):
        Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)