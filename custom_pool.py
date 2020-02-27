import multiprocessing
import multiprocessing.pool

# Sub-class multiprocessing.pool.Pool to allow creation of non-daemon processes
# This is necessary to be able to create processes that can spawn more processes
# This is used in experiments to run multiple instances of the algorithm at once while
# still being able to use multiprocessing to train the NN discriminator


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class CustomPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(CustomPool, self).__init__(*args, **kwargs)
