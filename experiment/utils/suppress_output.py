import os
import sys


class suppress_all_output:
    """
    Suppress all output to stdout and stderr, including C‐level writes.
    Usage:
        with suppress_all_output():
            # nothing printed here will reach the terminal
            some_noisy_function()
        # normal output resumes here
    """
    def __enter__(self):
        # Open a pair of null file descriptors (for stdout and stderr)
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in (0,1)]
        # Save the actual stdout (1) and stderr (2) so we can restore later
        self.save_fds = (os.dup(1), os.dup(2))
        # Duplicate the null fds over stdout and stderr
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the real stdout/stderr fds
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all the fds we opened
        for fd in self.null_fds + list(self.save_fds):
            os.close(fd)

