import sys


class TeeOutput:
    """Class to write output to both console and file simultaneously."""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = None
        if file_path is not None:
            self.log_file = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()  # Ensure immediate writing to file

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()

