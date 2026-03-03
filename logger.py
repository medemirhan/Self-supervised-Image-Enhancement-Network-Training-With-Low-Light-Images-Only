import sys

class Logger(object):
    """
    A custom logger that writes output to both the console and a file.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # Use 'w' mode to create a new log file for each run.
        # Specify UTF-8 encoding to prevent UnicodeEncodeError.
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility.
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()