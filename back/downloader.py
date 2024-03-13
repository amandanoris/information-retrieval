import nltk

def program():
    """
    Launches the NLTK download GUI, allowing the user to download additional NLTK data packages.

    This function does not take any arguments and does not return any value. It simply invokes the NLTK download GUI, which is a graphical user interface for downloading NLTK data packages.

    Note: This function is intended for use in environments where a graphical user interface is available.
    """
    nltk.download_gui()

program()
