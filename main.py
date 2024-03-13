import subprocess

def run_streamlit_app():
    command = ["streamlit", "run", "gui/app.py"]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    run_streamlit_app()
