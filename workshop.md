# Installation steps for workshop

## Windows

1. If you have not done already, done and install [Git](https://git-scm.com/downloads/win)
2. Download and install [Visual Studio Code](https://code.visualstudio.com/download)
3. Download and install [Conda](https://conda-forge.org/download/). See further instructions [here](https://github.com/conda-forge/miniforge)
4. Run the following command in Command Prompt
   ```bash
   git clone https://github.com/rootmemo/pyrootmemo.git
   ```
5. Create a virtual environment with Python
   ```bash
   conda create -n rrmm python
   ```
6. Once the environment is created, activate the environment
   ```bash
   conda activate rrmm
   ```
7. Navigate to the cloned repository and install pyrootmemo using the following command
   ```bash
   cd pyrootmemo
   pip install -e .
   ```
