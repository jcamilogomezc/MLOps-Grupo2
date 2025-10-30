FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install JupyterLab and Jupyter Notebook
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipywidgets

# Install NumPy and core scientific packages
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    ydata-profiling

# Install additional scientific packages (scipy-notebook equivalents)
RUN pip install --no-cache-dir \
    scikit-learn \
    scikit-image \
    sympy \
    statsmodels \
    numexpr \
    patsy \
    cloudpickle \
    dill \
    numba \
    bokeh \
    sqlalchemy \
    h5py \
    vincent \
    beautifulsoup4 \
    protobuf \
    xlrd \
    openpyxl

WORKDIR /home/jovyan/work

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=admin", "--NotebookApp.password=admin", "--allow-root"]