import kagglehub

# Download latest version
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")

print("Path to dataset files:", path)

# or download zip folder from the website
# https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database