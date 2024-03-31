import csv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

n_subj = 15 
n_class = 3 
n_chan = 14
n_data = 55000 

# Load EEG data
reader = csv.reader(open("eeg_data_whole_duration_task1.csv", "r"), delimiter=",") # replace 1 with 2-3 depending one which class you want to import data from
data = list(reader)
data = np.array(data).astype("float")
data = np.reshape(data,(n_subj,n_chan,n_data)) #(subjects, channels, data points)

# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data.reshape(n_subj, -1))

for electrode_index in range(n_chan):
    for subject_index in range(n_subj):
        electrode_data = data_std[subject_index, electrode_index * n_data:(electrode_index + 1) * n_data]

        #choose best epsilon parameter
        neighbors = NearestNeighbors(n_neighbors=2)
        neighbors_fit = neighbors.fit(electrode_data.reshape(-1, 1))
        distances, indices = neighbors_fit.kneighbors(electrode_data.reshape(-1, 1))
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.title(f"KNN for Subject {subject_index + 1}, Electrode {electrode_index + 1}")
        plt.savefig(f"epsilon_electrode_{electrode_index}_subject_{subject_index}.png")
        plt.show()

# after running the above comment lines 28-36 and choose the epsilon parameter
        
# DBSCAN parameters
eps = 0.002  # Maximum distance between samples in a neighborhood
min_samples = 4  # Minimum number of samples in a neighborhood for a core point

problematic_electrodes = []

# uncomment the following to apply dbscan algorithm to your data

#         # Apply DBSCAN to the EEG data for each electrode
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         dbscan.fit(electrode_data.reshape(-1, 1)) #reshape(-1, 1)

#         # Identify problematic electrodes (noise points)
#         noise_indices = np.where(dbscan.labels_ == -1)[0]
#         normalData_indices = np.where(dbscan.labels_ != -1)[0]   
#         # if len(noise_indices) > 165:  #165 is the 3% of signal length (55000), which is the normal percentage of outliers in a dataset, if data ~ normal distribution 
#         #     problematic_electrodes.append(electrode_index+1)

#         # # Create a separate figure for each electrode's data
#         # fig = plt.figure(figsize=(8, 4))
#         # plt.scatter(np.arange(len(normalData_indices)), electrode_data[normalData_indices], c=dbscan.labels_[normalData_indices], cmap='coolwarm')
#         # plt.scatter(noise_indices, electrode_data[noise_indices], c='black', s=100, marker='x', label='Outliers')
#         # plt.title(f"Subject {subject_index + 1}, Electrode {electrode_index + 1}")
#         # plt.legend()
#         # Save each figure with a unique name if needed
#         # plt.savefig(f"task3_electrode_{electrode_index}_subject_{subject_index}_whole.png")
#         # plt.show() 





