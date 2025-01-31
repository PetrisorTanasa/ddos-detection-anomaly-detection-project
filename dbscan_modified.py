# import re
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.cluster import DBSCAN
# from datetime import datetime
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from scipy.stats import entropy
# from sklearn.decomposition import PCA

# #import IsoaltionForest
# from sklearn.ensemble import IsolationForest
# # Load data from a file
# def load_logs(file_path):
#     with open(file_path, 'r') as f:
#         return f.readlines()

# # Regular expression to parse logs

# def parse_log(log):
#     # Split the log string into parts
#     parts = log.split(" ")

#     # Extract IP address
#     ip = parts[0] if parts[0] != '-' else None
    
#     # Extract timestamp (it's enclosed in square brackets, so split the timestamp part further)
#     # timestamp_part = parts[3]  # Removing the "[" and "]"
#     #  + " " + parts[4][:-1]
#     timestamp = parts[3] 
#     # datetime.strptime(timestamp_part, "%d/%b/%Y:%H:%M:%S")

#     # Extract method, path, status, size, referrer, and agent
#     method = parts[5][1:]  # Remove the quote around the method
#     path = parts[6]        # Path is always between the method and the HTTP version
#     status = parts[8]
#     size = parts[9] if parts[9] != '-' else None
#     referrer = parts[10][1:-1]  # Remove quotes around the referrer
#     agent = parts[11][1:-1]     # Remove quotes around the user-agent

#     return {
#         "ip": ip,
#         "timestamp": timestamp,
#         "method": method,
#         "path": path,
#         "status": status,
#         "size": size,
#         "referrer": referrer,
#         "agent": agent
#     }

# # Parse logs into structured data
# def parse_logs(data):
#     parsed_logs = [parse_log(log) for log in data]
#     return [log for log in parsed_logs if log is not None]

# import numpy as np

# import pandas as pd
# from datetime import datetime
# from scipy.stats import entropy

# def extract_features(parsed_logs):
#     features = []
#     for log in parsed_logs:
#         timestamp = datetime.strptime(log['timestamp'], "%d/%b/%Y:%H:%M:%S").timestamp()
#         ip = log['ip']
#         method = log['method']
#         path = log['path']
#         status = int(log['status'])
#         try:
#             size = int(log['size'])
#         except:
#             size = 0
#         path_length = len(path)
#         path_depth = path.count("/")
#         encoded_characters = sum(1 for c in path if c == '%')
#         response_size_category = 'small' if size < 1000 else 'medium' if size < 10000 else 'large'
#         features.append({
#             'ip': ip,
#             'timestamp': timestamp,
#             'method': method,
#             'path': path,
#             'path_length': path_length,
#             'path_depth': path_depth,
#             'encoded_characters': encoded_characters,
#             'status': status,
#             'size': size,
#             'response_size_category': response_size_category
#         })
#     return pd.DataFrame(features)

# def compute_entropy_for_window(window):
#     # Compute the entropy of the PCA components in the window
#     pca_counts = window.apply(lambda x: x.value_counts()).fillna(0)
#     return entropy(pca_counts)

# def entropy_feature(features_df):
#     # Convert timestamp to datetime for resampling
#     features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], unit='s')
    
#     # Encode categorical features
#     categorical_features = features_df[['ip', 'method', 'path', 'response_size_category']]
#     encoder = OneHotEncoder()
#     encoded_categorical = encoder.fit_transform(categorical_features).toarray()
    
#     # Normalize numerical features
#     numerical_features = features_df[['status', 'size', 'path_length', 'path_depth', 'encoded_characters']]
#     scaler = StandardScaler()
#     scaled_numerical = scaler.fit_transform(numerical_features)
    
#     # Combine all features
#     all_features = np.hstack([encoded_categorical, scaled_numerical])
    
#     # Apply PCA to reduce dimensionality
#     pca = PCA(n_components=2)
#     pca_features = pca.fit_transform(all_features)
    
#     # Create a DataFrame with PCA components and timestamp
#     pca_df = pd.DataFrame(pca_features, columns=['pca1', 'pca2'])
#     pca_df['timestamp'] = features_df['timestamp']
    
#     # Resample the data by one-second intervals and compute entropy for each window
#     entropy_per_window = pca_df.resample('1s', on='timestamp').apply(compute_entropy_for_window)
    
#     # Reset index to return just the entropy values
#     entropy_per_window = entropy_per_window.reset_index().rename(columns={0: 'entropy'})
    
#     return np.array(entropy_per_window['entropy'])


# def modified_dbscan(data, epsilon, MinPts):
#     labels = [-1] * len(data)
#     cluster_id = 0

#     def range_query(data, point, epsilon):
#         neighbors = []
#         for i in range(len(data)):
#             if np.linalg.norm(data[i] - point) <= epsilon:
#                 neighbors.append(i)
#         return neighbors

#     def expand(data, point_idx, cluster_id, epsilon, MinPts):
#         seeds = range_query(data, data[point_idx], epsilon)
#         if len(seeds) < MinPts:
#             labels[point_idx] = -1
#             return False
#         else:
#             labels[point_idx] = cluster_id
#             for seed in seeds:
#                 labels[seed] = cluster_id
#             while seeds:
#                 current_point_idx = seeds[0]
#                 current_neighbors = range_query(data, data[current_point_idx], epsilon)
#                 if len(current_neighbors) >= MinPts:
#                     for i in range(len(current_neighbors)):
#                         result_point_idx = current_neighbors[i]
#                         if labels[result_point_idx] == -1:
#                             labels[result_point_idx] = cluster_id
#                         if labels[result_point_idx] == -1:
#                             seeds.append(result_point_idx)
#                 seeds = seeds[1:]
#             return True

#     for point_idx in range(len(data)):
#         if labels[point_idx] == -1:  # UNCLASSIFIED
#             if expand(data, point_idx, cluster_id, epsilon, MinPts):
#                 cluster_id += 1

#     # Calculate centroids
#     centroids = []
#     for cid in range(cluster_id):
#         cluster_points = [data[i] for i in range(len(data)) if labels[i] == cid]
#         centroid = np.mean(cluster_points, axis=0)
#         centroids.append(centroid)

#     return labels, centroids

# # Example usage
# # if __name__ == "__main__":
# #     log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log"
# #     data = load_logs(log_file)
# #     parsed_logs = parse_logs(data)
# #     features_df = extract_features(parsed_logs)
# #     features_with_entropy = add_entropy_feature(features_df)
# #     print(features_with_entropy.head())

# if __name__ == "__main__":
#     log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log"  # Replace with your log file path
#     log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log.2025-01-10"
#     log_file = "/Users/ptanasa/Desktop/DDoS detection/small_dataset_10Jan2025.txt"
#     data = load_logs(log_file)
#     parsed_logs = parse_logs(data)
#     features_df = extract_features(parsed_logs)
#     entropy_values = entropy_feature(features_df)

#     # use • sklearn.model selection.train test split to split entropy_values
#     # into training and testing sets.
#     from sklearn.model_selection import train_test_split
#     X_train, X_test = train_test_split(entropy_values, test_size=0.33)

#     # Do clustering on X train

#     clusters, centroids = modified_dbscan(X_train, epsilon=0.1, MinPts=5)
#     #plot clusters and centroids
#     print(X_train[0][0])

#     plot_points_x = []
#     plot_points_y = []
#     for i in range(len(X_train)):
#         plot_points_x.append(X_train[i][0])
#         plot_points_y.append(X_train[i][1])
#     # Plot the clusters
#     plt.figure(figsize=(10, 7))
#     plt.scatter(plot_points_x, plot_points_y, c=clusters, cmap='viridis', label='Data Points')
#     for i, centroid in enumerate(centroids):
#         plt.scatter(centroid[:][0], centroid[:][1], c='red', marker='x', s=100, label=f'Centroid {i}' if i == 0 else "")
#     plt.xlabel('Index')
#     plt.ylabel('Entropy')
#     plt.title("DBSCAN Clustering")
#     plt.legend()
#     plt.show()

#     print(clusters)
#     print(centroids)


#     print(entropy_values.head())
#     print(len(entropy_values))
import re
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data from a file
def load_logs(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

# Regular expression to parse logs
def parse_log(log):
    parts = log.split(" ")
    ip = parts[0] if parts[0] != '-' else None
    timestamp = parts[3]
    method = parts[5][1:]
    path = parts[6]
    status = parts[8]
    size = parts[9] if parts[9] != '-' else None
    referrer = parts[10][1:-1]
    agent = parts[11][1:-1]
    return {
        "ip": ip,
        "timestamp": timestamp,
        "method": method,
        "path": path,
        "status": status,
        "size": size,
        "referrer": referrer,
        "agent": agent
    }

# Parse logs into structured data
def parse_logs(data):
    parsed_logs = [parse_log(log) for log in data]
    return [log for log in parsed_logs if log is not None]

def extract_features(parsed_logs, hour):
    features = []
    for log in parsed_logs:

        # doesnt start with 10.68.
        if log["ip"].startswith("10.68.") or log['timestamp'].split(":")[1] != hour:
            continue
        timestamp = datetime.strptime(log['timestamp'], "%d/%b/%Y:%H:%M:%S").timestamp()
        
        ip = log['ip']
        method = log['method']
        path = log['path']
        status = int(log['status'])
        try:
            size = int(log['size'])
        except:
            size = 0
        path_length = len(path)
        path_depth = path.count("/")
        encoded_characters = sum(1 for c in path if c == '%')
        response_size_category = 'small' if size < 1000 else 'medium' if size < 10000 else 'large'
        features.append({
            'ip': ip,
            'timestamp': timestamp,
            'method': method,
            'path': path,
            'path_length': path_length,
            'path_depth': path_depth,
            'encoded_characters': encoded_characters,
            'status': status,
            'size': size,
            'response_size_category': response_size_category
        })
    return pd.DataFrame(features)

def compute_entropy_for_window(window):
    value_counts = window.apply(lambda x: x.value_counts()).fillna(0)

    # Normalize the value counts to get probabilities
    probabilities = value_counts.div(value_counts.sum(axis=0), axis=1)

    # Compute the entropy for each feature
    feature_entropies = probabilities.apply(lambda x: entropy(x, base=2), axis=0)

    # Compute the joint entropy for pairs of features
    joint_entropies = []
    columns = window.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            # Create a joint probability distribution
            joint_values = window[[columns[i], columns[j]]].value_counts(normalize=True)
            joint_entropies.append(entropy(joint_values, base=2))

    # Combine the feature entropies and joint entropies
    combined_entropy = feature_entropies.sum() + (np.mean(joint_entropies) if joint_entropies else 0)

    return combined_entropy

def entropy_feature(features_df):
    # Convert timestamp to datetime for resampling
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], unit='s')
    
    # Encode categorical features
    categorical_features = features_df[['ip', 'method', 'path', 'response_size_category']]
    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(categorical_features).toarray()
    
    # Normalize numerical features
    numerical_features = features_df[['status', 'size', 'path_length', 'path_depth', 'encoded_characters']]
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    
    # Combine all features
    all_features = np.hstack([encoded_categorical, scaled_numerical])
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(all_features)
    
    # Create a DataFrame with PCA components and timestamp
    pca_df = pd.DataFrame(pca_features, columns=['pca1', 'pca2', 'pca3'])
    pca_df['timestamp'] = features_df['timestamp']
    
    # Resample the data by one-second intervals and compute entropy for each window
    entropy_per_window = pca_df.resample('1s', on='timestamp').apply(compute_entropy_for_window)
    
    # Reset index to return just the entropy values
    entropy_per_window = entropy_per_window.reset_index().rename(columns={0: 'entropy'})
    
    return np.array(entropy_per_window['entropy'])

class ModifiedDBSCAN:
    def __init__(self, epsilon=0.1, min_pts=5, labels=None):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.labels = labels
        self.centroids = None

    def fit(self, data):
        self.labels = ["UNCLASSFED"] * len(data)  # Initialize all points as UNCLASSIFIED
        #cluster_id = max(self.labels) + 1 if self.labels else 0
        cluster_id = 0

        def range_query(data, point, epsilon):
            neighbors = []
            for i in range(len(data)):
                if np.linalg.norm(data[i] - point) <= epsilon:
                    neighbors.append(i)
            return neighbors

        def expand(data, x_idx, cid, epsilon, min_pts):
            # Perform range query to find neighbors of x
            S = [i for i in range(len(data)) if np.linalg.norm(data[x_idx] - data[i]) <= epsilon]
            
            # If not enough data in the neighborhood of x, label x as NOISE and return false
            if len(S) < min_pts:
                self.labels[x_idx] = "NOISE"  # NOISE
                return False
            
            # Label all points in S with the current cluster ID
            for x_prime_idx in S:
                if self.labels[x_prime_idx] == "UNCLASSFED":  # UNCLASSIFIED
                    self.labels[x_prime_idx] = cid
            
            # Remove x from S
            S.remove(x_idx)
            
            # Iterate over all points in S
            while S:
                x_prime_idx = S[0]
                
                # Perform range query to find neighbors of x'
                T = [i for i in range(len(data)) if np.linalg.norm(data[x_prime_idx] - data[i]) <= epsilon]
                
                # If enough data in the neighborhood of x', expand the cluster
                if len(T) >= min_pts:
                    for y_idx in T:
                        if self.labels[y_idx] == "UNCLASSFED" or self.labels[y_idx] == "NOISE":  # UNCLASSIFIED or NOISE
                            if self.labels[y_idx] == "UNCLASSFED":  # UNCLASSIFIED
                                S.append(y_idx)
                            self.labels[y_idx] = cid
                
                # Remove x' from S
                S.remove(x_prime_idx)
            
            return True

        for point_idx in range(len(data)):
            if self.labels[point_idx] == "UNCLASSFED":  # UNCLASSIFIED
                if expand(data, point_idx, cluster_id, self.epsilon, self.min_pts):
                    cluster_id += 1

        # Calculate centroids
        cluster_points = [data[i] for i in range(len(data)) if self.labels[i] == "NOISE"]
        centroid = np.mean(cluster_points, axis=0)
        print(centroid)
        self.centroids = [centroid]
        for cid in range(cluster_id):
            cluster_points = [data[i] for i in range(len(data)) if self.labels[i] == cid]
            centroid = np.mean(cluster_points, axis=0)
            self.centroids.append(centroid)

        return self.labels, self.centroids

    def predict(self, data):
        predictions = []
        for point in data:
            min_dist = float('inf')
            label = -1

            # Because the centroids
            for cid, centroid in enumerate(self.centroids):
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    label = cid
            predictions.append(label)
        return predictions


# Example usage
if __name__ == "__main__":
    log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log"  # Replace with your log file path
    log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log.2025-01-10"
    log_file = "/Users/ptanasa/Desktop/DDoS detection/small_dataset_10Jan2025.txt"
    log_file = "/Users/ptanasa/Desktop/DDoS detection/access.log.2025-01-11"
    data = load_logs(log_file)
    parsed_logs = parse_logs(data)


    classes_per_hour = {}
    for i in range(0,24):
        i = format(i, '02d')
        print("Hour ", i)
        features_df = extract_features(parsed_logs, i)

        if(len(features_df) ==  0):
            continue
        entropy_values = entropy_feature(features_df)

        X_train, X_test = train_test_split(entropy_values, test_size=0.33)
        
        import numpy as np
        from sklearn.cluster import KMeans

        X_train_copy = X_train.copy().reshape(-1, 1)
        labels_kmeans = KMeans().fit(X_train_copy).predict(X_train_copy)

        model = ModifiedDBSCAN(epsilon=0.12, min_pts=3, labels=labels_kmeans)
        model.fit(X_train)

        predictions = model.predict(X_test)
        cluster_counts = {}
        for label in predictions:
            if label not in cluster_counts:
                cluster_counts[label] = 0
            cluster_counts[label] += 1
        print(cluster_counts)

        #make a folder to save the plots
        if not os.path.exists("plots_" + log_file):
            os.makedirs("plots_" + log_file)

        #plot cluster counts
        plt.figure(figsize=(10, 7))
        plt.bar(cluster_counts.keys(), cluster_counts.values())
        plt.xlabel('Cluster ID')
        plt.ylabel('Count')
        plt.title('Cluster Counts')
        #plt.show()

        #save plot
        plt.savefig(f"plots_{log_file}/cluster_counts_Hour{i}.png")

        #plot 1d predictions
        plt.figure(figsize=(10, 7))
        plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='viridis')
        plt.xlabel('Index')
        plt.ylabel('Cluster ID')
        plt.title('Cluster Assignments')
        #plt.show()

        #save plot
        plt.savefig(f"plots_{log_file}/cluster_assignments_Hour{i}.png")