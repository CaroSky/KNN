import pandas as pd
from sklearn.model_selection import train_test_split

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df_iris = pd.read_csv(URL, header=None, names=['sepal length', 'sepal width',
                             'petal length', 'petal width', 'class'])

K = 3


# 1. Distance Calculation
def euclidean_distance(p, q):
    return sum([(a - b) ** 2 for a, b in zip(p, q)]) ** 0.5

# 2. Finding the Nearest Neighbors
def find_nearest_neighbors(training_data, test_instance, k):
    distances = []
    for index, train_instance in training_data.iterrows():
        dist = euclidean_distance(test_instance, train_instance[:-1].values)
        distances.append((index, dist))
    sorted_distances = sorted(distances, key=lambda x: x[1])
    nearest_neighbors = sorted_distances[:k]
    return nearest_neighbors

# 3. Make Predictions
def predict(training_data, test_instance, k):
    nearest_neighbors = find_nearest_neighbors(training_data, test_instance, k)
    top_labels = [training_data.iloc[index].iloc[-1] for index, _ in nearest_neighbors]

    class_frequency = {}
    for label in top_labels:
        if label in class_frequency:
            class_frequency[label] += 1
        else:
            class_frequency[label] = 1

    # Using max function to get the class with maximum frequency
    return max(class_frequency, key=class_frequency.get)


# Split data into 90% training and 10% test
train_data, test_data = train_test_split(df_iris, test_size=0.10, random_state=42)
train_data = train_data.reset_index(drop=True)

correct_predictions = 0

for _, instance in test_data.iterrows():
    test_instance = instance[:-1].values
    true_class = instance.iloc[-1]
    predicted_class = predict(train_data, test_instance, K)

    if predicted_class == true_class:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data) * 100
print(f"Accuracy: {accuracy:.2f}%")

new_dp = [7.0, 3.1, 1.3, 0.7]
predicted_class = predict(train_data, new_dp, K)
print(f"The predicted class for data point {new_dp} is: {predicted_class}")
