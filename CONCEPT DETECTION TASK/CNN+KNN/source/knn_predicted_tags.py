import pandas as pd
df_test = pd.read_csv('/home/Sainte-Genevieve/imageclef2024/test_images.csv')
df_train = pd.read_csv('/home/Sainte-Genevieve/imageclef2024/stratified_data/train_mysplit.csv')

def replace_semicolon_with_comma(code):
    return code.replace(';', ',')


df_train['Code'] = df_train['Code'].apply(replace_semicolon_with_comma)


import pickle
with open('/home/Sainte-Genevieve/imageclef2024/embeddings_foivos/embedding_dict_train_ordered2024.pkl', 'rb') as f:
    embedding_dict_train = pickle.load(f)


with open('/home/Sainte-Genevieve/imageclef2024/embeddings_foivos/embedding_dict_test2024.pkl', 'rb') as f:
    embedding_dict_test = pickle.load(f)

df_test['Embedding'] = df_test['ID'].map(embedding_dict_test)
df_test['ID'] = df_test['ID'].str.rstrip('.jpg')

df_train['Embedding'] = df_train['Image'].map(embedding_dict_train)

import numpy as np
X_subset = np.array(df_train['Embedding'].to_list()).squeeze()
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
results = []
where = 0
for index, row in df_test.iterrows():
    print(where)
    where+=1
    test_image_path = row['ID']
    test_image_embedding = row['Embedding']

    if test_image_embedding is not None:
        knn_subset = NearestNeighbors(n_neighbors=33, metric='cosine')
        knn_subset.fit(X_subset)

        # Find k-nearest neighbors for the test image in the subset
        distances_subset, indices_subset = knn_subset.kneighbors(test_image_embedding, n_neighbors=33)
        neighbor_image_names = df_train.iloc[indices_subset[0]]['Image'].tolist()

        print(f"Closest neighbors for test image: {test_image_path}")
        if neighbor_image_names:  # Check if the list is not empty
            print(neighbor_image_names[0])
        
        # Extract tags from the k-nearest neighbors
        neighbor_tags = []
        for idx in indices_subset[0]:
            tag_value = df_train.iloc[idx]['Code']
            if isinstance(tag_value, str):  # Check if tag is not null
                neighbor_tags.extend(tag_value.split(','))

        # Count the occurrences of each tag
        tag_counts = Counter(neighbor_tags)
        # print(tag_counts)

        # Select the top tag
        top_tag, top_count = tag_counts.most_common(1)[0]

        # If the difference between the first and second most common tag is not big, pick the second tag as well
        second_tag, second_count = tag_counts.most_common(2)[-1]
        threshold = 0.58 
        if (top_count - second_count) / top_count < threshold:
            top_tags = [top_tag, second_tag]
        else:
            top_tags = [top_tag]

        threshold2 = 0.65

        third_tag, third_count = tag_counts.most_common(3)[-1]
        if (top_count - third_count) / top_count < threshold2:
            top_tags.append(third_tag)


        fourth_tag, fourth_count = tag_counts.most_common(4)[-1]
        if (top_count - fourth_count) / top_count < threshold2:
            top_tags.append(fourth_tag)

        
        results.append((test_image_path, ";".join(top_tags)))


result_df = pd.DataFrame(results, columns=['Image', 'Predicted_Tags'])
result_df.to_csv('/home/Sainte-Genevieve/annachatz/concept_detection_csvs/run.csv',sep=',', index=False, header=False)


