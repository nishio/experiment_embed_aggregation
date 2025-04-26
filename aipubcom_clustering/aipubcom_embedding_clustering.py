
import csv
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
CSV_PATH = '/home/ubuntu/repos/aipubcom-data/aipubcom_comments.csv'
OUTPUT_DIR = '/home/ubuntu/repos/kouchou-ai/experimental/aipubcom_clustering'
DISTANCE_THRESHOLD = 0.4  # クラスタリングの距離閾値、調整可能
BATCH_SIZE = 1000  # メモリ使用量を制御するためのバッチサイズ

def load_data(csv_path):
    """CSVからデータを読み込み、重複を除去する"""
    print(f"Loading data from {csv_path}...")
    comments = []
    ids = []
    
    already_seen = set()
    already_seen.add("")  # 空白だけのコメントを除去するために、予め追加しておく
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダーをスキップ
        
        for i, row in enumerate(reader):
            if len(row) >= 1:
                comment = row[0]  # commentカラム
                if comment not in already_seen:
                    already_seen.add(comment)
                    comments.append(comment)
                    ids.append(i)
    
    print(f"Loaded {len(comments)} unique comments.")
    return comments, ids

def create_embeddings(comments, model_name, batch_size=1000):
    """コメントの埋め込みベクトルを生成する"""
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    embeddings = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(comments)-1)//batch_size + 1}")
    
    return np.array(embeddings)

def perform_clustering(embeddings, distance_threshold=0.4):
    """凝集クラスタリングを実行する"""
    print(f"Performing agglomerative clustering with distance_threshold={distance_threshold}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='complete'
    )
    
    labels = clustering.fit_predict(embeddings)
    print(f"Found {len(set(labels))} clusters.")
    
    distances = clustering.distances_
    children = clustering.children_
    
    return labels, distances, children

def find_closest_pair(embeddings):
    """最も近いテキストペアを見つける"""
    print("Finding the closest pair of documents...")
    n = embeddings.shape[0]
    min_distance = float('inf')
    closest_pair = (-1, -1)
    
    for i in range(0, n, BATCH_SIZE):
        batch_i = embeddings[i:min(i+BATCH_SIZE, n)]
        for j in range(0, n, BATCH_SIZE):
            if j <= i:
                if j == i:
                    continue
                else:
                    continue
                    
            batch_j = embeddings[j:min(j+BATCH_SIZE, n)]
            
            for bi in range(len(batch_i)):
                for bj in range(len(batch_j)):
                    idx_i = i + bi
                    idx_j = j + bj
                    if idx_i != idx_j:  # 自分自身との比較を避ける
                        dist = cosine(embeddings[idx_i], embeddings[idx_j])
                        if dist < min_distance:
                            min_distance = dist
                            closest_pair = (idx_i, idx_j)
    
    return closest_pair, min_distance

def extract_merge_info(children, distances, comments, max_merges=50):
    """クラスタ併合情報を抽出する"""
    print(f"Extracting information about the first {max_merges} cluster merges...")
    merges = []
    
    sorted_indices = np.argsort(distances)
    sorted_children = children[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    for i in range(min(max_merges, len(sorted_distances))):
        child1, child2 = sorted_children[i]
        distance = sorted_distances[i]
        
        if child1 < len(comments):
            text1 = comments[child1]
            id1 = child1
        else:
            text1 = f"Cluster #{child1 - len(comments)}"
            id1 = child1
            
        if child2 < len(comments):
            text2 = comments[child2]
            id2 = child2
        else:
            text2 = f"Cluster #{child2 - len(comments)}"
            id2 = child2
            
        merges.append({
            'index': i,
            'id1': id1,
            'id2': id2,
            'text1': text1,
            'text2': text2,
            'distance': distance
        })
    
    return merges

def save_results(closest_pair, min_distance, merges, comments, output_dir):
    """結果をファイルに保存する"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/closest_pair.txt", 'w', encoding='utf-8') as f:
        idx1, idx2 = closest_pair
        f.write(f"最も近いペア（距離: {min_distance:.6f}）:\n")
        f.write(f"テキスト1: {comments[idx1]}\n")
        f.write(f"テキスト2: {comments[idx2]}\n")
    
    df = pd.DataFrame(merges)
    df.to_csv(f"{output_dir}/cluster_merges.csv", index=False, encoding='utf-8')
    
    with open(f"{output_dir}/cluster_merges.txt", 'w', encoding='utf-8') as f:
        f.write(f"クラスタ併合情報（最初の{len(merges)}件）:\n\n")
        for merge in merges:
            f.write(f"併合 #{merge['index'] + 1} （距離: {merge['distance']:.6f}）:\n")
            f.write(f"  テキスト1: {merge['text1'][:200]}{'...' if len(merge['text1']) > 200 else ''}\n")
            f.write(f"  テキスト2: {merge['text2'][:200]}{'...' if len(merge['text2']) > 200 else ''}\n")
            f.write("\n")

def main():
    comments, ids = load_data(CSV_PATH)
    
    embeddings = create_embeddings(comments, MODEL_NAME, BATCH_SIZE)
    
    closest_pair, min_distance = find_closest_pair(embeddings)
    
    labels, distances, children = perform_clustering(embeddings, DISTANCE_THRESHOLD)
    
    merges = extract_merge_info(children, distances, comments, max_merges=50)
    
    save_results(closest_pair, min_distance, merges, comments, OUTPUT_DIR)
    
    print("Processing completed successfully!")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
