import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# matplotlib 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지
plt.ion()  # 대화형 모드 활성화

# 데이터 로드
df = pd.read_csv('after_data/combined_data.csv')

# NaN 값을 빈 문자열로 대체
df['제목'] = df['제목'].fillna('')
df['KDC'] = df['KDC'].fillna('')
df['대출월'] = df['대출월'].fillna(0)
df['성별'] = df['성별'].fillna('')
df['연령'] = df['연령'].fillna('')

# 레이블 인코딩 전에 데이터 타입 확인
df['대출월'] = df['대출월'].astype(str)
df['성별'] = df['성별'].astype(str)
df['연령'] = df['연령'].astype(str)

# 텍스트 데이터를 TF-IDF 벡터로 변환
vectorizer = TfidfVectorizer(min_df=2)  # 최소 2번 이상 등장하는 단어만 사용
text_features = df['제목'] + ' ' + df['KDC'].astype(str)
X = vectorizer.fit_transform(text_features)

# 레이블 인코딩
le_month = LabelEncoder()
le_gender = LabelEncoder()
le_age = LabelEncoder()

y_month = le_month.fit_transform(df['대출월'])
y_gender = le_gender.fit_transform(df['성별'])
y_age = le_age.fit_transform(df['연령'])

# 학습, 검증, 테스트 데이터 분리
X_temp, X_test, y_month_temp, y_month_test, y_gender_temp, y_gender_test, y_age_temp, y_age_test = train_test_split(
    X, y_month, y_gender, y_age, test_size=0.2, random_state=42
)

X_train, X_val, y_month_train, y_month_val, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(
    X_temp, y_month_temp, y_gender_temp, y_age_temp, test_size=0.25, random_state=42
)

# KNN 모델 학습 (대출월, 성별, 연령)
knn_month = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
knn_gender = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
knn_age = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

knn_month.fit(X_train, y_month_train)
knn_gender.fit(X_train, y_gender_train)
knn_age.fit(X_train, y_age_train)

def predict_book_preference(title, kdc):
    # 입력 데이터를 동일한 형식으로 변환
    input_text = f"{title} {kdc}"
    input_vector = vectorizer.transform([input_text])
    
    # 예측
    predicted_month = le_month.inverse_transform(knn_month.predict(input_vector))
    predicted_gender = le_gender.inverse_transform(knn_gender.predict(input_vector))
    predicted_age = le_age.inverse_transform(knn_age.predict(input_vector))
    
    # 신뢰도 계산
    month_proba = knn_month.predict_proba(input_vector)[0]
    gender_proba = knn_gender.predict_proba(input_vector)[0]
    age_proba = knn_age.predict_proba(input_vector)[0]
    
    month_confidence = max(month_proba) * 100
    gender_confidence = max(gender_proba) * 100
    age_confidence = max(age_proba) * 100
    
    return (predicted_month[0], month_confidence,
            predicted_gender[0], gender_confidence,
            predicted_age[0], age_confidence)

# 모델 성능 평가
def evaluate_model(model, X_data, y_true):
    y_pred = model.predict(X_data)
    return accuracy_score(y_true, y_pred)

# 검증 세트 성능 평가
print("검증 세트 성능:")
print(f"대출월 예측 정확도: {evaluate_model(knn_month, X_val, y_month_val):.2f}")
print(f"성별 예측 정확도: {evaluate_model(knn_gender, X_val, y_gender_val):.2f}")
print(f"연령 예측 정확도: {evaluate_model(knn_age, X_val, y_age_val):.2f}")

# 테스트 세트 성능 평가
print("\n테스트 세트 성능:")
print(f"대출월 예측 정확도: {evaluate_model(knn_month, X_test, y_month_test):.2f}")
print(f"성별 예측 정확도: {evaluate_model(knn_gender, X_test, y_gender_test):.2f}")
print(f"연령 예측 정확도: {evaluate_model(knn_age, X_test, y_age_test):.2f}")

# 예측 예시
title = "해리포터와 마법사의 돌"
kdc = "813.7"

month, month_conf, gender, gender_conf, age, age_conf = predict_book_preference(title, kdc)
print(f"\n예측 결과:")
print(f"예상 인기 대출월: {month}월 (신뢰도: {month_conf:.1f}%)")
print(f"예상 선호 성별: {gender} (신뢰도: {gender_conf:.1f}%)")
print(f"예상 선호 연령대: {age} (신뢰도: {age_conf:.1f}%)")

# t-SNE를 사용하여 고차원 데이터를 2차원으로 축소
print("\n데이터 시각화를 위한 차원 축소 중...")
tsne = TSNE(
    n_components=2,
    random_state=42,
    init='pca',
    learning_rate='auto',
    perplexity=30,
    n_iter=1000
)
X_2d = tsne.fit_transform(X.toarray())

# 시각화 함수 정의
def visualize_books(X_2d, labels, title, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
    plt.title(title)
    plt.colorbar(scatter)
    plt.draw()
    plt.pause(1)  # 1초간 표시
    input("Press Enter to continue...")  # 사용자 입력 대기
    plt.close()

# 대출월, 성별, 연령별로 시각화
print("\n데이터 시각화 중...")
visualize_books(X_2d, y_month, '도서 분포 (대출월 기준)')
visualize_books(X_2d, y_gender, '도서 분포 (성별 기준)')
visualize_books(X_2d, y_age, '도서 분포 (연령 기준)')

# 특정 KDC 분류별로 시각화
kdc_categories = df['KDC'].astype(str).str[0]  # KDC 대분류
visualize_books(X_2d, kdc_categories.astype(int), '도서 분포 (KDC 대분류 기준)')

# 가장 많이 등장하는 단어들 시각화
def plot_top_words(vectorizer, n_top=20):
    feature_names = vectorizer.get_feature_names_out()
    total_weights = np.sum(X.toarray(), axis=0)
    top_indices = total_weights.argsort()[-n_top:][::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_top), total_weights[top_indices])
    plt.xticks(range(n_top), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.title(f'상위 {n_top}개 단어')
    plt.tight_layout()
    plt.show()

print("\n자주 등장하는 단어 분석...")
plot_top_words(vectorizer)

# 입력된 책과 가장 유사한 책들을 시각화하는 함수
def visualize_similar_books(title, kdc, n_neighbors=5):
    input_text = f"{title} {kdc}"
    input_vector = vectorizer.transform([input_text])
    
    # 코사인 유사도 계산
    similarities = X.dot(input_vector.T).toarray().flatten()
    top_indices = similarities.argsort()[-n_neighbors:][::-1]
    
    print(f"\n'{title}'와(과) 가장 유사한 책들:")
    for idx in top_indices:
        similarity = similarities[idx] * 100
        print(f"제목: {df['제목'].iloc[idx]}, 유사도: {similarity:.1f}%")

# 예시 책과 유사한 책들 찾기
print("\n유사한 책 검색 중...")
visualize_similar_books(title, kdc)

# 프로그램 종료 시 모든 창 닫기
plt.close('all')
