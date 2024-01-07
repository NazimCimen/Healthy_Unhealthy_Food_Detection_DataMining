import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree



# ----------------------------------------------veri ön işleme----------------------------------------------------------------------

# Verimizi bir pandas DataFrame'ine yükledik
df = pd.read_csv('besin.csv')

# 't' değerlerini NaN ile değiştirilir.t bozuk değerleri ifade etmektedir.
df.replace('t', np.nan, inplace=True)

# Sayısal değerleri olan sutünların değrleri numeric cols'a attık.
numeric_cols = ['Grams', 'Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eksik değerleri o sütunun ortalaması ile doldurduk
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()),axis=0)

# --------------------------------------------------------------------------------------------------------------------------------------



# Kullanmak istediğiniz besin ögelerini seçtik
selected_cols = ['Calories', 'Fat']

# Sağlıklı veya sağlıksız etiketlerini oluşturduk (örneğin, kalori ve yağ değerlerine göre)
df['Label'] = np.where((df['Calories'] > 200) & (df['Fat'] > 10), 'Unhealthy', 'Healthy')

# Veri setimizi girdi ve çıktı olarak böldük
X = df[selected_cols]
y = df['Label']

# Veri setimizi eğitim ve test olarak böldük
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Karar ağacı modelini oluşturduk ve eğitik
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0, ccp_alpha=0.01) # MCCP için değişiklikler
dt.fit(X_train, y_train)

# Modelinizi test verisiyle değerlendirdik
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc:.2f}")

# Modelinizi görselleştirdik (örneğin, karar ağacını çizerek)
plt.figure(figsize=(10, 8))
plot_tree(dt, feature_names=selected_cols, class_names=['Healthy', 'Unhealthy'], filled=True)
plt.show()

# Besinlerin isimlerini ve tahmin edilen etiketlerini gösteren bir tablo oluşturduk
table = pd.DataFrame({'Food': df['Food'], 'Predicted': dt.predict(X)})
print(table)
plt.figure(figsize=(8, 8))
plt.pie(table['Predicted'].value_counts(), labels=['Healthy', 'Unhealthy'], autopct='%1.1f%%')
plt.title('Besinlerin Tahmin Edilen Etiketleri')
plt.show()
