# CENG543 Vector Retrieval Benchmark

Bu proje, farklı vektör arama yöntemlerinin (Dense, Sparse, Hybrid) BEIR benchmark datasetleri üzerindeki performansını karşılaştırmaktadır.

## 📊 Sonuçların Özeti

| Dataset | En İyi Model | Recall@10 | QPS |
|---------|-------------|-----------|-----|
| SciFact | minilm+splade (α=0.5) | **0.9429** | 253 |
| ArguAna | minilm+splade (α=0.5) | **0.9226** | 249 |
| DBpedia-Entity | minilm+splade (α=0.5) | **0.4536** | 4.7 |
| FiQA | bge-m3 | **0.4833** | 61 |

---

## 📈 Figure Yorumları

### 1. Speed vs Quality Tradeoff (`speed_vs_quality.png`)

Bu grafik, modellerin **hız (QPS)** ve **kalite (Recall@10)** açısından konumlarını göstermektedir.

**Bölgeler:**
- 🟢 **Yeşil bölge (üst):** Yüksek kalite bölgesi (Recall@10 > 0.8)
- 🔵 **Mavi bölge (sağ):** Yüksek hız bölgesi (QPS > 100)
- **Sağ üst köşe** en ideal konumdur (hem hızlı hem kaliteli)

**Semboller:**
- ⚪ **Yuvarlak:** Dense modeller (MiniLM, BGE-M3)
- 🔺 **Üçgen:** Sparse modeller (SPLADE, BM25)
- ⬛ **Kare:** Hybrid modeller (Dense + Sparse kombinasyonu)

**Gözlemler:**
- `minilm` ve `bge-m3` sağ üst bölgede - hem hızlı hem kaliteli
- `bge-m3-all` yüksek kaliteli ama çok yavaş (sol tarafta)
- Hybrid modeller (`minilm+splade`) en yüksek recall'a ulaşıyor

---

### 2. Latency vs Recall Tradeoff (`latency_vs_recall.png`)

Bu grafik, **gecikme süresi (ms)** ile **kalite** arasındaki tradeoff'u gösterir.

**Yorum:**
- **Sol üst köşe** idealdir (düşük gecikme + yüksek recall)
- `minilm` en düşük gecikme süresine sahip (~1ms) ve yüksek kaliteli
- `bge-m3-all` en yüksek gecikmeye sahip (1000+ ms) - pratik kullanım için uygun değil
- Hybrid modeller orta gecikme süresinde (~10-50ms) en yüksek recall'ı sağlıyor

---

### 3. Recall by Dataset (`recall_by_dataset.png`)

Her dataset için modellerin Recall@10 karşılaştırması.

**Gözlemler:**
- **SciFact ve ArguAna:** Hybrid modeller açık ara önde
- **DBpedia-Entity:** En zor dataset, tüm modeller düşük performans
- **FiQA:** BGE-M3 dense model iyi performans gösteriyor

---

### 4. Alpha Sensitivity (`alpha_sensitivity.png`)

Alpha (α) değerinin hybrid model performansına etkisi.

**Alpha Değeri:**
- α = 0: Sadece Sparse (SPLADE/BM25)
- α = 1: Sadece Dense (MiniLM/Word2Vec)
- α = 0.5: Eşit ağırlık

**Gözlemler:**
- Çoğu dataset için **α = 0.25-0.5** optimal
- Dense ve Sparse'ın birleşimi tek başlarından daha iyi

---

### 5. Model Ranking (`model_ranking.png`)

Tüm datasetler üzerinden ortalama model sıralaması.

**Sonuç:** `minilm+splade` genel olarak en iyi performansı gösteriyor.

---

### 6. BGE-M3-ALL Comparison (`bge_m3_all_comparison.png`)

BGE-M3-ALL baseline ile en iyi hybrid modelin karşılaştırması.

**Önemli Bulgu:**
- Hybrid modeller, BGE-M3-ALL'dan daha yüksek recall sağlıyor
- Hybrid modeller **100-1000x daha hızlı**

---

## 🚀 Deneyi Yeniden Oluşturma (Recreate)

### Gereksinimler
- NVIDIA GPU (A100 önerilir)
- Docker
- 250GB+ disk alanı (embeddingler için)

### Adım 1: Repo'yu Klonla
```bash
git clone https://github.com/erengrkan/CENG543_Eren_Gurkan.git
cd CENG543_Eren_Gurkan
```

### Adım 2: Docker İmajını Oluştur
```bash
docker build -t vector-bench-gpu -f Dockerfile.gpu .
```

### Adım 3: Tüm Deneyi Çalıştır
```bash
./scripts/run_a100.sh
```

Bu script:
1. Datasetleri indirir (BEIR)
2. Tüm modeller için embedding oluşturur
3. Benchmark çalıştırır
4. Sonuçları `results/` klasörüne kaydeder

### Adım 4: Figure'ları Oluştur
```bash
docker run --rm -v $(pwd)/results:/app/results -v $(pwd)/scripts:/app/scripts \
    --entrypoint python vector-bench-gpu /app/scripts/generate_figures.py
```

### Tahmini Süre (A100 GPU)
| Aşama | Süre |
|-------|------|
| Embedding oluşturma | 2-3 saat |
| Benchmark | 30-60 dakika |
| **Toplam** | ~3-4 saat |

---

## ⚠️ BGE-M3-ALL Hakkında Önemli Not

### Neden Tüm Karşılaştırmalarda Yok?

**BGE-M3-ALL** modeli, üç farklı retrieval sinyalini birleştirir:
1. **Dense:** Cosine similarity
2. **Sparse:** Lexical weights
3. **ColBERT:** Token-level MaxSim

**Problem:** ColBERT MaxSim hesaplaması **O(n × m × d)** karmaşıklığındadır:
- n = Doküman sayısı
- m = Query token sayısı
- d = Doküman token sayısı

### Hesaplama Süresi Örnekleri

| Dataset | Doküman | Sorgu | ColBERT Süresi |
|---------|---------|-------|----------------|
| SciFact | 5K | 300 | ~8 dakika |
| ArguAna | 8K | 504 | ~15 dakika |
| FiQA | 57K | 612 | **~4 saat** (tahmini) |
| DBpedia | 100K | 393 | **~6+ saat** (tahmini) |

### Sonuç

- **Küçük datasetler** (SciFact, ArguAna): BGE-M3-ALL test edildi
- **Büyük datasetler** (FiQA, DBpedia): Pratik olmadığı için atlandı
- **Alternatif:** BGE-M3 (sadece dense) kullanıldı - çok daha hızlı, benzer kalite

### Makale için Not

> "BGE-M3-ALL modeli, ColBERT token-level matching kullandığından büyük ölçekli datasetlerde (50K+ doküman) pratik değildir. Bu nedenle karşılaştırmalarda sadece küçük datasetler (SciFact, ArguAna) için dahil edilmiştir."

---

## 📋 Ek Notlar ve Metodoloji

### 1. Hybrid Fusion Stratejisi

Score birleştirme formülü:
```
Final_Score = α × Dense_Score_Norm + (1-α) × Sparse_Score_Norm
```

Where:
- **Min-Max normalizasyon** query bazında uygulanır
- **α = 0.5** genellikle optimal

### 2. HNSW İndeksleme Parametreleri

Dense vektörler için FAISS HNSW:
- `M = 32` (bağlantı sayısı)
- `efConstruction = 200`
- `efSearch = 128`

### 3. Değerlendirme Metrikleri

- **Recall@10:** İlk 10 sonuçta bulunan relevant doküman oranı
- **NDCG@10:** Sıralama kalitesi
- **QPS:** Saniyede işlenen sorgu sayısı
- **Latency P99:** 99. percentile gecikme süresi

### 4. Kullanılan Modeller

| Model | Tip | Boyut | Kaynak |
|-------|-----|-------|--------|
| MiniLM | Dense | 384d | sentence-transformers |
| SPLADE | Sparse | ~30K | naver/splade-cocondenser |
| BM25 | Sparse | - | rank_bm25 |
| Word2Vec | Dense | 300d | Custom trained |
| BGE-M3 | Dense | 1024d | BAAI/bge-m3 |
| BGE-M3-ALL | Multi | 1024d + sparse + colbert | BAAI/bge-m3 |

### 5. Dataset İstatistikleri

| Dataset | Doküman | Query | Domain |
|---------|---------|-------|--------|
| SciFact | 5,183 | 300 | Bilimsel makaleler |
| ArguAna | 8,674 | 1,406 | Tartışma metinleri |
| FiQA | 57,638 | 648 | Finansal sorular |
| DBpedia-Entity | 4.6M (100K kullanıldı) | 400 | Wikipedia entities |

---

## 📁 Proje Yapısı

```
CENG543_Eren_Gurkan/
├── src/vector_experiments/
│   ├── models.py          # Embedding modelleri
│   ├── benchmark.py       # Ana benchmark scripti
│   ├── indexer.py         # HNSW indexleme
│   └── analyze_results.py # Analiz ve görselleştirme
├── scripts/
│   ├── run_a100.sh        # Ana çalıştırma scripti
│   └── generate_figures.py # Figure oluşturma
├── data/
│   ├── raw/               # BEIR datasetleri
│   └── embeddings/        # Ön-hesaplanmış embeddingler
├── results/
│   ├── figures/           # Grafikler
│   ├── tables/            # LaTeX/MD tablolar
│   └── benchmark_*.json   # Ham sonuçlar
├── Dockerfile.gpu         # GPU Docker yapılandırması
└── README.md              # Bu dosya
```

---

## 🔗 Referanslar

1. **BEIR Benchmark:** Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", NeurIPS 2021
2. **SPLADE:** Formal et al., "SPLADE v2: Sparse Lexical and Expansion Model for First Stage Ranking", 2022
3. **BGE-M3:** Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity", 2024
4. **MiniLM:** Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression", 2020

---

**Hazırlayan:** Eren Gürkan  
**Ders:** CENG543 - Information Retrieval  
**Tarih:** 2026
