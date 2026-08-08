# Hướng dẫn xây dataset mới cho Light-HGGEP

Tài liệu này mô tả **điều kiện bắt buộc** khi tạo một `torch.utils.data.Dataset` cho model `LightHGGEP`, dựa trên hợp đồng dữ liệu mà model đang kỳ vọng.

> Nếu chỉ giữ đúng các mục dưới, **model KHÔNG cần sửa gì** khi chuyển sang dữ liệu mới.

---

## 1. Model kỳ vọng gì (hợp đồng bắt buộc)

`forward(x, positions, section_name, local_indices)` — file `models/LightHGGEP.py:122`

| Tham số | Shape | Ý nghĩa | Bắt buộc? |
|---|---|---|---|
| `x` | `(B, 3, 224, 224)` | Patch RGB 224×224, **ImageNet normalize** | ✔ |
| `positions` | `(B, 2)` | Tọa độ spot | ✖ (model không dùng, chỉ để khớp vị trí tuple) |
| `section_name` | `str` | Tên section, khóa tra graph `model.A_norm_cache` | ✔ |
| `local_indices` | `(B,)` | Index **hàng/cột trong A_norm** của section đó | ✔ |

Target `exp`: `(B, n_genes)`, **log library-size normalize**, thứ tự gene khớp `gene_set`.

### Tuple `__getitem__` phải trả

- **Train** (5 phần tử): `(patch, positions, exp, section_name, local_idx)`
  - khớp `training_step` (`LightHGGEP.py:171`)
- **Test** (6 phần tử): `(patch, positions, exp, centers, section_name, local_idx)`
  - khớp `test_step` (`LightHGGEP.py:190`) — `centers` chèn giữa
- `section_collate_fn` phân biệt train/test bằng `len(batch[0]) == 5` → **thứ tự phần tử không được đổi**

---

## 2. Bốn bất biến quan trọng (sai là hỏng âm thầm, không báo lỗi)

### 2.1 `local_idx` phải khớp hàng của A_norm

A_norm được build theo thứ tự dòng của `loc_dict`/`meta`. `__getitem__` trả `idx` = vị trí dòng trong section.
Model dùng `A_norm_full[local_indices][:, local_indices]` (`LightHGGEP.py:156`) → **spot nào, hàng nấy**. Đổi thứ tự spot trong section mà không rebuild graph = graph sai hoàn toàn.

### 2.2 Mỗi batch chỉ chứa 1 section

Model nhận **1** `section_name`, dùng chung 1 A_norm cho cả batch.
Dataset trả `section_name` per-sample. Việc **gom 1 section/batch do `SectionBatchSampler` đảm bảo** (trong `run_pipeline.py`), dataset không tự làm.

### 2.3 `gene_set` giống nhau giữa train/test

`n_genes` phải đồng nhất. Nếu gene list chọn từ dữ liệu, phải tính **trước khi chia train/test** để cả hai instance ra cùng bộ gene.

### 2.4 Gene phải tồn tại ở MỌI section

`m[gene_set]` (chọn cột) sẽ `KeyError` nếu gene vắng mặt ở section nào đó.
→ Chọn gene từ **intersection gene của tất cả section**, rồi mới lấy top-N.

---

## 3. Checklist — dataset mới cần có

```
[ ] 1. __getitem__ trả tuple 5 (train) / 6 (test), đúng thứ tự
[ ] 2. local_idx = index dòng trong section, CÙNG thứ tự khi build A_norm
[ ] 3. exp = log(library_size_normalize(counts)) với gene_set
[ ] 4. gene_set cố định, giống train/test, đủ n_genes
[ ] 5. gene_set chọn từ intersection mọi section (tránh KeyError)
[ ] 6. A_norm_cache per section: A_norm = D^(-1/2) (A+I) D^(-1/2), K-NN
[ ] 7. Normalize ảnh ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
[ ] 8. Patch 224×224 crop quanh pixel tâm
[ ] 9. Dataset build graph nhưng KHÔNG tự gắn vào model
```

**Điểm 9**: graph do dataset xây trong `self.A_norm_cache`, nhưng phải có đoạn ở pipeline:

```python
for section, A_norm in dataset.A_norm_cache.items():
    model.set_graph(section, torch.from_numpy(A_norm).float())
```

Không có đoạn này → model forward bỏ qua nhánh SGC, chạy vẫn được nhưng mất thông tin không gian.

---

## 4. Pipeline (run_pipeline.py) giúp gì — dataset không cần làm

| Việc | Do ai |
|---|---|
| Crop patch, augment, normalize | Dataset |
| Build A_norm per section | Dataset (`_build_graphs`) |
| Gắn A_norm vào model | Pipeline: `model.set_graph(...)` |
| 1 section/batch | `SectionBatchSampler` |
| Giữ `section_name` là str (không thành list) | `section_collate_fn` |

`SectionBatchSampler` + `section_collate_fn` trong `run_pipeline.py` là **bắt buộc** — mỗi cái vá 1 lỗi ngầm:
- Sampler: trộn nhiều section 1 batch → graph sai
- Collate: default_collate biến `section_name` str thành list → `TypeError: unhashable`

---

## 5. Mẫu triển khai (kế thừa để giữ toàn bộ logic cũ)

Cách an toàn nhất: **kế thừa `LightHGGEP_HER2ST`** và chỉ override phần gene selection — giống `LightHGGEP_HER2ST_Top250` (`dataset.py:432`):

```python
class MyDataset(LightHGGEP_HER2ST):
    def _select_gene_list(self):
        # 1) Intersection gene của mọi section
        common = None
        for name in self.names:
            g = set(self.get_cnt(name).columns)
            common = g if common is None else (common & g)
        # 2) Tính mean, lấy top-N
        means = {}
        for name in self.names:
            cnt = self.get_cnt(name)
            for gene in common:
                means[gene] = means.get(gene, 0.0) + float(cnt[gene].mean())
        top = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:N]
        return [g for g, _ in top]
```

Kế thừa giữ nguyên: crop patch, augment, normalize, exp log-normalize, `_build_graphs`, `__getitem__`, `local_idx`.

Nếu dữ liệu mới **khác hoàn toàn** (nguồn khác, tọa độ khác, count format khác) → viết lại dataset từ đầu nhưng vẫn phải giữ **4 bất biến + 9 mục checklist** ở trên.

---

## 6. Khi thêm vào run_pipeline.py

```python
from dataset import MyDataset

# 1) Chọn class theo --datasets
DATASET_CLASS = MyDataset if DATASET == 'mine' else LightHGGEP_HER2ST

# 2) n_genes tự lấy từ gene_set (nếu để None)
if N_GENES is None:
    N_GENES = len(DATASET_CLASS(train=True, fold=FOLD, k_neighbors=K_NEIGHBORS).gene_set)

# 3) Gắn graph từ dataset → model (bắt buộc)
for section, A_norm in train_dataset.A_norm_cache.items():
    model.set_graph(section, torch.from_numpy(A_norm).float())
```

---

## 7. Lỗi thường gặp khi làm dataset mới

| Triệu chứng | Nguyên nhân |
|---|---|
| `KeyError: 'GENEX not in index'` | Gene không thuộc intersection, hoặc thiếu ở 1 section |
| PCC train ~0 nhưng test cao | Bình thường — batch train 32 spot ngẫu nhiên, graph cắt rời (xem thêm mục 8) |
| `TypeError: unhashable type 'list'` | Thiếu `section_collate_fn` |
| Graph sai không báo lỗi | `local_idx` không khớp hàng A_norm, hoặc batch chứa nhiều section |
| Số gen test ≠ số gen model | `gene_set` train/test lệch, hoặc `n_genes` không đồng nhất |

---

## 8. Lưu ý về chỉ số PCC khi theo dõi train

`train_pcc` (log từ `training_step`) tính Pearson trên **từng batch 32 spot** rồi trung bình cả epoch → mẫu nhỏ, nhiễu lớn (dao động -0.2..+0.3), **thường quanh 0**.

`test_pcc` cuối pipeline gom **toàn bộ spot** của 1 section rồi tính 1 lần (`predict.get_R`) → mẫu lớn, ổn định → là số đáng tin.

→ **Đừng dùng `train_pcc` để đánh giá model.** Muốn train_pcc phản ánh đúng: accumulate `y_hat`/`exp` cả epoch, cuối epoch tính PCC 1 lần — cách y hệt test.
