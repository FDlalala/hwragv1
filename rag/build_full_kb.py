import warnings
warnings.filterwarnings("ignore")
import json
import os
import numpy as np
from multiprocessing import Pool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ========== 配置 ==========
CASE_JSON_PATH = "./cases.json"
MODEL_PATH     = "./bge-small-zh-v1.5"
CHROMA_DIR     = "./chroma_cases"
NUM_GPUS       = 8      # 使用的 GPU 数量
BATCH_SIZE     = 512    # 每张卡每批编码数量
WRITE_BATCH    = 2000   # 写入 Chroma 的批次大小

# ========== 第一步：读取 JSON 案例库 ==========
print("读取案例库 JSON...")
with open(CASE_JSON_PATH, "r", encoding="utf-8") as f:
    case_data = json.load(f)

documents = []
skipped   = []

for case_id, case_info in case_data.items():
    case_name = case_info.get("case_name", "")
    full_text = "\n".join(case_info.get("text", [])).strip()
    if full_text:
        documents.append(Document(
            page_content=full_text,
            metadata={"case_id": case_id, "case_name": case_name, "source": CASE_JSON_PATH}
        ))
    else:
        skipped.append(case_id)

print(f"成功加载: {len(documents)} 个案例，跳过: {len(skipped)} 个")

# ========== 第二步：多进程多卡并行计算 Embedding ==========
def embed_worker(args):
    """子进程：在指定 GPU 上计算分配到的文本的 embedding"""
    gpu_id, texts = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 每个子进程独立加载模型到对应 GPU
    emb_model = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": True},
    )
    vectors = emb_model.embed_documents(texts)
    return vectors  # list of list[float]

texts     = [doc.page_content for doc in documents]
metadatas = [doc.metadata     for doc in documents]
total     = len(texts)

print(f"\n使用 {NUM_GPUS} 张 GPU 并行计算 Embedding，共 {total} 条...")

# 将文本均匀切分给各 GPU
chunks = [texts[i::NUM_GPUS] for i in range(NUM_GPUS)]
args   = [(gpu_id, chunk) for gpu_id, chunk in enumerate(chunks) if chunk]

with Pool(processes=len(args)) as pool:
    results = pool.map(embed_worker, args)

# 按原始顺序还原 embedding（交错切分的逆操作）
all_embeddings = [None] * total
for gpu_id, vecs in enumerate(results):
    for local_idx, vec in enumerate(vecs):
        original_idx = local_idx * NUM_GPUS + gpu_id
        if original_idx < total:
            all_embeddings[original_idx] = vec

print(f"Embedding 计算完成，共 {len(all_embeddings)} 条")

# ========== 第三步：批量写入 Chroma（主进程单卡）==========
print("\n初始化 Chroma 向量库...")
# 写入时只需 CPU embedding（实际不会再调用 embed，直接传入预计算向量）
write_embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(embedding_function=write_embeddings, persist_directory=CHROMA_DIR)
vectordb.delete_collection()
vectordb = Chroma(embedding_function=write_embeddings, persist_directory=CHROMA_DIR)

print(f"批量写入 Chroma（每批 {WRITE_BATCH} 条）...")
for i in range(0, total, WRITE_BATCH):
    batch_texts  = texts[i:i+WRITE_BATCH]
    batch_metas  = metadatas[i:i+WRITE_BATCH]
    batch_embeds = all_embeddings[i:i+WRITE_BATCH]
    # 直接传入预计算的 embeddings，Chroma 不会再调用 embed_documents
    vectordb._collection.add(
        embeddings=batch_embeds,
        documents=batch_texts,
        metadatas=batch_metas,
        ids=[str(i + j) for j in range(len(batch_texts))],
    )
    done = min(i + WRITE_BATCH, total)
    print(f"  写入进度: {done}/{total} ({done * 100 // total}%)")

final_count = vectordb._collection.count()
print(f"\n✅ 案例知识库构建完成！共 {final_count} 条记录")
